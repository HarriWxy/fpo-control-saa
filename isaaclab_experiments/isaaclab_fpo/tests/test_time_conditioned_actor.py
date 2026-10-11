"""CPU training and replay contracts for the optional time-conditioned actor."""

from __future__ import annotations

import copy
import pickle

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab")

from isaaclab_fpo.modules import ActorCritic, IMFActorCritic, PMFActorCritic
from isaaclab_fpo.rl_cfg import FpoRslRlPpoActorCriticCfg


@pytest.fixture(autouse=True)
def cpu_execution(monkeypatch):
    """Run eager CPU forwards without leaking thread or random-number settings."""
    monkeypatch.setattr(torch, "compile", lambda fn, *args, **kwargs: fn)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(29)
            yield
    finally:
        torch.set_num_threads(previous_threads)


def make_policy(policy_type=ActorCritic):
    return policy_type(
        3,
        3,
        2,
        FpoRslRlPpoActorCriticCfg(
            actor_architecture="time_conditioned_mlp",
            actor_time_condition_dim=12,
            actor_hidden_dims=[12, 8],
            critic_hidden_dims=[8, 8],
            activation="elu",
            timestep_embed_dim=8,
            actor_scale=0.7,
            actor_mlp_output_scale=0.8,
            action_perturb_std=0.0,
            sampling_steps=1,
            cfm_loss_reduction="mean",
        ),
    ).double()


def assert_conditioning_gradients(actor):
    """Every advertised time-conditioning stage must receive a learning signal."""
    stages = [actor.time_encoder, *(block.modulation for block in actor.blocks)]
    for stage in stages:
        gradients = [parameter.grad for parameter in stage.parameters()]
        assert all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.abs().sum() for gradient in gradients) > 0


def test_legacy_pickled_config_restores_mlp_checkpoint_without_new_fields():
    """Older agent.pkl configs must still construct their original MLP policy."""
    cfg = FpoRslRlPpoActorCriticCfg(
        actor_hidden_dims=[12, 8],
        critic_hidden_dims=[8, 8],
        activation="elu",
        action_perturb_std=0.0,
        sampling_steps=2,
    )
    original = ActorCritic(3, 3, 2, cfg)
    observations = torch.randn(2, 3)
    with torch.no_grad():
        expected = original.act_inference(observations)

    # pickle restores the saved instance dictionary without rerunning config
    # initialization; agent.pkl written before these fields has neither key.
    del cfg.__dict__["actor_architecture"]
    del cfg.__dict__["actor_time_condition_dim"]
    restored_cfg = pickle.loads(pickle.dumps(cfg))
    restored = ActorCritic(3, 3, 2, restored_cfg)
    restored.load_state_dict(original.state_dict(), strict=True)

    with torch.no_grad():
        torch.testing.assert_close(restored.act_inference(observations), expected)


def test_cfm_time_only_gradients_and_training_inference_endpoints_agree():
    """Time modulation must learn and agree across 3-D scores and 2-D flow calls."""
    policy = make_policy()
    observations = torch.randn(2, 3, dtype=torch.float64)
    actions = torch.randn(2, 2, dtype=torch.float64)
    # Equal endpoints make the path state independent of t. Consequently any
    # time derivative of this score comes from the actor's time conditions.
    eps = (actions / policy.actor_scale)[:, None, :].expand(2, 3, 2)
    t = torch.tensor(
        [[[0.2], [0.5], [0.8]], [[0.3], [0.6], [0.9]]],
        dtype=torch.float64,
        requires_grad=True,
    )

    score, x1, x0 = policy.get_cfm_loss(observations, actions, eps, t)
    flat_observations = observations[:, None, :].expand(2, 3, 3).reshape(6, 3)
    flat_state = eps.reshape(6, 2)
    flat_t = t.reshape(6, 1)
    inference_x0 = policy.flow_step(
        flat_observations, flat_state, torch.zeros_like(flat_t), flat_t
    )
    inference_x1 = policy.flow_step(
        flat_observations, flat_state, torch.ones_like(flat_t), flat_t
    )

    torch.testing.assert_close(x0.reshape(6, 2), inference_x0)
    torch.testing.assert_close(x1.reshape(6, 2), inference_x1)
    assert torch.isfinite(score).all()
    score.mean().backward()
    assert torch.isfinite(t.grad).all()
    assert t.grad.abs().sum() > 0
    assert_conditioning_gradients(policy.actor)


def test_frozen_time_conditioned_actor_replays_after_current_actor_learns():
    """Optimizing the current transport must preserve a copied old-policy map."""
    policy = make_policy(PMFActorCritic)
    old_actor = copy.deepcopy(policy.actor).requires_grad_(False)
    observations = torch.randn(2, 3, dtype=torch.float64)
    noise = torch.randn(2, 3, 2, dtype=torch.float64)
    optimizer = torch.optim.SGD(policy.actor.parameters(), lr=0.02)

    with torch.no_grad():
        expected_old = policy.transport_actions(observations, noise, actor=old_actor)
    current = policy.transport_actions(observations, noise)
    torch.testing.assert_close(current, expected_old)
    (current - 0.5).square().mean().backward()
    assert_conditioning_gradients(policy.actor)
    optimizer.step()

    with torch.no_grad():
        replayed_old = policy.transport_actions(observations, noise, actor=old_actor)
        learned_current = policy.transport_actions(observations, noise)
    torch.testing.assert_close(replayed_old, expected_old)
    assert not torch.allclose(learned_current, expected_old)
    assert all(parameter.grad is None for parameter in old_actor.parameters())


@pytest.mark.parametrize("policy_type", (IMFActorCritic, PMFActorCritic))
def test_meanflow_conditioning_jvp_matches_trajectory_finite_difference(policy_type):
    """The conditioned actor must retain the time derivative used by MeanFlow."""
    policy = make_policy(policy_type)
    observations = torch.randn(2, 3, dtype=torch.float64)
    actions = torch.randn(2, 2, dtype=torch.float64)
    eps = torch.randn(2, 2, 2, dtype=torch.float64)
    r = torch.tensor([[[0.1], [0.2]], [[0.3], [0.4]]], dtype=torch.float64)
    t = torch.tensor([[[0.6], [0.7]], [[0.8], [0.9]]], dtype=torch.float64)
    get_score = (
        policy.get_imf_loss if policy_type is IMFActorCritic else policy.get_pmf_loss
    )
    score, _, boundary_x0, components = get_score(
        observations, actions, eps, r, t, return_components=True
    )

    flat_observations = observations[:, None, :].expand(2, 2, 3).reshape(4, 3)
    flat_state = (
        t * eps + (1.0 - t) * (actions / policy.actor_scale)[:, None, :]
    ).reshape(4, 2)
    flat_r, flat_t = r.reshape(4, 1), t.reshape(4, 1)
    with torch.no_grad():
        # Public flow_step returns z - (t-r)*u, which provides an independent
        # evaluation of the field without using the score's JVP helper.
        tangent = (flat_state - boundary_x0.reshape(4, 2)) / flat_t

        def velocity(state, time):
            transported = policy.flow_step(flat_observations, state, flat_r, time)
            return (state - transported) / (time - flat_r)

        step = 1.0e-5
        finite_difference = (
            velocity(flat_state + step * tangent, flat_t + step)
            - velocity(flat_state - step * tangent, flat_t - step)
        ) / (2.0 * step)
    torch.testing.assert_close(
        components["jvp_norm"].reshape(4),
        finite_difference.norm(dim=-1),
        atol=1.0e-7,
        rtol=1.0e-5,
    )
    assert components["jvp_norm"].abs().sum() > 0
    assert torch.isfinite(score).all()
    score.mean().backward()
    assert_conditioning_gradients(policy.actor)
