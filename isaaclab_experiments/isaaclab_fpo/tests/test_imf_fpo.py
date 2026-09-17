"""Focused CPU contracts for the experimental Improved MeanFlow FPO path."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab")

from isaaclab_fpo.algorithms import FPO, IMFFPO
from isaaclab_fpo.modules import ActorCritic, IMFActorCritic
from isaaclab_fpo.rl_cfg import (
    FpoRslRlPpoActorCriticCfg,
    FpoRslRlPpoAlgorithmCfg,
)


@pytest.fixture(autouse=True)
def disable_compile(monkeypatch):
    """Keep small CPU tests out of TorchInductor/CUDA-graph setup."""
    monkeypatch.setattr(torch, "compile", lambda fn, *args, **kwargs: fn)
    torch.set_num_threads(1)


def make_policy_cfg() -> FpoRslRlPpoActorCriticCfg:
    return FpoRslRlPpoActorCriticCfg(
        actor_hidden_dims=[8, 8],
        critic_hidden_dims=[8, 8],
        activation="elu",
        actor_scale=0.5,
        action_perturb_std=0.0,
        sampling_steps=1,
        cfm_loss_reduction="mean",
        imf_fm_proportion=0.5,
    )


def make_algorithm_cfg(class_name: str) -> FpoRslRlPpoAlgorithmCfg:
    return FpoRslRlPpoAlgorithmCfg(
        class_name=class_name,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1.0e-3,
        weight_decay=0.0,
        schedule="fixed",
        knn_entropy_coef=0.0,
        n_samples_per_action=2,
        ema_decay=0.0,
        normalize_advantage=False,
        cfm_loss_clamp=-1.0,
    )


def collect_two_steps(algorithm: FPO, obs_dim: int = 3, action_dim: int = 2):
    algorithm.init_storage(
        num_envs=2,
        num_transitions_per_env=2,
        actor_obs_shape=[obs_dim],
        critic_obs_shape=[obs_dim],
        actions_shape=[action_dim],
    )
    obs = torch.randn(2, obs_dim)
    for _ in range(2):
        actions = algorithm.act(obs, obs)
        assert actions.shape == (2, action_dim)
        algorithm.process_env_step(
            torch.randn(2), torch.zeros(2, dtype=torch.bool), {}
        )
        obs = torch.randn(2, obs_dim)
    algorithm.compute_returns(obs)


def test_legacy_fpo_still_updates_after_meanflow_storage_extension():
    """The extra stored endpoint must not change legacy generator unpacking."""
    policy = ActorCritic(3, 3, 2, make_policy_cfg())
    algorithm = FPO(policy, make_algorithm_cfg("FPO"), device="cpu")

    collect_two_steps(algorithm)
    result = algorithm.update()

    assert torch.isfinite(torch.tensor(result["surrogate_loss"]))
    assert torch.isfinite(torch.tensor(result["metrics"]["cfm_score"]))


def test_imf_score_replays_and_update_is_finite():
    """Rollout and update must use the identical stored eps/r/t triplet."""
    policy = IMFActorCritic(3, 3, 2, make_policy_cfg())
    algorithm = IMFFPO(policy, make_algorithm_cfg("IMFFPO"), device="cpu")
    algorithm.init_storage(
        num_envs=2,
        num_transitions_per_env=2,
        actor_obs_shape=[3],
        critic_obs_shape=[3],
        actions_shape=[2],
    )

    obs = torch.randn(2, 3)
    algorithm.act(obs, obs)
    old_score = algorithm.transition.initial_cfm_loss.clone()
    replay_score, _, _, _ = algorithm._compute_flow_score(
        obs,
        algorithm.transition.actions,
        algorithm.transition.cfm_loss_eps,
        algorithm.transition.cfm_loss_t,
        algorithm.transition.meanflow_loss_r,
    )
    assert torch.allclose(old_score, replay_score.detach(), atol=1.0e-6, rtol=1.0e-5)

    algorithm.process_env_step(
        torch.randn(2), torch.zeros(2, dtype=torch.bool), {}
    )
    obs = torch.randn(2, 3)
    algorithm.act(obs, obs)
    algorithm.process_env_step(
        torch.randn(2), torch.zeros(2, dtype=torch.bool), {}
    )
    algorithm.compute_returns(torch.randn(2, 3))
    result = algorithm.update()

    assert torch.isfinite(torch.tensor(result["surrogate_loss"]))
    assert torch.isfinite(
        torch.tensor(result["metrics"]["experimental_imf_score"])
    )
    assert "experimental_imf_score/u_loss" in result["metrics"]
    assert "experimental_imf_score/v_loss" in result["metrics"]
    assert "experimental_imf_score/anchor_fraction" in result["metrics"]


def test_imf_boundary_anchor_and_one_nfe_contract():
    """At r=t the JVP term vanishes, and one NFE is a full mean-flow jump."""
    policy = IMFActorCritic(3, 3, 2, make_policy_cfg())
    batch_size, n_samples = 2, 3
    observations = torch.randn(batch_size, 3)
    actions = torch.randn(batch_size, 2)
    eps = torch.randn(batch_size, n_samples, 2)
    t = torch.rand(batch_size, n_samples, 1)

    score, _, _, components = policy.get_imf_loss(
        observations, actions, eps, t, t, return_components=True
    )
    scaled_actions = actions / policy.actor_scale
    x_t = t * eps + (1.0 - t) * scaled_actions[:, None, :]
    target = eps - scaled_actions[:, None, :]
    flat_observations = (
        observations[:, None, :]
        .expand(batch_size, n_samples, -1)
        .reshape(batch_size * n_samples, 3)
    )
    mean_velocity, _ = policy._predict_u_and_v(
        flat_observations,
        x_t.reshape(batch_size * n_samples, 2),
        t.reshape(batch_size * n_samples, 1),
        t.reshape(batch_size * n_samples, 1),
    )
    expected_u_loss = policy._compute_squared_error(
        mean_velocity, target.reshape(batch_size * n_samples, 2)
    ).reshape(batch_size, n_samples)

    assert torch.allclose(
        components["u_loss"], expected_u_loss, atol=1.0e-6, rtol=1.0e-5
    )
    score.mean().backward()
    final_weight_grad = policy.actor[-1].weight.grad
    assert final_weight_grad is not None
    assert final_weight_grad[: policy.num_actions].abs().sum() > 0
    assert final_weight_grad[policy.num_actions :].abs().sum() > 0

    policy.eval()
    seed = 17
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.randn(batch_size, 2, generator=generator)
    expected_action = policy.actor_scale * policy.flow_step(
        observations,
        noise,
        torch.zeros(batch_size, 1),
        torch.ones(batch_size, 1),
    )
    inferred_action = policy.act_inference(
        observations, eval_mode="fixed_seed", eval_fixed_seed=seed
    )
    assert torch.allclose(expected_action, inferred_action, atol=1.0e-6, rtol=1.0e-5)
