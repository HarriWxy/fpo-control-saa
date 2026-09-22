"""Numerical and replay contracts for cached/compiled flow actor execution."""

from __future__ import annotations

import copy
from itertools import pairwise

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab")

from isaaclab_fpo.modules import ActorCritic, IMFActorCritic, PMFActorCritic
from isaaclab_fpo.rl_cfg import FpoRslRlPpoActorCriticCfg

POLICY_TYPES = (ActorCritic, IMFActorCritic, PMFActorCritic)


@pytest.fixture
def cpu_execution(monkeypatch):
    """Keep CPU contracts eager without disabling the real CUDA compile test."""
    monkeypatch.setattr(torch, "compile", lambda fn, *args, **kwargs: fn)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(19)
    yield
    torch.set_num_threads(previous_threads)


@pytest.fixture
def full_fp32_precision():
    """Match Joint's small-sigma replay contract without leaking global settings."""
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(previous_precision)


def make_policy(policy_type=PMFActorCritic, **overrides):
    settings = {
        "actor_hidden_dims": [12, 8],
        "critic_hidden_dims": [8, 8],
        "activation": "elu",
        "timestep_embed_dim": 8,
        "actor_scale": 0.7,
        "actor_mlp_output_scale": 0.8,
        "action_perturb_std": 0.0,
        "sampling_steps": 4,
    }
    settings.update(overrides)
    return policy_type(3, 3, 2, FpoRslRlPpoActorCriticCfg(**settings))


def reference_embedding(time, width):
    frequencies = 2 ** torch.arange(width // 2, device=time.device, dtype=time.dtype)
    phase = time * frequencies
    return torch.cat((phase.cos(), phase.sin()), dim=-1)


def reference_integrate(policy, observations, noise):
    """Evaluate the original Euler/mean-flow equations without policy helpers."""
    path = torch.linspace(
        1.0,
        0.0,
        policy.sampling_steps + 1,
        device=observations.device,
        dtype=observations.dtype,
    )
    state = noise
    for current, following in pairwise(path):
        delta = following - current
        current = current.expand(observations.shape[0], 1)
        if isinstance(policy, IMFActorCritic):
            following = following.expand_as(current)
            conditions = (
                reference_embedding(following, policy.timestep_embed_dim),
                reference_embedding(current, policy.timestep_embed_dim),
            )
        else:
            time = (
                -delta.expand_as(current)
                if isinstance(policy, PMFActorCritic)
                else current
            )
            conditions = (reference_embedding(time, policy.timestep_embed_dim),)
        output = policy.actor(torch.cat((observations, *conditions, state), dim=-1))
        mean = policy.mlp_output_scale * output[..., : policy.num_actions]
        if isinstance(policy, PMFActorCritic):
            velocity = (state - mean) / current.clamp_min(policy.pmf_time_eps)
        else:
            velocity = mean
        state = state + delta * velocity
    return policy.actor_scale * state


def reference_transport(policy, observations, noise, actor=None):
    """Original flattened pMF h=1 map; independent of all cache/core helpers."""
    actor = policy.actor if actor is None else actor
    if noise.ndim == 3:
        observations = (
            observations[:, None, :]
            .expand(noise.shape[0], noise.shape[1], -1)
            .reshape(-1, policy.num_actor_obs)
        )
    flat_noise = noise.reshape(-1, policy.num_actions)
    interval = flat_noise.new_ones((flat_noise.shape[0], 1))
    embedding = reference_embedding(interval, policy.timestep_embed_dim)
    output = actor(torch.cat((observations, embedding, flat_noise), dim=-1))
    mean = policy.mlp_output_scale * output[..., : policy.num_actions]
    return (policy.actor_scale * mean).reshape_as(noise)


@pytest.mark.usefixtures("cpu_execution")
@pytest.mark.parametrize("policy_type", POLICY_TYPES)
@pytest.mark.parametrize("steps", (1, 4))
@pytest.mark.parametrize("mode", ("zero", "fixed_seed", "random"))
def test_inference_matches_reference_flow(policy_type, steps, mode):
    policy = make_policy(policy_type, sampling_steps=steps)
    observations = torch.randn(5, 3)
    seed = 101
    torch.manual_seed(seed)
    if mode == "zero":
        noise = observations.new_zeros((5, 2))
    elif mode == "fixed_seed":
        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn(5, 2, generator=generator)
    else:
        noise = torch.randn(5, 2)
    expected = reference_integrate(policy, observations, noise)
    torch.manual_seed(seed)
    actual = policy.act_inference(observations, mode, eval_fixed_seed=seed)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.usefixtures("cpu_execution")
@pytest.mark.parametrize("policy_type", POLICY_TYPES)
def test_act_preserves_training_noise_perturbation_and_eval_mode(policy_type):
    policy = make_policy(policy_type, action_perturb_std=0.13)
    observations = torch.randn(5, 3)
    torch.manual_seed(51)
    expected = reference_integrate(policy, observations, torch.randn(5, 2))
    expected = expected + policy.action_perturb_std * torch.randn_like(expected)
    torch.manual_seed(51)
    torch.testing.assert_close(policy.act(observations), expected)
    policy.eval()
    expected = reference_integrate(policy, observations, torch.zeros(5, 2))
    torch.testing.assert_close(policy.act(observations), expected)


@pytest.mark.usefixtures("cpu_execution")
@pytest.mark.parametrize("time_eps", (0.05, 1.0, 1.5))
def test_pmf_one_step_preserves_denominator_clamp(time_eps):
    policy = make_policy(sampling_steps=1, pmf_time_eps=time_eps)
    observations = torch.randn(5, 3)
    expected = reference_integrate(policy, observations, torch.zeros(5, 2))
    torch.testing.assert_close(policy.act_inference(observations), expected)


@pytest.mark.usefixtures("cpu_execution")
@pytest.mark.parametrize("samples", (None, 1, 4))
@pytest.mark.parametrize("external_actor", (False, True))
def test_transport_matches_reference_outputs_and_gradients(samples, external_actor):
    policy = make_policy()
    actor = copy.deepcopy(policy.actor) if external_actor else policy.actor
    reference_actor = copy.deepcopy(actor)
    # Strided inputs catch accidental assumptions that view/flatten is free.
    observations = torch.randn(5, 6)[:, ::2].detach().requires_grad_()
    shape = (5, 4) if samples is None else (5, samples, 4)
    noise = torch.randn(*shape)[..., ::2].detach().requires_grad_()
    reference_obs = observations.detach().clone().requires_grad_()
    reference_noise = noise.detach().clone().requires_grad_()
    expected = reference_transport(
        policy, reference_obs, reference_noise, reference_actor
    )
    actual = policy.transport_actions(
        observations, noise, actor=actor if external_actor else None
    )
    torch.testing.assert_close(actual, expected)
    weights = torch.randn_like(expected)
    actual_loss = (actual * weights).sum() + actual.square().sum()
    expected_loss = (expected * weights).sum() + expected.square().sum()
    actual_gradients = torch.autograd.grad(
        actual_loss, (observations, noise, *actor.parameters())
    )
    expected_gradients = torch.autograd.grad(
        expected_loss, (reference_obs, reference_noise, *reference_actor.parameters())
    )
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.usefixtures("cpu_execution")
def test_transport_honors_frozen_old_actor_after_policy_changes():
    policy = make_policy()
    old_actor = copy.deepcopy(policy.actor).requires_grad_(False)
    observations, noise = torch.randn(5, 3), torch.randn(5, 4, 2)
    with torch.no_grad():
        expected = reference_transport(policy, observations, noise, old_actor)
        policy.actor[-1].bias[:2].add_(0.3)
        actual = policy.transport_actions(observations, noise, actor=old_actor)
        current = policy.transport_actions(observations, noise)
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(current, expected)


@pytest.mark.usefixtures("cpu_execution")
def test_deepcopied_compiled_transport_uses_copied_policy_scales(monkeypatch):
    def compile_as_function(function, *args, **kwargs):
        # Like torch.compile, return a function: deepcopy preserves its closure.
        # Returning the bound method itself would hide a stale-self regression.
        def execute(*inputs, **kwinputs):
            return function(*inputs, **kwinputs)

        return execute

    monkeypatch.setattr(torch, "compile", compile_as_function)
    original = make_policy()
    copied = copy.deepcopy(original)
    original.actor_scale *= 3.0
    copied.actor_scale *= 0.5
    copied.mlp_output_scale *= 0.25
    with torch.no_grad():
        copied.actor[-1].bias[:2].add_(0.3)
    observations, noise = torch.randn(5, 3), torch.randn(5, 4, 2)
    # Route the public CPU call through the stored compiler wrapper; this checks
    # the CUDA dispatch's ownership contract without needing a CUDA device.
    copied._transport_core = copied._compiled_transport
    torch.testing.assert_close(
        copied.transport_actions(observations, noise),
        reference_transport(copied, observations, noise),
    )


@pytest.mark.usefixtures("cpu_execution")
@pytest.mark.parametrize("policy_type", POLICY_TYPES)
def test_caches_preserve_legacy_checkpoint_and_strict_load(policy_type):
    policy = make_policy(policy_type)
    legacy_state = {
        **{
            f"actor.{key}": value.clone()
            for key, value in policy.actor.state_dict().items()
        },
        **{
            f"critic.{key}": value.clone()
            for key, value in policy.critic.state_dict().items()
        },
    }
    observations = torch.randn(5, 3)
    policy.act_inference(observations)
    if isinstance(policy, PMFActorCritic):
        policy.transport_actions(observations, torch.randn(5, 4, 2))
    assert policy.state_dict().keys() == legacy_state.keys()
    restored = make_policy(policy_type)
    result = restored.load_state_dict(legacy_state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    torch.testing.assert_close(
        policy.act_inference(observations), restored.act_inference(observations)
    )


@pytest.mark.usefixtures("cpu_execution")
@pytest.mark.parametrize("policy_type", POLICY_TYPES)
def test_caches_follow_dtype_sampling_steps_and_time_floor_changes(policy_type):
    policy = make_policy(policy_type)
    for dtype in (torch.float32, torch.float64, torch.float32):
        policy.to(dtype=dtype)
        observations = torch.randn(5, 3, dtype=dtype)
        for steps in (4, 1, 3, 4):
            policy.sampling_steps = steps
            if isinstance(policy, PMFActorCritic):
                # Mutating time_eps must invalidate any cached velocity denominator.
                policy.pmf_time_eps = 1.5 if steps == 1 else 0.05
                noise = torch.randn(5, 4, 2, dtype=dtype)
                torch.testing.assert_close(
                    policy.transport_actions(observations, noise),
                    reference_transport(policy, observations, noise),
                )
            actual = policy.act_inference(observations)
            expected = reference_integrate(
                policy, observations, observations.new_zeros(5, 2)
            )
            assert actual.dtype == dtype
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.usefixtures("cpu_execution")
@pytest.mark.parametrize("policy_type", POLICY_TYPES)
def test_inference_cache_warmup_allows_later_training(policy_type):
    policy = make_policy(policy_type)
    observations = torch.randn(5, 3)
    with torch.inference_mode():
        policy.act_inference(observations)
        if isinstance(policy, PMFActorCritic):
            policy.transport_actions(observations, torch.randn(5, 4, 2))
    loss = policy.act(observations).square().sum()
    if isinstance(policy, PMFActorCritic):
        loss = (
            loss
            + policy.transport_actions(observations, torch.randn(5, 4, 2))
            .square()
            .sum()
        )
    loss.backward()
    for parameter in policy.actor.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires real CUDA TorchInductor"
)
def test_cuda_compiled_transport_replay_backward_rollback_and_device_migration(
    full_fp32_precision,
):
    """Keep actual compilation enabled to catch graph output and parameter reuse bugs."""
    torch.manual_seed(19)
    policy = make_policy(sampling_steps=1, action_perturb_std=0.02)
    cpu_observations = torch.randn(5, 3)
    with torch.inference_mode():
        policy.act_inference(cpu_observations)
    policy.cuda()
    observations = cpu_observations.cuda()
    old_actor = copy.deepcopy(policy.actor).requires_grad_(False)
    state = copy.deepcopy(policy.state_dict())
    noise = torch.randn(5, 4, 2, device="cuda")
    retained = []
    with torch.inference_mode():
        for index in range(5):
            current_noise = noise + index * 0.1
            actual = policy.transport_actions(observations, current_noise)
            expected = reference_transport(policy, observations, current_noise)
            retained.append((actual, expected.clone()))
        for actual, expected in retained:
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)

    # Rollout and replay use different GEMM batch sizes. At sigma=0.02, a
    # tiny mean error can become a meaningful density/importance-ratio error.
    rollout_observations = torch.randn(15, 3, device="cuda")
    with torch.no_grad():
        samples = [
            policy.sample_transport(part) for part in rollout_observations.split(5)
        ]
        actions, latents, means, log_probs = (
            torch.cat(parts) for parts in zip(*samples)
        )
        replay_means = policy.transport_actions(rollout_observations, latents)
        replay_log_probs = policy.conditional_action_log_prob(actions, replay_means)
        # These are the tolerances enforced by FSPPOJoint._validate_replay.
        torch.testing.assert_close(replay_means, means, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(replay_log_probs, log_probs, rtol=1e-4, atol=1e-4)

    # Both forwards remain live until backward, as happens with joint replay and map loss.
    reference_actor = copy.deepcopy(policy.actor)
    actual_loss = policy.transport_actions(observations, noise).square().mean()
    actual_loss = (
        actual_loss
        + policy.transport_actions(observations, noise[:, 0]).square().mean()
    )
    expected_loss = (
        reference_transport(policy, observations, noise, reference_actor)
        .square()
        .mean()
    )
    expected_loss = (
        expected_loss
        + reference_transport(policy, observations, noise[:, 0], reference_actor)
        .square()
        .mean()
    )
    actual_loss.backward()
    expected_loss.backward()
    for actual_parameter, expected_parameter in zip(
        policy.actor.parameters(), reference_actor.parameters()
    ):
        torch.testing.assert_close(
            actual_parameter.grad, expected_parameter.grad, atol=2e-5, rtol=2e-4
        )

    with torch.no_grad():
        policy.actor[-1].bias[:2].add_(0.3)
        changed = policy.transport_actions(observations, noise)
        old = policy.transport_actions(observations, noise, actor=old_actor)
        assert not torch.allclose(changed, old)
        torch.testing.assert_close(
            old,
            reference_transport(policy, observations, noise, old_actor),
            atol=2e-5,
            rtol=2e-4,
        )
        policy.load_state_dict(state, strict=True)
        restored = policy.transport_actions(observations, noise)
        torch.testing.assert_close(restored, old, atol=2e-5, rtol=2e-4)

    policy.cpu()
    torch.testing.assert_close(
        policy.act_inference(cpu_observations),
        reference_integrate(policy, cpu_observations, torch.zeros(5, 2)),
    )
