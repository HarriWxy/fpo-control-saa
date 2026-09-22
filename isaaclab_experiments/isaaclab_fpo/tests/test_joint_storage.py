# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused CPU contracts for the explicit joint-policy storage interface."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab")

from isaaclab_fpo.modules import PMFActorCritic
from isaaclab_fpo.rl_cfg import FpoRslRlPpoActorCriticCfg
from isaaclab_fpo.storage import JointRolloutStorage


@pytest.fixture(autouse=True)
def disable_compile(monkeypatch):
    """Keep the small CPU tests out of TorchInductor/CUDA graph setup."""
    monkeypatch.setattr(torch, "compile", lambda fn, *args, **kwargs: fn)
    torch.set_num_threads(1)


def make_policy_cfg(action_perturb_std: float = 0.2) -> FpoRslRlPpoActorCriticCfg:
    """Build a compact, one-NFE pMF policy configuration for CPU tests."""
    return FpoRslRlPpoActorCriticCfg(
        actor_hidden_dims=[8, 8],
        critic_hidden_dims=[8, 8],
        activation="elu",
        actor_scale=0.5,
        action_perturb_std=action_perturb_std,
        sampling_steps=1,
        cfm_loss_reduction="mean",
    )


def make_transition(step: int, num_envs: int) -> JointRolloutStorage.Transition:
    """Create one identifiable joint transition for storage tests."""
    identifiers = torch.arange(num_envs, dtype=torch.float32) + step * num_envs
    transition = JointRolloutStorage.Transition()
    transition.observations = torch.stack(
        [identifiers, identifiers + 10.0, identifiers + 20.0], dim=-1
    )
    transition.privileged_observations = transition.observations + 100.0
    transition.actions = torch.stack([identifiers, -identifiers], dim=-1)
    transition.rewards = identifiers + 1.0
    transition.dones = torch.zeros(num_envs, dtype=torch.bool)
    transition.values = (identifiers / 10.0).unsqueeze(-1)
    transition.action_latent = torch.stack([identifiers, identifiers + 1000.0], dim=-1)
    transition.action_mean = transition.actions + 2000.0
    transition.action_log_prob = (identifiers - 3000.0).unsqueeze(-1)
    return transition


def test_sample_transport_replays_latent_mean_and_conditional_log_prob():
    """The policy exposes a finite conditional Gaussian without a latent prior."""
    torch.manual_seed(11)
    policy = PMFActorCritic(3, 3, 2, make_policy_cfg()).eval()
    observations = torch.tensor([[0.1, -0.2, 0.3], [-0.4, 0.5, 0.6]])

    actions, latent, means, log_prob = policy.sample_transport(observations)

    assert actions.shape == latent.shape == means.shape == (2, 2)
    assert log_prob.shape == (2, 1)
    assert torch.allclose(means, policy.transport_actions(observations, latent))
    assert torch.allclose(log_prob, policy.conditional_action_log_prob(actions, means))
    expected_log_prob = -0.5 * (
        ((actions - means) / 0.2).square() + math.log(2.0 * math.pi * 0.2**2)
    ).sum(dim=-1, keepdim=True)
    assert torch.allclose(log_prob, expected_log_prob)


def test_joint_policy_interface_rejects_a_degenerate_action_density():
    """A deterministic perturbation cannot supply a Gaussian joint ratio."""
    policy = PMFActorCritic(3, 3, 2, make_policy_cfg(action_perturb_std=0.0))
    observations = torch.zeros(2, 3)

    with pytest.raises(ValueError, match="finite, positive"):
        policy.sample_transport(observations)
    with pytest.raises(ValueError, match="finite, positive"):
        policy.conditional_action_log_prob(torch.zeros(2, 2), torch.zeros(2, 2))


def test_joint_storage_copies_data_reuses_gae_and_keeps_all_uneven_batches(
    monkeypatch,
):
    """Joint fields replay exactly while GAE and uneven mini-batches stay sound."""
    storage = JointRolloutStorage(
        num_envs=2,
        num_transitions_per_env=3,
        obs_shape=[3],
        privileged_obs_shape=[3],
        actions_shape=[2],
    )
    first_transition = make_transition(0, num_envs=2)
    expected_latent = first_transition.action_latent.clone()
    expected_mean = first_transition.action_mean.clone()
    expected_log_prob = first_transition.action_log_prob.clone()
    storage.add_transitions(first_transition)

    # A caller reuses and mutates its transition object after this copy.
    first_transition.action_latent.zero_()
    first_transition.action_mean.zero_()
    first_transition.action_log_prob.zero_()
    assert torch.equal(storage.action_latents[0], expected_latent)
    assert torch.equal(storage.action_means[0], expected_mean)
    assert torch.equal(storage.action_log_probs[0], expected_log_prob)

    for step in range(1, 3):
        storage.add_transitions(make_transition(step, num_envs=2))

    assert storage.n_samples_per_action == 0
    assert storage.cfm_loss_eps.shape[2] == 0
    last_values = torch.tensor([[0.6], [0.7]])
    storage.compute_returns(last_values, gamma=0.9, lam=0.95, normalize_advantage=False)

    expected_advantages = torch.zeros_like(storage.advantages)
    running_advantage = torch.zeros_like(last_values)
    for step in reversed(range(storage.num_transitions_per_env)):
        next_values = (
            last_values
            if step == storage.num_transitions_per_env - 1
            else storage.values[step + 1]
        )
        not_done = 1.0 - storage.dones[step].float()
        delta = (
            storage.rewards[step] + not_done * 0.9 * next_values - storage.values[step]
        )
        running_advantage = delta + not_done * 0.9 * 0.95 * running_advantage
        expected_advantages[step] = running_advantage
    assert torch.allclose(storage.advantages, expected_advantages)
    assert torch.allclose(storage.returns, storage.values + expected_advantages)

    permutations = [torch.arange(6), torch.arange(5, -1, -1)]
    calls: list[int] = []

    def fake_randperm(size: int, device=None) -> torch.Tensor:
        calls.append(size)
        return permutations[len(calls) - 1].to(device=device)

    monkeypatch.setattr(torch, "randperm", fake_randperm)
    batches = list(storage.mini_batch_generator(num_mini_batches=4, num_epochs=2))

    expected_keys = {
        "obs",
        "critic_obs",
        "actions",
        "values",
        "advantages",
        "returns",
        "action_latents",
        "action_means",
        "action_log_probs",
    }
    assert calls == [6, 6]
    assert len(batches) == 8
    assert [batch["obs"].shape[0] for batch in batches] == [2, 2, 1, 1] * 2
    assert all(set(batch) == expected_keys for batch in batches)
    first_epoch_ids = torch.cat(
        [batch["action_latents"][:, 0] for batch in batches[:4]]
    )
    second_epoch_ids = torch.cat(
        [batch["action_latents"][:, 0] for batch in batches[4:]]
    )
    assert torch.equal(first_epoch_ids, torch.arange(6, dtype=torch.float32))
    assert torch.equal(second_epoch_ids, torch.arange(5, -1, -1, dtype=torch.float32))


def test_joint_storage_rejects_partial_rollouts_and_too_many_mini_batches():
    """The joint replay contract never silently drops incomplete samples."""
    storage = JointRolloutStorage(
        num_envs=1,
        num_transitions_per_env=2,
        obs_shape=[3],
        privileged_obs_shape=[3],
        actions_shape=[2],
    )
    storage.add_transitions(make_transition(0, num_envs=1))

    with pytest.raises(RuntimeError, match="full rollout"):
        next(storage.mini_batch_generator(num_mini_batches=1, num_epochs=1))

    storage.add_transitions(make_transition(1, num_envs=1))
    with pytest.raises(ValueError, match="cannot exceed"):
        next(storage.mini_batch_generator(num_mini_batches=3, num_epochs=1))
