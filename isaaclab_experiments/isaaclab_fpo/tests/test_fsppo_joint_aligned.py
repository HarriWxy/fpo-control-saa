"""CPU behavior contracts for the aligned pMF joint-PPO experiment."""

from __future__ import annotations

import copy

import pytest
import torch
from test_fsppo_joint import (
    collect,
    cpu_execution,  # noqa: F401 -- shared autouse CPU fixture
    make_algorithm_cfg,
    make_policy_cfg,
)

from isaaclab_fpo.algorithms.fsppo_joint_aligned import FSPPOJointAligned
from isaaclab_fpo.modules import LearnedStdPMFActorCritic, PMFActorCritic


def make_aligned(*, learn_std=False, **algorithm_options):
    policy = LearnedStdPMFActorCritic(
        3,
        3,
        2,
        make_policy_cfg(
            class_name="LearnedStdPMFActorCritic", learn_action_perturb_std=learn_std
        ),
    )
    cfg = make_algorithm_cfg(
        class_name="FSPPOJointAligned",
        fsppo_joint_enable_budget=False,
        use_clipped_value_loss=True,
        fsppo_joint_entropy_coef=0.01,
        **algorithm_options,
    )
    return FSPPOJointAligned(policy, cfg)


def test_new_scale_policy_preserves_original_flow_map_and_both_pmf_heads():
    torch.manual_seed(11)
    original = PMFActorCritic(3, 3, 2, make_policy_cfg())
    torch.manual_seed(11)
    scaled = LearnedStdPMFActorCritic(
        3, 3, 2, make_policy_cfg(learn_action_perturb_std=True)
    )
    observations = torch.randn(4, 3)
    latents = torch.randn(4, 3, 2)
    assert torch.allclose(
        scaled.transport_actions(observations, latents),
        original.transport_actions(observations, latents),
    )
    h = torch.rand(4, 1)
    expected_heads = original._predict_x_heads(observations, latents[:, 0], h)
    actual_heads = scaled._predict_x_heads(observations, latents[:, 0], h)
    for expected, actual in zip(expected_heads, actual_heads):
        assert torch.equal(actual, expected)
    # The latent map is retained rather than replaced by a Gaussian mean MLP.
    assert not torch.allclose(
        scaled.transport_actions(observations, latents[:, 0]),
        scaled.transport_actions(observations, latents[:, 1]),
    )


@pytest.mark.parametrize("learn_std,aux_coef", [(False, 0.0), (True, 0.01)])
def test_rollout_update_learns_flow_and_optional_std_and_replays_full_density(
    learn_std, aux_coef
):
    algorithm = make_aligned(learn_std=learn_std, fsppo_joint_aux_loss_coef=aux_coef)
    collect(algorithm)
    batch = next(algorithm.storage.mini_batch_generator(1, 1))
    old_std = batch["action_stds"].clone()
    before_std = algorithm.policy.conditional_action_std().detach().clone()
    before_parameters = [
        parameter.detach().clone() for parameter in algorithm.policy.actor.parameters()
    ]
    output = algorithm.update()
    assert all(
        torch.isfinite(torch.as_tensor(value)) for value in output["metrics"].values()
    )
    assert output["metrics"]["joint/accepted_updates"] == 4
    assert any(
        not torch.equal(old, current)
        for old, current in zip(before_parameters, algorithm.policy.actor.parameters())
    )
    current_std = algorithm.policy.conditional_action_std()
    assert torch.equal(before_std, current_std) != learn_std
    assert (output["pmf_aux_loss"] > 0) == (aux_coef > 0)
    current_means = algorithm.policy.transport_actions(
        batch["obs"], batch["action_latents"]
    )
    old_distribution = torch.distributions.Normal(batch["action_means"], old_std)
    current_distribution = torch.distributions.Normal(current_means, current_std)
    expected_ratio = (
        current_distribution.log_prob(batch["actions"])
        - old_distribution.log_prob(batch["actions"])
    ).sum(-1, keepdim=True)
    actual_ratio = (
        algorithm.policy.conditional_action_log_prob(batch["actions"], current_means)
        - batch["action_log_probs"]
    )
    assert torch.allclose(actual_ratio, expected_ratio, atol=2e-6)
    _, statistics = algorithm._batch_loss(batch)
    expected_kl = (
        torch.distributions.kl_divergence(old_distribution, current_distribution)
        .sum(-1)
        .mean()
    )
    assert torch.allclose(statistics["map_kl"], expected_kl, atol=2e-6)
    assert torch.equal(batch["action_stds"], old_std)


def test_adaptive_update_and_resume_retain_scale_optimizer_and_lr_contract():
    algorithm = make_aligned(
        learn_std=True, schedule="adaptive", desired_kl=1e-10, learning_rate=0.01
    )
    collect(algorithm)
    output = algorithm.update()
    assert algorithm.learning_rate < 0.01
    assert output["metrics"]["joint/adaptive_lr_enabled"] == 1
    policy_state = copy.deepcopy(algorithm.policy.state_dict())
    optimizer_state = copy.deepcopy(algorithm.optimizer.state_dict())
    algorithm_state = algorithm.state_dict()
    resumed = make_aligned(
        learn_std=True, schedule="adaptive", desired_kl=1e-10, learning_rate=0.01
    )
    resumed.policy.load_state_dict(policy_state)
    resumed.optimizer.load_state_dict(optimizer_state)
    resumed.load_state_dict(algorithm_state)
    assert resumed.learning_rate == algorithm.learning_rate
    assert all(
        group["lr"] == algorithm.learning_rate
        for group in resumed.optimizer.param_groups
    )
    observations = torch.randn(3, 3)
    latents = torch.randn(3, 2)
    assert torch.equal(
        resumed.policy.transport_actions(observations, latents),
        algorithm.policy.transport_actions(observations, latents),
    )
    assert torch.equal(
        resumed.policy.conditional_action_std(),
        algorithm.policy.conditional_action_std(),
    )
    collect(resumed)
    assert resumed.update()["metrics"]["joint/accepted_updates"] == 4
    incompatible = make_aligned(learn_std=True, schedule="fixed")
    with pytest.raises(ValueError, match="learning-rate contract mismatch"):
        incompatible.load_state_dict(algorithm_state)
