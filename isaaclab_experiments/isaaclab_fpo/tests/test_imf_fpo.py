"""Focused CPU contracts for FPO and its experimental MeanFlow variants."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab")

from isaaclab_fpo.algorithms import FPO, FSPPO, IMFFPO, PMFFPO
from isaaclab_fpo.cli_args import update_fpo_cfg
from isaaclab_fpo.modules import ActorCritic, IMFActorCritic, PMFActorCritic
from isaaclab_fpo.rl_cfg import (
    FpoRslRlPpoActorCriticCfg,
    FpoRslRlPpoAlgorithmCfg,
    FpoRslRlOnPolicyRunnerCfg,
)
from isaaclab_fpo.runners import OnPolicyRunner


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


def test_fpo_score_ratio_sign_and_identity():
    """A lower current loss must increase support for the sampled action."""
    policy = ActorCritic(3, 3, 2, make_policy_cfg())
    algorithm = FPO(policy, make_algorithm_cfg("FPO"), device="cpu")
    old_score = torch.tensor([[1.0, 1.0, 1.0]])
    current_score = torch.tensor([[0.75, 1.0, 1.25]])

    raw_log_ratio, ratio = algorithm._compute_importance_ratio(
        old_score, current_score
    )

    assert torch.allclose(raw_log_ratio, torch.tensor([[0.25, 0.0, -0.25]]))
    assert ratio[0, 0] > 1.0
    assert ratio[0, 1] == 1.0
    assert ratio[0, 2] < 1.0


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


def test_pmf_x_prediction_replays_and_one_nfe_returns_x_head():
    """pMF must train in v-space while one NFE directly returns its x head."""
    policy = PMFActorCritic(3, 3, 2, make_policy_cfg())
    algorithm = PMFFPO(policy, make_algorithm_cfg("PMFFPO"), device="cpu")
    sampled_t, sampled_r = algorithm._sample_flow_score_times(num_envs=2)
    assert torch.all(sampled_r <= sampled_t)
    anchor_fraction = torch.isclose(sampled_r, sampled_t).float().mean()
    assert torch.isclose(anchor_fraction, torch.tensor(0.5))

    observations = torch.randn(2, 3)
    actions = torch.randn(2, 2)
    eps = torch.randn(2, 2, 2)
    t = torch.tensor([[[0.8], [0.6]], [[0.7], [0.9]]])
    r = torch.tensor([[[0.2], [0.6]], [[0.1], [0.4]]])

    score, _, _, components = policy.get_pmf_loss(
        observations, actions, eps, r, t, return_components=True
    )
    assert score.shape == (2, 2)
    assert all(torch.isfinite(value).all() for value in components.values())
    score.mean().backward()
    final_weight_grad = policy.actor[-1].weight.grad
    assert final_weight_grad is not None
    assert final_weight_grad[: policy.num_actions].abs().sum() > 0
    assert final_weight_grad[policy.num_actions :].abs().sum() > 0

    policy.eval()
    noise = torch.randn(2, 2)
    x_mean, _ = policy._predict_x_heads(
        observations, noise, torch.ones(2, 1)
    )
    one_nfe_scaled = policy.flow_step(
        observations, noise, torch.zeros(2, 1), torch.ones(2, 1)
    )
    assert torch.allclose(one_nfe_scaled, x_mean, atol=1.0e-6, rtol=1.0e-5)


def test_pmf_fpo_update_is_finite():
    """The pMF score must satisfy the same rollout/update storage contract."""
    policy = PMFActorCritic(3, 3, 2, make_policy_cfg())
    algorithm = PMFFPO(policy, make_algorithm_cfg("PMFFPO"), device="cpu")
    collect_two_steps(algorithm)
    result = algorithm.update()

    assert torch.isfinite(torch.tensor(result["surrogate_loss"]))
    assert torch.isfinite(
        torch.tensor(result["metrics"]["experimental_pmf_score"])
    )
    assert "experimental_pmf_score/u_loss" in result["metrics"]
    assert "experimental_pmf_score/v_loss" in result["metrics"]


def test_fsppo_same_noise_map_penalty_uses_frozen_rollout_actor():
    """D_map is zero at the snapshot and reacts only through the online map."""
    torch.manual_seed(7)
    policy = PMFActorCritic(3, 3, 2, make_policy_cfg())
    cfg = make_algorithm_cfg("FSPPO")
    cfg.fsppo_map_loss_coef = 2.0
    cfg.fsppo_map_num_samples = 1
    algorithm = FSPPO(policy, cfg, device="cpu")
    algorithm._prepare_update()

    observations = torch.randn(3, 3)
    noise = torch.randn(3, 2, 2)
    with torch.no_grad():
        frozen_actions_before = policy.transport_actions(
            observations, noise[:, :1], actor=algorithm._old_actor
        ).clone()

    regularization, diagnostics = algorithm._compute_policy_regularization(
        observations, noise
    )
    assert torch.equal(regularization, torch.zeros_like(regularization))
    assert torch.equal(
        diagnostics["distance"], torch.zeros_like(diagnostics["distance"])
    )

    # Moving only pMF's terminal x-mean bias by 0.2 changes each public action
    # by actor_scale * 0.2 = 0.1, while the frozen rollout actor stays fixed.
    with torch.no_grad():
        policy.actor[-1].bias[: policy.num_actions].add_(0.2)

    with torch.no_grad():
        frozen_actions_after = policy.transport_actions(
            observations, noise[:, :1], actor=algorithm._old_actor
        )
    assert torch.equal(frozen_actions_before, frozen_actions_after)

    regularization, diagnostics = algorithm._compute_policy_regularization(
        observations, noise
    )
    expected_distance = torch.tensor(2 * (policy.actor_scale * 0.2) ** 2)
    assert torch.allclose(diagnostics["distance"], expected_distance)
    assert torch.allclose(regularization, 2.0 * expected_distance)

    algorithm.optimizer.zero_grad()
    regularization.backward()
    assert policy.actor[-1].bias.grad[: policy.num_actions].abs().sum() > 0
    assert policy.actor[-1].bias.grad[policy.num_actions :].abs().sum() == 0
    assert all(
        parameter.grad is None for parameter in algorithm._old_actor.parameters()
    )

    algorithm.fsppo_map_loss_coef = 0.0
    zero_regularization, zero_coef_diagnostics = (
        algorithm._compute_policy_regularization(observations, noise)
    )
    assert torch.equal(zero_regularization, torch.zeros_like(zero_regularization))
    assert zero_coef_diagnostics["distance"] > 0


def test_fsppo_update_is_finite_and_reports_map_trust_region():
    """The new regularizer participates in the full PMF rollout/update path."""
    policy = PMFActorCritic(3, 3, 2, make_policy_cfg())
    cfg = make_algorithm_cfg("FSPPO")
    cfg.num_learning_epochs = 2
    cfg.fsppo_map_num_samples = 1
    algorithm = FSPPO(policy, cfg, device="cpu")

    collect_two_steps(algorithm)
    result = algorithm.update()

    assert torch.isfinite(torch.tensor(result["total_loss"]))
    assert torch.isfinite(torch.tensor(result["map_trust_region_loss"]))
    assert "experimental_fsppo_pmf_score/u_loss" in result["metrics"]
    assert "map_trust_region/distance" in result["metrics"]
    assert "map_trust_region/distance_per_action_dim" in result["metrics"]
    assert result["metrics"]["map_trust_region/samples_per_action"] == 1.0


def test_fsppo_requires_the_one_nfe_pmf_transport_map():
    """The terminal x head is not the deployed map for multi-step sampling."""
    policy_cfg = make_policy_cfg()
    policy_cfg.sampling_steps = 2
    policy = PMFActorCritic(3, 3, 2, policy_cfg)

    with pytest.raises(ValueError, match="sampling_steps=1"):
        FSPPO(policy, make_algorithm_cfg("FSPPO"), device="cpu")


def test_fsppo_cli_selects_pmf_policy_and_safe_defaults():
    """The standalone selector must not mutate the legacy FPO/PMF choices."""
    agent_cfg = SimpleNamespace(
        seed=42,
        resume=False,
        load_run=".*",
        load_checkpoint="model_.*.pt",
        run_name="",
        experiment_name="unit_test",
        logger="tensorboard",
        policy=SimpleNamespace(class_name="ActorCritic", sampling_steps=64),
        algorithm=SimpleNamespace(
            class_name="FPO", schedule="adaptive", knn_entropy_coef=0.1
        ),
    )
    args = argparse.Namespace(
        seed=None,
        resume=None,
        load_run=None,
        checkpoint=None,
        run_name=None,
        experiment_name=None,
        logger=None,
        log_project_name=None,
        algorithm="fsppo",
    )

    updated = update_fpo_cfg(agent_cfg, args)

    assert updated.policy.class_name == "PMFActorCritic"
    assert updated.algorithm.class_name == "FSPPO"
    assert updated.policy.sampling_steps == 1
    assert updated.algorithm.schedule == "fixed"
    assert updated.algorithm.knn_entropy_coef == 0.0
    assert updated.experiment_name == "unit_test_fsppo"


def test_runner_accepts_only_valid_fsppo_policy_pair():
    """FSPPO shares PMFActorCritic without weakening other pair checks."""

    class FakeEnv:
        num_actions = 2
        num_envs = 2

        def get_observations(self):
            return torch.zeros(self.num_envs, 3), {"observations": {}}

    policy_cfg = make_policy_cfg()
    policy_cfg.class_name = "PMFActorCritic"
    runner_cfg = FpoRslRlOnPolicyRunnerCfg(
        policy=policy_cfg,
        algorithm=make_algorithm_cfg("FSPPO"),
        experiment_name="unit_test",
        num_steps_per_env=2,
    )
    runner = OnPolicyRunner(FakeEnv(), runner_cfg, device="cpu")
    assert isinstance(runner.alg, FSPPO)

    runner_cfg.policy.class_name = "ActorCritic"
    with pytest.raises(ValueError, match="must be selected together"):
        OnPolicyRunner(FakeEnv(), runner_cfg, device="cpu")
