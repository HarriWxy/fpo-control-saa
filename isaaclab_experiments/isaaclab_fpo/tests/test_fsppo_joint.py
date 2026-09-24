"""CPU contracts for exact latent replay and transactional KL-budget updates."""

from __future__ import annotations

import argparse
import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab")

from isaaclab_fpo.algorithms import FSPPOJoint
from isaaclab_fpo.cli_args import update_fpo_cfg
from isaaclab_fpo.modules import PMFActorCritic
from isaaclab_fpo.rl_cfg import (
    FpoRslRlOnPolicyRunnerCfg,
    FpoRslRlPpoActorCriticCfg,
    FpoRslRlPpoAlgorithmCfg,
)
from isaaclab_fpo.runners import OnPolicyRunner


@pytest.fixture(autouse=True)
def cpu_execution(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, *args, **kwargs: fn)
    torch.set_num_threads(1)
    torch.manual_seed(7)


def make_policy_cfg(**overrides):
    settings = dict(
        class_name="PMFActorCritic",
        actor_hidden_dims=[8, 8],
        critic_hidden_dims=[8, 8],
        activation="elu",
        sampling_steps=1,
        action_perturb_std=0.2,
        actor_scale=0.5,
        pmf_adaptive_gradient_norm_p=0.0,
    )
    settings.update(overrides)
    return FpoRslRlPpoActorCriticCfg(**settings)


def make_algorithm_cfg(**overrides):
    settings = dict(
        class_name="FSPPOJoint",
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        schedule="fixed",
        trust_region_mode="ppo",
        n_samples_per_action=2,
        normalize_advantage=False,
        knn_entropy_coef=0.0,
        storage_action_noise_std=0.0,
        ema_decay=0.0,
        fsppo_joint_kl_target=0.02,
        fsppo_joint_probe_size=8,
        fsppo_joint_map_samples=2,
    )
    settings.update(overrides)
    return FpoRslRlPpoAlgorithmCfg(**settings)


def make_algorithm(**overrides):
    return FSPPOJoint(
        PMFActorCritic(3, 3, 2, make_policy_cfg()), make_algorithm_cfg(**overrides)
    )


def collect(algorithm, steps=3, num_envs=2):
    algorithm.init_storage(num_envs, steps, [3], [3], [2])
    for _ in range(steps):
        obs = torch.randn(num_envs, 3)
        algorithm.act(obs, obs)
        algorithm.process_env_step(
            torch.randn(num_envs), torch.zeros(num_envs, dtype=torch.bool), {}
        )
    algorithm.compute_returns(torch.randn(num_envs, 3))


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def test_joint_ratio_equals_gaussian_density_ratio_and_replays_actual_latent():
    alg = make_algorithm()
    collect(alg)
    batch = next(alg.storage.mini_batch_generator(1, 1))
    obs, actions, latent = batch["obs"], batch["actions"], batch["action_latents"]
    old_mean = alg.policy.transport_actions(obs, latent)
    assert torch.allclose(old_mean, batch["action_means"])
    old_logp = alg.policy.conditional_action_log_prob(actions, old_mean)
    assert torch.allclose(old_logp, batch["action_log_probs"])
    with torch.no_grad():
        alg.policy.actor[-1].bias[:2].add_(0.03)
    mean = alg.policy.transport_actions(obs, latent)
    log_ratio = alg.policy.conditional_action_log_prob(actions, mean) - old_logp
    expected = (
        torch.distributions.Normal(mean, alg.sigma).log_prob(actions)
        - torch.distributions.Normal(old_mean, alg.sigma).log_prob(actions)
    ).sum(-1, keepdim=True)
    assert torch.allclose(log_ratio, expected, atol=1e-6)
    gaussian_kl = torch.distributions.kl_divergence(
        torch.distributions.Normal(old_mean, alg.sigma),
        torch.distributions.Normal(mean, alg.sigma),
    ).sum(-1)
    map_kl = ((mean - old_mean) / alg.sigma).square().sum(-1) / 2
    assert torch.allclose(gaussian_kl, map_kl, atol=1e-6)


def test_v_head_cannot_change_joint_ratio_or_map_and_is_not_in_policy_gradient():
    alg = make_algorithm()
    collect(alg)
    alg._old_actor.load_state_dict(alg.policy.actor.state_dict())
    batch = next(alg.storage.mini_batch_generator(1, 1))
    before = alg.policy.transport_actions(batch["obs"], batch["action_latents"])
    with torch.no_grad():
        alg.policy.actor[-1].bias[2:].add_(0.75)
    after = alg.policy.transport_actions(batch["obs"], batch["action_latents"])
    assert torch.equal(before, after)
    logp = alg.policy.conditional_action_log_prob(batch["actions"], after)
    assert torch.allclose(logp, batch["action_log_probs"])
    loss, _ = alg._batch_loss(batch)
    loss.backward()
    assert torch.count_nonzero(alg.policy.actor[-1].bias.grad[2:]) == 0
    assert torch.count_nonzero(alg.policy.actor[-1].weight.grad[2:]) == 0


def test_default_update_never_evaluates_pmf_score(monkeypatch):
    alg = make_algorithm()
    monkeypatch.setattr(
        alg.policy, "get_pmf_loss", lambda *a, **k: pytest.fail("Unexpected JVP score")
    )
    collect(alg)
    results = alg.update()
    assert all(
        torch.isfinite(torch.tensor(value)) for value in results["metrics"].values()
    )
    assert results["metrics"]["joint/accepted_updates"] > 0
    assert results["metrics"]["joint/probe_kl_mean"] <= alg.kl_target
    assert results["pmf_aux_loss"] == 0.0
    assert alg.storage.step == 0


def test_no_budget_mode_applies_all_steps_at_configured_learning_rate():
    alg = make_algorithm(fsppo_joint_enable_budget=False)
    collect(alg)
    results = alg.update()

    assert results["metrics"]["joint/budget_enabled"] == 0.0
    assert results["metrics"]["joint/accepted_updates"] == 4
    assert results["metrics"]["joint/rejected_candidates"] == 0
    assert results["metrics"]["joint/early_stop"] == 0.0
    assert results["metrics"]["joint/last_step_learning_rate"] == 1e-3
    assert results["map_kl_loss"] == 0.0
    assert results["metrics"]["joint/kl_coefficient_used"] == 0.0
    assert results["metrics"]["joint/kl_coefficient_next"] == 0.0


def test_auxiliary_regression_is_separate_positive_loss():
    alg = make_algorithm(fsppo_joint_aux_loss_coef=0.1)
    collect(alg)
    results = alg.update()
    assert results["pmf_aux_loss"] > 0
    assert results["metrics"]["joint/aux_v_loss"] > 0
    assert torch.isfinite(torch.tensor(results["total_loss"]))


def test_rejected_candidate_restores_model_and_adam_state_even_after_previous_updates():
    alg = make_algorithm(fsppo_joint_max_backtracks=1)
    collect(alg)
    alg.update()  # Populate optimizer momentum before testing rollback.
    collect(alg)
    alg._old_actor.load_state_dict(alg.policy.actor.state_dict())
    probe = alg._make_probe()
    alg.optimizer.zero_grad(set_to_none=True)
    # A known nonzero action-output gradient must violate an effectively zero budget.
    alg.policy.actor[-1].bias[:2].sum().backward()
    model_before = copy.deepcopy(alg.policy.state_dict())
    optimizer_before = copy.deepcopy(alg.optimizer.state_dict())
    alg.kl_target = 1e-30
    accepted, rejected, lr = alg._step_with_budget(probe)
    assert not accepted and rejected == 2 and lr == 0.0
    assert_tree_equal(alg.policy.state_dict(), model_before)
    assert_tree_equal(alg.optimizer.state_dict(), optimizer_before)


def test_backtracking_can_accept_smaller_first_step_and_restores_base_learning_rate():
    alg = make_algorithm(
        fsppo_joint_kl_target=0.002, fsppo_joint_max_backtracks=8, learning_rate=0.1
    )
    collect(alg)
    alg._old_actor.load_state_dict(alg.policy.actor.state_dict())
    probe = alg._make_probe()
    alg.optimizer.zero_grad(set_to_none=True)
    alg.policy.actor[-1].bias[:2].sum().backward()
    accepted, rejected, lr = alg._step_with_budget(probe)
    assert accepted and rejected > 0 and 0 < lr < 0.1
    assert float(alg._probe_l1(probe).mean()) <= alg.kl_target
    assert alg.optimizer.param_groups[0]["lr"] == 0.1
    # Rejected Adam trials must not accumulate optimizer steps.
    assert all(float(state["step"]) == 1 for state in alg.optimizer.state.values())


def test_all_rejected_update_reports_zero_accepted_steps():
    alg = make_algorithm(fsppo_joint_kl_target=1e-30, fsppo_joint_max_backtracks=0)
    collect(alg)
    before = copy.deepcopy(alg.policy.state_dict())
    result = alg.update()
    assert result["metrics"]["joint/accepted_updates"] == 0
    assert result["metrics"]["joint/rejected_candidates"] == 1
    assert result["metrics"]["joint/early_stop"] == 1
    assert result["metrics"]["joint/probe_kl_mean"] == 0
    assert_tree_equal(alg.policy.state_dict(), before)


def test_corrupted_generating_latent_is_rejected_before_any_update():
    alg = make_algorithm()
    collect(alg)
    before = copy.deepcopy(alg.policy.state_dict())
    alg.storage.action_latents.add_(10)
    with pytest.raises(RuntimeError, match="replay mismatch"):
        alg.update()
    assert_tree_equal(alg.policy.state_dict(), before)


def test_raw_action_and_observation_survive_environment_buffer_reuse():
    alg = make_algorithm()
    alg.init_storage(2, 1, [3], [3], [2])
    obs = torch.randn(2, 3)
    original_obs = obs.clone()
    actions = alg.act(obs, obs)
    original_actions = actions.clone()
    actions.zero_()
    obs.add_(100)
    alg.process_env_step(torch.zeros(2), torch.ones(2, dtype=torch.bool), {})
    assert torch.equal(alg.storage.actions[0], original_actions)
    assert torch.equal(alg.storage.observations[0], original_obs)


def test_timeout_bootstraps_value_but_true_termination_does_not():
    alg = make_algorithm()
    alg.init_storage(2, 1, [3], [3], [2])
    with torch.no_grad():
        for parameter in alg.policy.critic.parameters():
            parameter.zero_()
        alg.policy.critic[-1].bias.fill_(5)
    obs = torch.zeros(2, 3)
    alg.act(obs, obs)
    alg.process_env_step(
        torch.ones(2),
        torch.ones(2, dtype=torch.bool),
        {"time_outs": torch.tensor([True, False])},
    )
    alg.compute_returns(obs)
    assert torch.allclose(
        alg.storage.returns[0, :, 0], torch.tensor([1 + alg.gamma * 5, 1])
    )


def test_state_restores_dual_counter_and_checks_sampling_contract():
    alg = make_algorithm()
    collect(alg)
    alg.update()
    other = make_algorithm()
    other.load_state_dict(alg.state_dict())
    assert_tree_equal(alg.state_dict(), other.state_dict())
    changed = copy.deepcopy(alg.state_dict())
    changed["sampler"]["action_perturb_std"] *= 2
    with pytest.raises(ValueError, match="sampling contract"):
        other.load_state_dict(changed)
    other.policy.action_perturb_std *= 2
    with pytest.raises(ValueError, match="fixed"):
        other.act(torch.randn(2, 3), torch.randn(2, 3))


def test_joint_policy_requires_consistent_full_precision_across_batch_shapes():
    original = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        alg = make_algorithm()
        assert torch.get_float32_matmul_precision() == "highest"
        torch.set_float32_matmul_precision("high")
        with pytest.raises(ValueError, match="fixed"):
            alg.act(torch.randn(2, 3), torch.randn(2, 3))
    finally:
        torch.set_float32_matmul_precision(original)


@pytest.mark.parametrize(
    "overrides",
    [
        {"fsppo_joint_kl_target": 0},
        {"fsppo_joint_kl_target": float("nan")},
        {"fsppo_joint_kl_coef": -1},
        {"fsppo_joint_map_samples": 0},
        {"fsppo_joint_max_backtracks": -1},
        {"fsppo_joint_backtrack_factor": 1},
        {"trust_region_mode": "aspo"},
        {"schedule": "adaptive"},
        {"storage_action_noise_std": 0.1},
    ],
)
def test_invalid_algorithm_contracts_fail_early(overrides):
    with pytest.raises(ValueError):
        make_algorithm(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"sampling_steps": 2},
        {"action_perturb_std": 0},
        {"action_perturb_std": float("nan")},
        {"pmf_time_eps": 2},
    ],
)
def test_invalid_policy_contracts_fail_early(overrides):
    with pytest.raises(ValueError):
        FSPPOJoint(
            PMFActorCritic(3, 3, 2, make_policy_cfg(**overrides)), make_algorithm_cfg()
        )


class FakeEnv:
    num_actions = 2
    num_envs = 2
    device = "cpu"
    max_episode_length = 10
    episode_length_buf = torch.zeros(2, dtype=torch.long)

    def __init__(self):
        self.unwrapped = self
        self.cfg = SimpleNamespace()
        self.common_step_counter = 0

    def get_observations(self):
        obs = torch.zeros(2, 3)
        return obs, {"observations": {"critic": obs}}


def make_runner(ema_decay=0.0, action_perturb_std=0.2):
    cfg = FpoRslRlOnPolicyRunnerCfg(
        policy=make_policy_cfg(action_perturb_std=action_perturb_std),
        algorithm=make_algorithm_cfg(ema_decay=ema_decay, ema_warmup_steps=0),
        num_steps_per_env=3,
        max_iterations=2,
        save_interval=1,
        empirical_normalization=False,
        enable_post_training_eval=False,
        logger="tensorboard",
    )
    runner = OnPolicyRunner(FakeEnv(), cfg, device="cpu")
    runner.logger_type = "tensorboard"
    return runner


def test_runner_checkpoint_round_trip_and_cross_algorithm_rejection(tmp_path):
    runner = make_runner()
    collect(runner.alg)
    runner.alg.update()
    runner.current_learning_iteration = 3
    path = str(tmp_path / "joint.pt")
    runner.save(path)
    restored = make_runner()
    restored.load(path)
    assert_tree_equal(restored.alg.policy.state_dict(), runner.alg.policy.state_dict())
    assert_tree_equal(
        restored.alg.optimizer.state_dict(), runner.alg.optimizer.state_dict()
    )
    assert_tree_equal(restored.alg.state_dict(), runner.alg.state_dict())
    assert restored.current_learning_iteration == 3
    with pytest.raises(ValueError, match="sampling contract"):
        make_runner(action_perturb_std=0.1).load(path)
    saved = torch.load(path, weights_only=False)
    saved["algorithm_class_name"] = "FSPPO"
    torch.save(saved, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="algorithm"):
        restored.load(str(tmp_path / "old.pt"), load_optimizer=False)


def test_resume_uses_online_actor_not_ema_weights(tmp_path):
    runner = make_runner(ema_decay=0.95)
    collect(runner.alg)
    runner.alg.update()
    for parameter in runner.alg.ema.shadow_params.values():
        parameter.add_(10)
    path = str(tmp_path / "ema.pt")
    runner.save(path)
    restored = make_runner(ema_decay=0.95)
    restored.load(path)
    assert_tree_equal(restored.alg.policy.state_dict(), runner.alg.policy.state_dict())
    assert_tree_equal(
        restored.alg.optimizer.state_dict(), runner.alg.optimizer.state_dict()
    )


def test_joint_resume_restores_runner_clocks_only_with_optimizer(tmp_path):
    runner = make_runner()
    # A no-log-dir runner does not advance its legacy global counter in log(),
    # so the checkpoint must derive it from the completed joint updates.
    runner.current_learning_iteration = 2
    runner.tot_time = 3.5
    runner.env.common_step_counter = 6
    path = str(tmp_path / "joint_runner_state.pt")
    runner.save(path)

    saved = torch.load(path, weights_only=False)
    assert saved["fsppo_joint_runner_state_dict"] == {
        "version": 1,
        "tot_timesteps": 12,
        "tot_time": 3.5,
        "completed_env_steps": 6,
    }

    resumed = make_runner()
    resumed.load(path)
    assert resumed.current_learning_iteration == 2
    assert resumed.tot_timesteps == 12
    assert resumed.tot_time == pytest.approx(3.5)
    assert resumed.env.common_step_counter == 6

    # A model-only load is not a training resume and leaves runner clocks
    # untouched, even though it still validates the joint checkpoint contract.
    model_only = make_runner()
    model_only.current_learning_iteration = 7
    model_only.tot_timesteps = 11
    model_only.tot_time = 13.0
    model_only.env.common_step_counter = 17
    model_only.load(path, load_optimizer=False)
    assert model_only.current_learning_iteration == 7
    assert model_only.tot_timesteps == 11
    assert model_only.tot_time == 13.0
    assert model_only.env.common_step_counter == 17

    # Checkpoints written before the runner-state extension retain a safe
    # derivation from the completed FSPPOJoint iteration count.
    saved.pop("fsppo_joint_runner_state_dict")
    fallback_path = str(tmp_path / "joint_runner_state_legacy.pt")
    torch.save(saved, fallback_path)
    fallback = make_runner()
    fallback.load(fallback_path)
    assert fallback.tot_timesteps == 12
    assert fallback.env.common_step_counter == 6


def test_cli_selects_independent_joint_variant():
    cfg = FpoRslRlOnPolicyRunnerCfg(
        policy=make_policy_cfg(), algorithm=make_algorithm_cfg()
    )
    args = argparse.Namespace(
        algorithm="fsppo_joint",
        seed=None,
        resume=False,
        load_run=None,
        checkpoint=None,
        run_name=None,
        experiment_name=None,
        logger=None,
        log_project_name=None,
    )
    result = update_fpo_cfg(cfg, args)
    assert result.algorithm.class_name == "FSPPOJoint"
    assert result.policy.class_name == "PMFActorCritic"
    assert result.policy.sampling_steps == 1
    assert result.algorithm.trust_region_mode == "ppo"
