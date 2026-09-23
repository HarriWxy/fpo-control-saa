"""CPU contracts for terminal-endpoint MAE FSPPO."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab")

from isaaclab_fpo.algorithms import FSPPOJointEndpoint
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
    torch.manual_seed(11)


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
        class_name="FSPPOJointEndpoint",
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        schedule="fixed",
        trust_region_mode="ppo",
        n_samples_per_action=1,
        normalize_advantage=False,
        knn_entropy_coef=0.0,
        storage_action_noise_std=0.0,
        ema_decay=0.0,
        fsppo_joint_enable_budget=False,
        fsppo_joint_endpoint_mae_coef=0.2,
    )
    settings.update(overrides)
    return FpoRslRlPpoAlgorithmCfg(**settings)


def make_algorithm(**overrides):
    return FSPPOJointEndpoint(
        PMFActorCritic(3, 3, 2, make_policy_cfg()),
        make_algorithm_cfg(**overrides),
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


def test_endpoint_ratio_and_mae_use_only_terminal_transport_output(monkeypatch):
    algorithm = make_algorithm()
    collect(algorithm)
    batch = next(algorithm.storage.mini_batch_generator(1, 1))
    def fail_pmf_loss(*args, **kwargs):
        del args, kwargs
        pytest.fail("Endpoint variant must not use pMF JVP loss")

    monkeypatch.setattr(algorithm.policy, "get_pmf_loss", fail_pmf_loss)

    loss, statistics = algorithm._batch_loss(batch)
    endpoint = algorithm.policy.transport_actions(
        batch["obs"], batch["action_latents"]
    )
    expected_mae = (endpoint - batch["actions"]).abs().mean(dim=-1)
    old_mae = (batch["action_means"] - batch["actions"]).abs().mean(dim=-1)
    expected_ratio = torch.exp(
        (old_mae - expected_mae).clamp(max=algorithm.cfm_diff_clamp_max)
    )

    assert torch.isfinite(loss)
    assert torch.allclose(statistics["endpoint_mae"], expected_mae.mean())
    assert torch.allclose(statistics["endpoint_old_mae"], old_mae.mean())
    assert torch.allclose(statistics["ratio_mean"], expected_ratio.mean())


def test_endpoint_update_has_no_map_sample_or_budget_path(monkeypatch):
    algorithm = make_algorithm()
    def fail_pmf_loss(*args, **kwargs):
        del args, kwargs
        pytest.fail("Endpoint variant must not use pMF JVP loss")

    monkeypatch.setattr(algorithm.policy, "get_pmf_loss", fail_pmf_loss)
    collect(algorithm)
    results = algorithm.update()

    assert algorithm.storage.step == 0
    assert set(("surrogate_loss", "value_loss", "mae_loss", "total_loss")) <= set(
        results
    )
    assert results["mae_loss"] > 0
    assert "map_kl_loss" not in results
    assert "joint/map_kl_train" not in results["metrics"]
    assert results["metrics"]["endpoint/accepted_updates"] == 4
    assert all(
        torch.isfinite(torch.tensor(value))
        for value in results["metrics"].values()
    )


def test_endpoint_variant_is_selectable_from_cli_config():
    cfg = FpoRslRlOnPolicyRunnerCfg(
        policy=FpoRslRlPpoActorCriticCfg(
            class_name="ActorCritic",
            actor_hidden_dims=[8],
            critic_hidden_dims=[8],
            activation="elu",
        ),
        algorithm=FpoRslRlPpoAlgorithmCfg(),
    )
    args = SimpleNamespace(
        seed=None,
        resume=None,
        load_run=None,
        checkpoint=None,
        run_name=None,
        experiment_name=None,
        logger=None,
        log_project_name=None,
        algorithm="fsppo_joint_endpoint",
    )

    updated = update_fpo_cfg(cfg, args)

    assert updated.policy.class_name == "PMFActorCritic"
    assert updated.algorithm.class_name == "FSPPOJointEndpoint"
    assert updated.policy.sampling_steps == 1
    assert updated.algorithm.trust_region_mode == "ppo"
    assert updated.algorithm.fsppo_joint_enable_budget is False


class FakeEnv:
    """Minimal vector environment needed to exercise runner checkpoint wiring."""

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


def test_endpoint_runner_checkpoint_round_trip(tmp_path):
    cfg = FpoRslRlOnPolicyRunnerCfg(
        policy=make_policy_cfg(),
        algorithm=make_algorithm_cfg(),
        num_steps_per_env=3,
        max_iterations=2,
        save_interval=1,
        empirical_normalization=False,
        enable_post_training_eval=False,
        logger="tensorboard",
    )
    runner = OnPolicyRunner(FakeEnv(), cfg, device="cpu")
    runner.logger_type = "tensorboard"
    collect(runner.alg)
    runner.alg.update()
    runner.current_learning_iteration = 1
    path = str(tmp_path / "endpoint.pt")
    runner.save(path)

    restored = OnPolicyRunner(FakeEnv(), cfg, device="cpu")
    restored.logger_type = "tensorboard"
    restored.load(path)

    assert restored.alg.state_dict() == runner.alg.state_dict()
    assert restored.current_learning_iteration == 1
