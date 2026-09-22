"""CPU regression checks for deferred scalar logging and gradient clipping."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab")

from isaaclab_fpo.algorithms import FPO, FSPPOJoint
from isaaclab_fpo.modules import ActorCritic, PMFActorCritic
from isaaclab_fpo.rl_cfg import FpoRslRlPpoActorCriticCfg, FpoRslRlPpoAlgorithmCfg


@pytest.fixture(autouse=True)
def cpu_execution(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, *args, **kwargs: fn)
    torch.set_num_threads(1)
    torch.manual_seed(19)


def make_algorithm(joint=False, **overrides):
    policy_cfg = FpoRslRlPpoActorCriticCfg(
        actor_hidden_dims=[8, 8],
        critic_hidden_dims=[8, 8],
        activation="elu",
        sampling_steps=1,
        actor_scale=0.5,
        action_perturb_std=0.2 if joint else 0.0,
    )
    settings = {
        "class_name": "FSPPOJoint" if joint else "FPO",
        "num_learning_epochs": 2,
        "num_mini_batches": 2,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "schedule": "fixed",
        "desired_kl": 0.01,
        "trust_region_mode": "ppo",
        "n_samples_per_action": 3,
        "normalize_advantage": False,
        "knn_entropy_coef": 0.0,
        "storage_action_noise_std": 0.0,
        "ema_decay": 0.0,
        "fsppo_joint_kl_target": 1000.0,
        "fsppo_joint_probe_size": 8,
        "fsppo_joint_map_samples": 2,
    }
    settings.update(overrides)
    policy_cls, algorithm_cls = (
        (PMFActorCritic, FSPPOJoint) if joint else (ActorCritic, FPO)
    )
    return algorithm_cls(
        policy_cls(3, 3, 2, policy_cfg), FpoRslRlPpoAlgorithmCfg(**settings)
    )


def collect(algorithm):
    algorithm.init_storage(2, 3, [3], [3], [2])
    for _ in range(3):
        obs = torch.randn(2, 3)
        algorithm.act(obs, obs)
        algorithm.process_env_step(torch.randn(2), torch.zeros(2, dtype=torch.bool), {})
    algorithm.compute_returns(torch.randn(2, 3))


def record_gradient_norms(monkeypatch):
    """Measure actual gradient tensors independently of reported clip norms."""
    original_clip = torch.nn.utils.clip_grad_norm_
    observations = []

    def clip(parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)

        def actual_norm():
            gradients = [
                p.grad.detach().flatten().double()
                for p in parameters
                if p.grad is not None
            ]
            return float(torch.cat(gradients).norm())

        before = actual_norm()
        result = original_clip(parameters, max_norm, *args, **kwargs)
        observations.append((before, actual_norm()))
        return result

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    return observations


def assert_scalar_results(result):
    for group in (
        {key: value for key, value in result.items() if key != "metrics"},
        result["metrics"],
    ):
        assert all(isinstance(value, (float, int)) for value in group.values())
        assert all(math.isfinite(value) for value in group.values())


@pytest.mark.parametrize("max_grad_norm", [0.0, 0.1, 100.0])
@pytest.mark.parametrize("schedule", ["fixed", "adaptive"])
def test_fpo_reports_actual_gradient_norms_and_scalar_normalizer_metrics(
    monkeypatch, max_grad_norm, schedule
):
    algorithm = make_algorithm(
        max_grad_norm=max_grad_norm, schedule=schedule, knn_entropy_coef=0.01
    )
    collect(algorithm)
    norms = record_gradient_norms(monkeypatch)
    normalizer = SimpleNamespace(std=torch.tensor([0.5, 1.5, 4.0]))
    result = algorithm.update(normalizer, normalizer)
    assert_scalar_results(result)
    assert len(norms) == 4
    for index, name in enumerate(("before", "after")):
        expected = sum(pair[index] for pair in norms) / len(norms)
        assert result["metrics"][f"mean_grad_norm_{name}_clip"] == pytest.approx(
            expected, rel=2e-6, abs=1e-8
        )
    for prefix in ("obs_norm", "privileged_obs_norm"):
        assert result["metrics"][f"{prefix}_min_std"] == 0.5
        assert result["metrics"][f"{prefix}_max_std"] == 4.0
        assert result["metrics"][f"{prefix}_mean_std"] == 2.0
    assert "entropy_loss" in result
    assert ("kl" in result["metrics"]) == (schedule == "adaptive")


@pytest.mark.parametrize("reject_all", [False, True])
def test_joint_accumulates_only_accepted_statistics_and_preserves_counter_types(
    monkeypatch, reject_all
):
    algorithm = make_algorithm(
        joint=True,
        max_grad_norm=0.1,
        fsppo_joint_kl_target=1e-30 if reject_all else 1000.0,
        fsppo_joint_max_backtracks=0,
    )
    collect(algorithm)
    batch_statistics = []
    original_loss = algorithm._batch_loss

    def batch_loss(batch):
        loss, statistics = original_loss(batch)
        batch_statistics.append(
            {name: float(value.detach()) for name, value in statistics.items()}
        )
        return loss, statistics

    monkeypatch.setattr(algorithm, "_batch_loss", batch_loss)
    norms = record_gradient_norms(monkeypatch)
    result = algorithm.update()
    assert_scalar_results(result)
    metrics = result["metrics"]
    accepted = metrics["joint/accepted_updates"]
    assert type(accepted) is int
    assert type(metrics["joint/attempted_batches"]) is int
    assert type(metrics["joint/rejected_candidates"]) is int
    assert accepted == (0 if reject_all else 4)
    for key in (
        "surrogate_loss",
        "value_loss",
        "map_kl_loss",
        "pmf_aux_loss",
        "total_loss",
    ):
        expected = sum(row[key] for row in batch_statistics[:accepted]) / max(
            accepted, 1
        )
        assert result[key] == pytest.approx(expected, rel=2e-6, abs=1e-8)
    for index, name in enumerate(("before", "after")):
        expected = sum(pair[index] for pair in norms[:accepted]) / max(accepted, 1)
        assert metrics[f"mean_grad_norm_{name}_clip"] == pytest.approx(
            expected, rel=2e-6, abs=1e-8
        )
    if reject_all:
        assert metrics["joint/ratio_mean"] == 1.0
        assert metrics["joint/probe_kl_mean"] == 0.0
    assert algorithm.storage.step == 0


@pytest.mark.parametrize("joint", [False, True])
def test_constant_returns_keep_explained_variance_zero(joint):
    algorithm = make_algorithm(joint=joint)
    collect(algorithm)
    algorithm.storage.returns.fill_(1.0)
    result = algorithm.update()
    assert_scalar_results(result)
    assert result["metrics"]["explained_variance"] == 0.0
