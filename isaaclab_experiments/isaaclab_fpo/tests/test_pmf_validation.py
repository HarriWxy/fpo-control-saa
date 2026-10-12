"""Checkpoint/bank reproducibility and rollback at the public validation API."""

from __future__ import annotations

import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab")

from isaaclab_fpo.pmf_validation import validate_pmf_checkpoint
from isaaclab_fpo.rl_cfg import (
    FpoRslRlOnPolicyRunnerCfg,
    FpoRslRlPpoActorCriticCfg,
    FpoRslRlPpoAlgorithmCfg,
)
from isaaclab_fpo.runners import OnPolicyRunner


class BoundedEnv:
    """A done flag at step five, despite a nominal episode limit of one."""

    num_envs = 2
    num_actions = 2
    device = "cpu"
    max_episode_length = 1
    clip_actions = None

    def __init__(self):
        self.unwrapped = self
        self.steps = 0
        self.resets = 0
        self.cfg = SimpleNamespace(decimation=1)
        self.observation_manager = SimpleNamespace(
            active_terms={"policy": ["position", "command"]},
            group_obs_term_dim={"policy": [(2,), (1,)]},
        )

    def get_observations(self):
        obs = (
            torch.tensor([[2.0, 4.0, 8.0], [3.0, 6.0, 9.0]])
            + self.steps
            + torch.rand(2, 3)
            + float(np.random.random())
            + random.random()
        )
        return obs, {"observations": {"policy": obs}}

    def reset(self):
        self.steps = 0
        self.resets += 1
        return self.get_observations()

    def step(self, actions):
        self.steps += 1
        obs, infos = self.get_observations()
        dones = torch.tensor([self.steps == 5, False])
        infos["time_outs"] = dones.clone()
        return obs, torch.ones(2), dones, infos


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, *args, **kwargs: fn)
    policy_cfg = FpoRslRlPpoActorCriticCfg(
        class_name="PMFActorCritic",
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
        activation="elu",
        sampling_steps=1,
        actor_scale=0.5,
        pmf_time_eps=0.1,
    )
    algorithm_cfg = FpoRslRlPpoAlgorithmCfg(
        class_name="FSPPOJoint",
        learning_rate=1e-3,
        ema_decay=0.0,
        knn_entropy_coef=0.0,
        schedule="fixed",
        trust_region_mode="ppo",
        storage_action_noise_std=0.0,
    )
    result = OnPolicyRunner(
        BoundedEnv(),
        FpoRslRlOnPolicyRunnerCfg(
            policy=policy_cfg,
            algorithm=algorithm_cfg,
            empirical_normalization=True,
            device="cpu",
            num_steps_per_env=2,
        ),
        device="cpu",
    )
    # A real pMF policy with an analytic observation-dependent mean endpoint.
    # The auxiliary endpoint is identical, so no diagnostic solver is mocked.
    actor = torch.nn.Linear(3 + policy_cfg.timestep_embed_dim + 2, 4)
    with torch.no_grad():
        actor.weight.zero_()
        actor.bias.zero_()
        actor.weight[0, 0] = actor.weight[2, 0] = 1.0
        actor.weight[1, 1] = actor.weight[3, 1] = 1.0
    result.alg.policy.actor = actor
    result.alg.policy.train()
    result.alg.policy.critic.eval()
    result.obs_normalizer.train()
    result.privileged_obs_normalizer.eval()
    return result


def save_checkpoint(runner, path, mean):
    norm = copy.deepcopy(runner.obs_normalizer.state_dict())
    norm["_mean"] = torch.tensor([mean], dtype=torch.float32)
    norm["_std"] = torch.tensor([[2.0, 4.0, 1.0]])
    norm["_var"] = norm["_std"].square()
    norm["count"] = torch.tensor(500)
    online = copy.deepcopy(runner.alg.policy.state_dict())
    online["actor.bias"].fill_(100.0)
    torch.save(
        {
            "model_state_dict": runner.alg.policy.state_dict(),
            "online_model_state_dict": online,
            "obs_norm_state_dict": norm,
            "privileged_obs_norm_state_dict": norm,
            "policy_class_name": "PMFActorCritic",
            "algorithm_class_name": "FSPPOJoint",
            "fsppo_joint_checkpoint_version": 1,
            "algorithm_state_dict": runner.alg.state_dict(),
            "ema_state_dict": {"shadow_params": online},
            "iter": 7,
        },
        path,
    )


def snapshot(runner):
    modules = (
        runner.alg.policy,
        runner.obs_normalizer,
        runner.privileged_obs_normalizer,
    )
    return [copy.deepcopy(module.state_dict()) for module in modules], [
        child.training for module in modules for child in module.modules()
    ]


def assert_restored(runner, original):
    current, modes = snapshot(runner)
    for before, after in zip(original[0], current):
        assert before.keys() == after.keys()
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    assert modes == original[1]


def test_checkpoint_normalization_bank_reuse_and_true_completed_episodes(
    runner, tmp_path
):
    checkpoint = tmp_path / "model_7.pt"
    save_checkpoint(runner, checkpoint, [1.0, 2.0, 3.0])
    original = snapshot(runner)
    rng = torch.random.get_rng_state().clone()
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    report = validate_pmf_checkpoint(
        runner,
        checkpoint,
        tmp_path / "first",
        task_name="example-task",
        num_states=4,
        samples_per_state=3,
        warmup_steps=1,
        collection_steps=4,
        ode_steps=(2, 4),
        batch_size=3,
    )
    assert_restored(runner, original)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert random.getstate() == python_rng
    restored_numpy_rng = np.random.get_state()
    assert restored_numpy_rng[0] == numpy_rng[0]
    np.testing.assert_array_equal(restored_numpy_rng[1], numpy_rng[1])
    assert restored_numpy_rng[2:] == numpy_rng[2:]
    assert report["collection"]["completed_episodes"] == 1
    assert report["collection"]["mean_completed_episode_return"] == 5.0
    assert report["collection"]["mean_completed_episode_length"] == 5.0
    assert report["collection"]["completed_episode_timeout_rate"] == 1.0
    bank_path = tmp_path / "first" / "state_bank.pt"
    bank = torch.load(bank_path, weights_only=True)
    assert bank["sampled_vector_steps"] == [1, 2, 3, 4]
    assert bank["sampled_env_ids"] == [0, 1, 0, 1]
    assert bank["initial_noise"].shape == (4, 3, 2)
    raw = bank["raw_observations"]
    expected_max_action = 0.5 * max(
        ((raw[:, 0] - 1.0) / 2.01).abs().max().item(),
        ((raw[:, 1] - 2.0) / 4.01).abs().max().item(),
    )
    assert report["diagnostics"]["one_step"]["max_abs"] == pytest.approx(
        expected_max_action
    )

    regenerated = validate_pmf_checkpoint(
        runner,
        checkpoint,
        tmp_path / "regenerated",
        task_name="example-task",
        num_states=4,
        samples_per_state=3,
        warmup_steps=1,
        collection_steps=4,
        ode_steps=(2, 4),
        batch_size=3,
    )
    assert (
        regenerated["state_bank"]["fingerprint"] == report["state_bank"]["fingerprint"]
    )
    assert_restored(runner, original)

    # B uses exactly A's raw states/noise but B's own normalizer.  Requested
    # default counts do not regenerate or truncate an existing bank.
    second_checkpoint = tmp_path / "model_8.pt"
    save_checkpoint(runner, second_checkpoint, [3.0, 4.0, 5.0])
    steps, resets = runner.env.steps, runner.env.resets
    second = validate_pmf_checkpoint(
        runner,
        second_checkpoint,
        tmp_path / "second",
        state_bank_path=bank_path,
        task_name="example-task",
        ode_steps=(2, 4),
    )
    assert runner.env.steps == steps and runner.env.resets == resets
    assert_restored(runner, original)
    assert second["state_bank"]["num_states"] == 4
    assert second["state_bank"]["samples_per_state"] == 3
    assert second["state_bank"]["fingerprint"] == report["state_bank"]["fingerprint"]
    assert second["collection"] is None
    assert (
        second["diagnostics"]["one_step"]["max_abs"]
        != report["diagnostics"]["one_step"]["max_abs"]
    )
    saved = json.loads((tmp_path / "second" / "report.json").read_text())
    assert saved["checkpoint"]["model_state_key"] == "model_state_dict"


@pytest.mark.parametrize("corruption", ["task", "nonfinite"])
def test_rejected_bank_restores_model_and_normalization(runner, tmp_path, corruption):
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(runner, checkpoint, [1.0, 2.0, 3.0])
    validate_pmf_checkpoint(
        runner,
        checkpoint,
        tmp_path / "source",
        task_name="example-task",
        num_states=2,
        samples_per_state=2,
        warmup_steps=0,
        collection_steps=2,
        ode_steps=(2,),
    )
    bank = torch.load(tmp_path / "source" / "state_bank.pt", weights_only=True)
    if corruption == "task":
        bank["environment_contract"]["task_name"] = "different-task"
        expected = "environment contract mismatch: task_name"
    else:
        bank["initial_noise"][0, 0, 0] = float("nan")
        expected = "initial_noise contains nonfinite"
    bank_path = tmp_path / "invalid.pt"
    torch.save(bank, bank_path)
    original = snapshot(runner)
    steps, resets = runner.env.steps, runner.env.resets
    with pytest.raises(ValueError, match=expected):
        validate_pmf_checkpoint(
            runner,
            checkpoint,
            tmp_path / "failed",
            state_bank_path=bank_path,
            task_name="example-task",
            ode_steps=(2,),
        )
    assert_restored(runner, original)
    assert runner.env.steps == steps and runner.env.resets == resets
