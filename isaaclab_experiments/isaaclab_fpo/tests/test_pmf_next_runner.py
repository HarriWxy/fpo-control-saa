"""Exercise new runner dispatch, true terminal bootstraps and safe resume."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab")

from isaaclab_fpo.rl_cfg import (
    FpoRslRlOnPolicyRunnerCfg,
    FpoRslRlPpoActorCriticCfg,
    FpoRslRlPpoAlgorithmCfg,
)
from isaaclab_fpo.runners import OnPolicyRunner
from isaaclab_fpo.wrapper import FpoRslRlVecEnvWrapper


class TimeoutEnv:
    """A vector env whose final states differ visibly from reset states."""

    num_envs = 2
    num_actions = 1
    max_episode_length = 2
    device = "cpu"

    def __init__(self):
        self.unwrapped = self
        self.cfg = SimpleNamespace(is_finite_horizon=False, compute_final_obs=True)
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.extras = {}

    def get_observations(self):
        obs = torch.zeros(2, 3)
        return obs, {"observations": {"critic": obs}}

    def step(self, actions):
        self.episode_length_buf += 1
        timeouts = self.episode_length_buf >= 2
        # Reuse extras, like the native environment, to expose stale tensor bugs.
        self.extras["final_obs"] = {
            "policy": torch.full((2, 3), 10.0),
            "critic": torch.full((2, 3), 50.0),
        }
        self.episode_length_buf[timeouts] = 0
        observations = {
            "policy": torch.full((2, 3), -9.0),
            "critic": torch.full((2, 3), -9.0),
        }
        return (
            observations,
            torch.ones(2),
            torch.zeros(2, dtype=torch.bool),
            timeouts,
            self.extras,
        )


def make_runner(monkeypatch, *, aligned=False, normalize=False):
    monkeypatch.setattr(torch, "compile", lambda function, *args, **kwargs: function)
    torch.set_num_threads(1)
    native = TimeoutEnv()
    # No simulator constructor is needed to exercise this real step adapter.
    wrapper = object.__new__(FpoRslRlVecEnvWrapper)
    wrapper.env = native
    wrapper.clip_actions = None
    wrapper.num_envs, wrapper.num_actions, wrapper.num_privileged_obs = 2, 1, 3
    wrapper.device, wrapper.max_episode_length = "cpu", 2
    wrapper.get_observations = native.get_observations
    cfg = FpoRslRlOnPolicyRunnerCfg(
        policy=FpoRslRlPpoActorCriticCfg(
            class_name="LearnedStdPMFActorCritic" if aligned else "PMFActorCritic",
            actor_hidden_dims=[8],
            critic_hidden_dims=[8],
            activation="elu",
            sampling_steps=1,
            action_perturb_std=0.1,
        ),
        algorithm=FpoRslRlPpoAlgorithmCfg(
            class_name="FSPPOJointAligned" if aligned else "PMFPMD",
            schedule="adaptive" if aligned else "fixed",
            num_learning_epochs=1,
            num_mini_batches=1,
            n_samples_per_action=1,
            ema_decay=0.0,
            fsppo_joint_enable_budget=False,
            fsppo_joint_aux_loss_coef=0.0,
            fsppo_joint_entropy_coef=0.0,
            storage_action_noise_std=0.0,
            knn_entropy_coef=0.0,
            trust_region_mode="ppo",
            pmd_q_warmup_updates=0,
            pmd_q_epochs=1,
            pmd_projection_epochs=1,
            pmd_q_hidden_dims=[8],
            pmd_batch_size=2,
        ),
        num_steps_per_env=2,
        empirical_normalization=normalize,
        enable_post_training_eval=False,
    )
    return OnPolicyRunner(wrapper, cfg, device="cpu")


def test_timeout_bootstrap_uses_normalized_final_state_without_updating_stats(
    monkeypatch,
):
    runner = make_runner(monkeypatch, normalize=True)
    obs, _ = runner.env.get_observations()
    runner.privileged_obs_normalizer(torch.tensor([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]))
    normalizer = runner.privileged_obs_normalizer
    before = copy.deepcopy(normalizer.state_dict())
    runner.alg.act(obs, obs)
    runner.env.step(torch.zeros(2, 1))
    _, rewards, dones, infos = runner.env.step(torch.zeros(2, 1))
    assert bool(dones.all())
    runner._normalize_terminal_observations(infos)
    assert normalizer.training
    for key, value in before.items():
        assert torch.equal(value, normalizer.state_dict()[key])
    terminal = infos["terminal_critic_observations"]
    assert not torch.equal(terminal, infos["observations"]["critic"])
    expected = rewards + runner.alg.gamma * runner.alg.policy.evaluate(
        terminal
    ).squeeze(-1)
    runner.alg.process_env_step(rewards, dones, infos)
    assert torch.allclose(runner.alg.storage.rewards[0, :, 0], expected)
    # The next ordinary step must not expose the previous terminal observations.
    assert "terminal_critic_observations" not in runner.env.step(torch.zeros(2, 1))[3]


@pytest.mark.parametrize("aligned", [False, True])
def test_new_runner_can_train_resume_and_reject_contract_before_mutation(
    monkeypatch, tmp_path, aligned
):
    runner = make_runner(monkeypatch, aligned=aligned)
    runner.learn(1)
    checkpoint = tmp_path / "model.pt"
    runner.save(str(checkpoint))
    resumed = make_runner(monkeypatch, aligned=aligned)
    resumed.load(str(checkpoint))
    for key, value in runner.alg.policy.state_dict().items():
        assert torch.equal(value, resumed.alg.policy.state_dict()[key])
    if not aligned:
        for key, value in runner.alg.q.state_dict().items():
            assert torch.equal(value, resumed.alg.q.state_dict()[key])
    resumed.learn(1)
    assert resumed.current_learning_iteration == 2
    assert resumed.tot_timesteps == 8
    saved = torch.load(checkpoint, weights_only=False)
    if aligned:
        saved["algorithm_state_dict"]["aligned_desired_kl"] *= 2
    else:
        saved["algorithm_state_dict"]["pmd_contract"]["gamma"] = 0.123
    for value in saved["model_state_dict"].values():
        if value.is_floating_point():
            value.add_(100)
    torch.save(saved, checkpoint)
    before = copy.deepcopy(resumed.alg.policy.state_dict())
    with pytest.raises(ValueError, match="mismatch"):
        resumed.load(str(checkpoint))
    for key, value in before.items():
        assert torch.equal(value, resumed.alg.policy.state_dict()[key])
