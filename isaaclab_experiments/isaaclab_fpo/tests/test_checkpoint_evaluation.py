"""Checkpoint evaluation counts real episodes and preserves learner state."""

from __future__ import annotations

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


class EpisodeEnv:
    """Three autoresetting environments with independently known episode returns."""

    num_envs = 3
    num_actions = 2
    device = "cpu"
    max_episode_length = 4

    def __init__(self, *, report_dones=True):
        self.unwrapped = self
        self.cfg = SimpleNamespace()
        self.episode_length_buf = torch.zeros(3, dtype=torch.long)
        self.report_dones = report_dones
        self.total_steps = 0
        self.reset_samples = []
        self.actions = []

    def seed(self, seed):
        torch.manual_seed(seed)

    def get_observations(self):
        obs = torch.full((3, 3), 8.0)
        return obs, {"observations": {"critic": obs}}

    def reset(self):
        self.episode_length_buf.zero_()
        self.reset_samples.append(torch.rand(3))
        return self.get_observations()

    def step(self, actions):
        self.total_steps += 1
        self.actions.append(actions.detach().clone())
        self.episode_length_buf += 1
        dones = self.episode_length_buf == torch.tensor([2, 3, 4])
        if not self.report_dones:
            dones.zero_()
        timeouts = dones & torch.tensor([True, False, True])
        self.episode_length_buf[dones] = 0
        obs, infos = self.get_observations()
        infos["time_outs"] = timeouts
        return obs, torch.tensor([1.0, 2.0, 3.0]), dones, infos


@pytest.fixture
def runner(monkeypatch):
    # These tests exercise episode bookkeeping and real CPU inference; CUDA
    # compilation is unrelated to that boundary and never runs here.
    monkeypatch.setattr(torch, "compile", lambda function, *args, **kwargs: function)
    torch.set_num_threads(1)
    cfg = FpoRslRlOnPolicyRunnerCfg(
        seed=13,
        policy=FpoRslRlPpoActorCriticCfg(
            class_name="PMFActorCritic",
            actor_hidden_dims=[4],
            critic_hidden_dims=[4],
            activation="elu",
            init_noise_std=1.0,
            sampling_steps=1,
            actor_scale=1.0,
            action_perturb_std=0.2,
        ),
        algorithm=FpoRslRlPpoAlgorithmCfg(
            class_name="FSPPOJoint",
            num_learning_epochs=1,
            num_mini_batches=1,
            learning_rate=1.0e-3,
            clip_param=0.2,
            gamma=0.99,
            lam=0.95,
            value_loss_coef=1.0,
            use_clipped_value_loss=False,
            max_grad_norm=1.0,
            schedule="fixed",
            trust_region_mode="ppo",
            knn_entropy_coef=0.0,
            storage_action_noise_std=0.0,
            ema_decay=0.0,
        ),
        num_steps_per_env=4,
        empirical_normalization=True,
        flow_eval_modes=["zero", "random"],
        enable_post_training_eval=False,
    )
    result = OnPolicyRunner(EpisodeEnv(), cfg, device="cpu")
    result.logger_type = "tensorboard"
    return result


@pytest.mark.parametrize(
    ("num_episodes", "mean_reward", "mean_length", "timeout_rate"),
    [(5, 5.6, 2.8, 0.6), (2, 4.0, 2.5, 0.5)],
)
def test_evaluation_uses_exact_balanced_episode_count(
    runner, tmp_path, num_episodes, mean_reward, mean_length, timeout_rate
):
    """Remainders and zero quotas must not undercount or overcount episodes."""
    runner.cfg.eval_episodes = num_episodes
    path = tmp_path / "model.pt"
    runner.save(str(path))
    results = runner.evaluate_checkpoint(str(path), 0)
    assert results is not None
    for metrics in results.values():
        assert metrics["num_episodes"] == num_episodes
        assert metrics["mean_reward"] == pytest.approx(mean_reward)
        assert metrics["mean_length"] == pytest.approx(mean_length)
        assert metrics["timeout_rate"] == pytest.approx(timeout_rate)
    # All modes start with the same reset randomization, even though random
    # inference consumes additional random numbers between reset calls.
    assert torch.equal(runner.env.reset_samples[0], runner.env.reset_samples[1])


def test_evaluation_uses_saved_actor_and_normalizer_then_restores_state(
    runner, tmp_path
):
    """Evaluation must use the checkpoint while leaving an existing eval actor intact."""
    with torch.no_grad():
        for parameter in runner.alg.policy.actor.parameters():
            parameter.zero_()
        runner.alg.policy.actor[0].weight[0, 0] = 1.0
        runner.alg.policy.actor[-1].weight[0, 0] = 1.0
        runner.alg.policy.actor[-1].bias[0] = 0.5
        runner.obs_normalizer._mean.fill_(4.0)
        runner.obs_normalizer._std.fill_(2.0)
        runner.obs_normalizer._var.fill_(4.0)
    runner.cfg.eval_episodes = 2
    path = tmp_path / "model.pt"
    runner.save(str(path))
    with torch.no_grad():
        runner.alg.policy.actor[-1].bias.add_(10.0)
        runner.obs_normalizer._mean.fill_(20.0)
    runner.eval_mode()
    # Preserve individually selected modes, not just the root policy flag.
    runner.alg.policy.critic.train()
    runner.privileged_obs_normalizer.train()
    state = {
        name: value.clone() for name, value in runner.alg.policy.state_dict().items()
    }
    obs_norm = {
        name: value.clone()
        for name, value in runner.obs_normalizer.state_dict().items()
    }
    rng_state = torch.random.get_rng_state()

    assert runner.evaluate_checkpoint(str(path), 0) is not None
    expected = torch.tensor([(8.0 - 4.0) / 2.01 + 0.5, 0.0]).expand(3, 2)
    assert torch.allclose(runner.env.actions[0], expected)
    for name, value in state.items():
        assert torch.equal(runner.alg.policy.state_dict()[name], value)
    for name, value in obs_norm.items():
        assert torch.equal(runner.obs_normalizer.state_dict()[name], value)
    assert not runner.alg.policy.training
    assert runner.alg.policy.critic.training
    assert not runner.obs_normalizer.training
    assert runner.privileged_obs_normalizer.training
    assert torch.equal(torch.random.get_rng_state(), rng_state)


def test_evaluation_without_dones_fails_bounded_and_restores_state(
    runner, tmp_path, capsys
):
    """A step cap must fail without manufacturing completed horizon episodes."""
    runner.env.report_dones = False
    runner.cfg.eval_episodes = 2
    path = tmp_path / "model.pt"
    runner.save(str(path))
    with torch.no_grad():
        runner.alg.policy.actor[-1].bias.add_(3.0)
        runner.obs_normalizer._mean.fill_(17.0)
    state = {
        name: value.clone() for name, value in runner.alg.policy.state_dict().items()
    }
    runner.eval_mode()
    rng_state = torch.random.get_rng_state()

    assert runner.evaluate_checkpoint(str(path), 0) is None
    assert 0 < runner.env.total_steps <= 8
    assert "completed 0/2 episodes" in capsys.readouterr().out
    for name, value in state.items():
        assert torch.equal(runner.alg.policy.state_dict()[name], value)
    assert torch.equal(runner.obs_normalizer.mean, torch.full((3,), 17.0))
    assert not runner.alg.policy.training
    assert not runner.obs_normalizer.training
    assert not runner.privileged_obs_normalizer.training
    assert torch.equal(torch.random.get_rng_state(), rng_state)
