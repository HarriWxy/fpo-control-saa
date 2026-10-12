"""CPU behavior contracts for action-conditioned PMD and flow projection.

The tests guard per-state mass redistribution, actual Q discrimination from
returns, terminal bootstrapping, actor rollback, and exact continuation. Existing
Joint tests cover collection/replay; these exercise PMD's different update path.
"""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab")

from isaaclab_fpo.algorithms import PMFPMD
from isaaclab_fpo.modules import PMFActorCritic
from isaaclab_fpo.rl_cfg import FpoRslRlPpoActorCriticCfg, FpoRslRlPpoAlgorithmCfg


@pytest.fixture(autouse=True)
def cpu_execution(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda function, *args, **kwargs: function)
    torch.set_num_threads(1)
    torch.manual_seed(31)


def make_algorithm(**overrides):
    policy = PMFActorCritic(
        3,
        3,
        1,
        FpoRslRlPpoActorCriticCfg(
            class_name="PMFActorCritic",
            actor_hidden_dims=[16, 16],
            critic_hidden_dims=[16, 16],
            activation="elu",
            sampling_steps=1,
            actor_scale=1.0,
            action_perturb_std=0.8,
            pmf_logit_normal_mean=0.0,
            pmf_logit_normal_std=0.2,
            pmf_adaptive_gradient_norm_p=0.0,
        ),
    )
    settings = {
        "class_name": "PMFPMD",
        "num_learning_epochs": 1,
        "num_mini_batches": 2,
        "learning_rate": 0.005,
        "weight_decay": 0.0,
        "schedule": "fixed",
        "trust_region_mode": "ppo",
        "knn_entropy_coef": 0.0,
        "storage_action_noise_std": 0.0,
        "ema_decay": 0.0,
        "n_samples_per_action": 1,
        "fsppo_joint_aux_loss_coef": 0.0,
        "fsppo_joint_entropy_coef": 0.0,
        "fsppo_joint_enable_budget": False,
        "fsppo_joint_map_samples": 3,
        "fsppo_joint_probe_size": 16,
        "fsppo_joint_max_backtracks": 0,
        "pmd_q_hidden_dims": [32, 32],
        "pmd_q_epochs": 2,
        "pmd_projection_epochs": 1,
        "pmd_batch_size": 64,
        "pmd_q_warmup_updates": 0,
        "pmd_q_learning_rate": 0.005,
        "pmd_value_learning_rate": 0.005,
    }
    settings.update(overrides)
    return PMFPMD(policy, FpoRslRlPpoAlgorithmCfg(**settings))


def collect_bandit(algorithm, count=64):
    """One-step episodes with a known action-dependent optimum at +0.7."""
    algorithm.init_storage(count, 1, [3], [3], [1])
    observations = torch.zeros(count, 3)
    actions = algorithm.act(observations, observations)
    rewards = -(actions[:, 0] - 0.7).square()
    algorithm.process_env_step(rewards, torch.ones(count, dtype=torch.bool), {})
    algorithm.compute_returns(observations)


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def test_weights_are_statewise_shift_invariant_and_respect_ess():
    algorithm = make_algorithm(pmd_temperature=0.01, pmd_min_ess_fraction=0.5)
    scores = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        ]
    )
    weights, temperatures = algorithm.candidate_weights(scores)
    shifted, _ = algorithm.candidate_weights(scores + torch.tensor([[100.0], [-50.0]]))
    assert torch.allclose(weights, shifted, atol=1e-6)
    assert torch.allclose(weights.sum(-1), torch.ones(2))
    assert torch.equal(weights[1], torch.full((8,), 0.125))
    assert weights[0, -1] > weights[0, 0]
    normalized_ess = 1 / (8 * weights.square().sum(-1))
    assert bool((normalized_ess >= 0.5 - 1e-6).all())
    assert temperatures[0] > 0.01
    assert temperatures[1] == pytest.approx(0.01)


def test_q_learns_action_ranking_while_warmup_preserves_actor():
    algorithm = make_algorithm(
        pmd_q_warmup_updates=1, pmd_q_epochs=150, pmd_q_learning_rate=0.01
    )
    before = copy.deepcopy(algorithm.policy.actor.state_dict())
    collect_bandit(algorithm, count=512)
    result = algorithm.update()
    assert_tree_equal(algorithm.policy.actor.state_dict(), before)
    assert result["metrics"]["pmd/q_warmup"] == 1
    assert result["metrics"]["pmd/accepted_projection_steps"] == 0
    # The state is identical. A V-only critic cannot pass this action ranking.
    q_values = (
        algorithm.q(torch.zeros(2, 3), torch.tensor([[0.7], [-0.7]])).detach().flatten()
    )
    assert q_values[0] > q_values[1] + 1.0
    assert result["metrics"]["pmd/q_validation_mse"] < 0.1


def test_weighted_flow_projection_moves_generated_distribution_to_better_candidate():
    algorithm = make_algorithm(pmd_num_candidates=2)
    observations = torch.zeros(32, 3)
    candidates = torch.tensor([[-1.0], [1.0]]).expand(32, 2, 1).clone()
    weights = torch.tensor([[0.01, 0.99]]).expand(32, 2)
    probe_noise = torch.randn(32, 64, 1)
    before = (
        algorithm.policy.transport_actions(observations, probe_noise).detach().mean()
    )
    for _ in range(150):
        loss, _ = algorithm._projection_loss(observations, candidates, weights)
        algorithm._train_step(
            loss, algorithm.optimizer, algorithm.policy.actor.parameters()
        )
        algorithm.optimizer.step()
    after = (
        algorithm.policy.transport_actions(observations, probe_noise).detach().mean()
    )
    assert after > before + 0.5
    assert (after - 0.98).abs() < 0.5 * (before - 0.98).abs()


def test_time_limit_uses_terminal_value_and_missing_terminal_state_fails():
    algorithm = make_algorithm(pmd_q_warmup_updates=1)
    algorithm.init_storage(4, 1, [3], [3], [1])
    current, terminal = torch.zeros(4, 3), torch.full((4, 3), 5.0)
    algorithm.act(current, current)
    terminal_values = algorithm.policy.evaluate(terminal).detach().flatten()
    assert not torch.allclose(
        terminal_values, algorithm.policy.evaluate(current).detach().flatten()
    )
    with pytest.raises(ValueError, match="terminal_critic_observations"):
        algorithm.process_env_step(
            torch.ones(4),
            torch.ones(4, dtype=torch.bool),
            {"time_outs": torch.ones(4, dtype=torch.bool)},
        )
    algorithm.process_env_step(
        torch.ones(4),
        torch.ones(4, dtype=torch.bool),
        {
            "time_outs": torch.ones(4, dtype=torch.bool),
            "terminal_critic_observations": terminal,
        },
    )
    algorithm.compute_returns(current)
    assert torch.allclose(
        algorithm.storage.returns.flatten(), 1 + algorithm.gamma * terminal_values
    )


def test_squared_gaussian_budget_rolls_back_actor_and_existing_adam_moments():
    algorithm = make_algorithm(
        fsppo_joint_enable_budget=True, fsppo_joint_kl_target=1e6
    )
    collect_bandit(algorithm)
    assert algorithm.update()["metrics"]["pmd/accepted_projection_steps"] > 0
    assert algorithm.optimizer.state
    probe = algorithm._make_probe()
    current_means = algorithm.policy.transport_actions(probe["obs"], probe["noise"])
    gaussian_oracle = torch.distributions.kl_divergence(
        torch.distributions.Normal(probe["old_actions"], algorithm.sigma),
        torch.distributions.Normal(current_means, algorithm.sigma),
    ).sum(-1)
    assert gaussian_oracle.mean() > 0
    assert torch.allclose(algorithm._probe_distance(probe), gaussian_oracle, atol=1e-6)
    actor_before = copy.deepcopy(algorithm.policy.actor.state_dict())
    optimizer_before = copy.deepcopy(algorithm.optimizer.state_dict())
    # Tighten the runtime budget for a second real projection, with no mock KL.
    algorithm.kl_target = 1e-20
    collect_bandit(algorithm)
    result = algorithm.update()
    assert result["metrics"]["pmd/rejected_candidates"] > 0
    assert result["metrics"]["pmd/accepted_projection_steps"] == 0
    assert result["metrics"]["pmd/probe_conditional_joint_kl_mean"] == 0
    assert_tree_equal(algorithm.policy.actor.state_dict(), actor_before)
    assert_tree_equal(algorithm.optimizer.state_dict(), optimizer_before)


def test_checkpoint_restores_q_value_actor_and_exact_continuation():
    original = make_algorithm(
        pmd_batch_size=7,
        action_perturb_std_final=0.4,
        action_perturb_std_decay_env_steps=1000,
    )
    original.begin_rollout(64)
    collect_bandit(original)
    original.update()
    checkpoint = original.state_dict()
    restored = make_algorithm(
        pmd_batch_size=7,
        action_perturb_std_final=0.4,
        action_perturb_std_decay_env_steps=1000,
    )
    restored.load_state_dict(checkpoint)
    assert_tree_equal(original.state_dict(), restored.state_dict())
    rng = torch.get_rng_state()
    original.begin_rollout(128)
    collect_bandit(original)
    expected = original.update()
    torch.set_rng_state(rng)
    restored.begin_rollout(128)
    collect_bandit(restored)
    actual = restored.update()
    assert actual["metrics"]["pmd/accepted_projection_steps"] == 2
    assert actual["metrics"]["pmd/projection_flow_examples"] == 64 * 8
    assert all(float(state["step"]) == 4 for state in restored.optimizer.state.values())
    assert_tree_equal(expected, actual)
    assert_tree_equal(original.state_dict(), restored.state_dict())
    incompatible = make_algorithm(
        pmd_num_candidates=4,
        pmd_batch_size=7,
        action_perturb_std_final=0.4,
        action_perturb_std_decay_env_steps=1000,
    )
    with pytest.raises(ValueError, match="contract mismatch"):
        incompatible.load_state_dict(checkpoint)
