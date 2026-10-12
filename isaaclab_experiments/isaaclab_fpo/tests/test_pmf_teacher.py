"""Native teacher schema, conditional covariance and offline student replay."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")

from isaaclab_fpo.modules import PMFActorCritic
from isaaclab_fpo.pmf_teacher import (
    conditional_action_covariance,
    covariance_report,
    distill_teacher,
    load_native_ppo_teacher,
    load_pmf_checkpoint,
    load_raw_state_bank,
    validate_checkpoint_bank_contract,
)
from rsl_rl.models import MLPModel
from tensordict import TensorDict


@pytest.fixture
def teacher_bank(tmp_path):
    """An actual native actor, with analytically known normalized outputs."""
    torch.set_num_threads(1)
    config = {
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "clip_actions": None,
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [2],
            "activation": "relu",
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 0.5,
                "std_type": "scalar",
            },
        },
    }
    actor = MLPModel(
        TensorDict({"policy": torch.zeros(1, 2)}, batch_size=[1]),
        config["obs_groups"],
        "actor",
        2,
        **{key: value for key, value in config["actor"].items() if key != "class_name"},
    )
    with torch.no_grad():
        for layer in (actor.mlp[0], actor.mlp[2]):
            layer.weight.copy_(torch.eye(2))
            layer.bias.zero_()
        actor.distribution.std_param.copy_(torch.tensor([0.25, 0.5]))
        actor.obs_normalizer._mean.copy_(torch.tensor([[1.0, 2.0]]))
        actor.obs_normalizer._std.copy_(torch.tensor([[2.0, 4.0]]))
    run = tmp_path / "teacher"
    (run / "params").mkdir(parents=True)
    env = {
        "observations": {
            "policy": {"position": {"func": "task:position", "scale": 1.0}}
        },
        "actions": {
            "joint_pos": {"scale": 0.5, "joint_names": [".*"], "preserve_order": False}
        },
        "commands": {"velocity": {"ranges": {"lin_vel_x": [0.0, 1.0]}}},
        "sim": {"dt": 0.005, "gravity": [0.0, 0.0, -9.81]},
        "decimation": 4,
    }
    for name, values in (("agent", config), ("env", env)):
        (run / "params" / f"{name}.yaml").write_text(yaml.safe_dump(values))
    checkpoint = run / "model_5.pt"
    torch.save({"actor_state_dict": actor.state_dict(), "iter": 5}, checkpoint)
    raw = torch.tensor([[3.0, 6.0], [5.0, 10.0], [7.0, 14.0]])
    contract = {
        "actor_observation_dim": 2,
        "action_dim": 2,
        "policy_observation_cfg": env["observations"]["policy"],
        "actions_cfg": env["actions"],
        "commands_cfg": env["commands"],
        "sim_dt": 0.005,
        "gravity": [0.0, 0.0, -9.81],
        "decimation": 4,
        "wrapper_action_clip": None,
    }
    bank = {
        "version": 1,
        "observation_coordinates": "raw",
        "noise_distribution": "standard_normal",
        "environment_contract": contract,
        "raw_observations": raw,
        "initial_noise": torch.randn(3, 2, 2),
    }
    return load_native_ppo_teacher(checkpoint), bank, env


def _student_config():
    return {
        "empirical_normalization": False,
        "clip_actions": None,
        "policy": {
            "class_name": "PMFActorCritic",
            "actor_hidden_dims": [4],
            "critic_hidden_dims": [4],
            "activation": "relu",
            "actor_scale": 0.5,
            "actor_mlp_output_scale": 1.0,
            "actor_final_layer_weight_scale": None,
            "timestep_embed_dim": 8,
            "cfm_loss_t_inverse_cdf_beta": 1.0,
            "cfm_loss_reduction": "mean",
            "sampling_steps": 1,
            "training_sampling_steps": 1,
            "action_perturb_std": 0.3,
            "pmf_logit_normal_mean": 0.0,
            "pmf_logit_normal_std": 0.5,
            "pmf_fm_proportion": 0.5,
            "pmf_aux_v_loss_coef": 1.0,
            "pmf_time_eps": 0.05,
            "pmf_adaptive_gradient_norm_p": 0.0,
            "pmf_adaptive_gradient_norm_eps": 0.01,
        },
        "algorithm": {"class_name": "FSPPOJoint", "ema_decay": 0.0},
    }


def test_native_teacher_restores_own_normalizer_and_contract(teacher_bank, tmp_path):
    teacher, bank, _ = teacher_bank
    mean, std = teacher.moments(bank["raw_observations"])
    expected = torch.tensor(
        [[2 / 2.01, 4 / 4.01], [4 / 2.01, 8 / 4.01], [6 / 2.01, 12 / 4.01]]
    )
    torch.testing.assert_close(mean, expected)
    torch.testing.assert_close(std, torch.tensor([[0.25, 0.5]]).expand(3, -1))
    assert all(not parameter.requires_grad for parameter in teacher.actor.parameters())
    validate_checkpoint_bank_contract(teacher.checkpoint_path, bank)
    mismatch = copy.deepcopy(bank)
    mismatch["environment_contract"]["actions_cfg"]["joint_pos"]["scale"] = 1.0
    with pytest.raises(ValueError, match="actions_cfg"):
        validate_checkpoint_bank_contract(teacher.checkpoint_path, mismatch)
    mismatch["observation_coordinates"] = "normalized"
    bank_path = tmp_path / "invalid.pt"
    torch.save(mismatch, bank_path)
    with pytest.raises(ValueError, match="raw"):
        load_raw_state_bank(bank_path)


def test_covariance_conditions_on_state_instead_of_pooling():
    centered = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    samples = torch.stack((centered, centered + torch.tensor([100.0, -200.0])))
    stats = conditional_action_covariance(samples)
    torch.testing.assert_close(
        stats["covariance"], torch.eye(2, dtype=torch.float64).expand(2, 2, 2) * (2 / 3)
    )
    torch.testing.assert_close(
        stats["independent_action_delta_rms"],
        torch.tensor((4 / 3) ** 0.5, dtype=torch.float64),
    )


@pytest.mark.parametrize("learned_scale", [False, True])
def test_covariance_decomposes_latent_and_terminal_sources(
    teacher_bank, tmp_path, learned_scale
):
    teacher, bank, env = teacher_bank
    config = _student_config()
    policy_class = PMFActorCritic
    terminal_std = torch.tensor([0.3, 0.3])
    if learned_scale:
        from isaaclab_fpo.modules.learned_std_actor_critic import (
            LearnedStdPMFActorCritic,
        )

        config["policy"].update(
            class_name="LearnedStdPMFActorCritic",
            learn_action_perturb_std=True,
            action_perturb_std_min=0.01,
            action_perturb_std_max=2.0,
        )
        policy_class = LearnedStdPMFActorCritic
        terminal_std = torch.tensor([0.2, 0.4])
    policy = policy_class(2, 2, 2, SimpleNamespace(**config["policy"]))
    flow = policy.actor.flow if learned_scale else policy.actor
    # ReLU(+z,-z) implements the known linear map (z1, z1+z2),
    # followed by actor_scale=0.5. Its covariance is known independently.
    with torch.no_grad():
        if learned_scale:
            policy.actor.action_std_logit.copy_(
                torch.logit((terminal_std - 0.01) / 1.99)
            )
        flow[0].weight.zero_()
        flow[0].bias.zero_()
        flow[0].weight[:, -2:] = torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
        )
        flow[2].weight.zero_()
        flow[2].bias.zero_()
        flow[2].weight[:2] = torch.tensor(
            [[1.0, -1.0, 0.0, 0.0], [1.0, -1.0, 1.0, -1.0]]
        )
    run = tmp_path / "pmf"
    (run / "params").mkdir(parents=True)
    (run / "params" / "agent.yaml").write_text(yaml.safe_dump(config))
    (run / "params" / "env.yaml").write_text(yaml.safe_dump(env))
    torch.save({"model_state_dict": policy.state_dict()}, run / "model_1.pt")
    pmf = load_pmf_checkpoint(run / "model_1.pt")
    report, _ = covariance_report(teacher, pmf, bank, samples_per_state=12000)
    expected = torch.tensor([[0.25, 0.25], [0.25, 0.5]], dtype=torch.float64)
    observed = torch.tensor(
        report["distributions"]["pmf_latent_transport"]["mean_conditional_covariance"],
        dtype=torch.float64,
    )
    torch.testing.assert_close(observed, expected, atol=0.015, rtol=0)
    full = torch.tensor(
        report["distributions"]["pmf_executed_raw"]["mean_conditional_covariance"],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        full, expected + torch.diag(terminal_std.double().square()), atol=0.02, rtol=0
    )
    teacher_cov = torch.tensor(
        report["distributions"]["ppo_full_gaussian"]["mean_conditional_covariance"],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        teacher_cov,
        torch.diag(torch.tensor([0.0625, 0.25], dtype=torch.float64)),
        atol=0.01,
        rtol=0,
    )


def test_distillation_updates_student_freezes_teacher_and_replays(
    teacher_bank, tmp_path
):
    teacher, bank, _ = teacher_bank
    before = copy.deepcopy(teacher.actor.state_dict())
    report = distill_teacher(
        teacher,
        bank,
        _student_config(),
        tmp_path / "student",
        updates=8,
        batch_size=8,
        flow_samples=2,
        learning_rate=0.01,
    )
    assert all(
        torch.equal(value, before[key])
        for key, value in teacher.actor.state_dict().items()
    )
    assert report["history"][-1]["gradient_norm"] > 0
    assert report["history"][-1]["pmf_loss"] > 0
    replay = load_pmf_checkpoint(report["checkpoint"], weights="evaluation")
    validate_checkpoint_bank_contract(replay.checkpoint_path, bank)
    generated = replay.policy.transport_actions(
        bank["raw_observations"], bank["initial_noise"]
    )
    assert torch.isfinite(generated).all()
    assert generated.shape == (3, 2, 2)
    assert replay.terminal_sigma == 0.0
    checkpoint = torch.load(report["checkpoint"], weights_only=False)
    assert checkpoint["infos"]["rl_resume_supported"] is False
