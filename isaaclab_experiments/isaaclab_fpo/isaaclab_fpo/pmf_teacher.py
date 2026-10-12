"""Offline Gaussian-teacher distillation and fixed-state action diagnostics.

Inputs always contain raw observations. Each policy applies its own frozen
normalizer. This module imports Torch model APIs, never simulator APIs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml
from torch import nn

from isaaclab_fpo.modules import EmpiricalNormalization, PMFActorCritic
from isaaclab_fpo.pmf_validation import (
    _bank_fingerprint,
    _environment_contract,
    _validate_bank,
)


class _ConfigLoader(yaml.SafeLoader):
    """Read serialized config metadata without constructing Python objects."""


def _yaml_tag(loader: yaml.SafeLoader, tag: str, node: yaml.Node) -> Any:
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    value = loader.construct_scalar(node)
    return tag.removeprefix("python/name:") if tag.startswith("python/name:") else value


_ConfigLoader.add_multi_constructor("tag:yaml.org,2002:", _yaml_tag)


def read_agent_config(path: str | Path) -> dict[str, Any]:
    """Read a saved YAML configuration, including inert Python tuple tags."""
    with Path(path).open() as stream:
        config = yaml.load(stream, Loader=_ConfigLoader)
    if not isinstance(config, dict):
        raise TypeError(f"Expected mapping in {path}")
    return config


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _canonical(item)
            for key, item in value.items()
            if not key.startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    # Saved YAML uses module:qualname; validation banks serialize callables
    # as module.qualname. These describe the same resolved config term.
    if (
        isinstance(value, str)
        and ":" in value
        and not value.startswith(("/", "http", "omniverse"))
    ):
        return value.replace(":", ".")
    return value


def validate_checkpoint_bank_contract(
    checkpoint_path: str | Path,
    bank: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reject incompatible observation/action ordering and control coordinates.

    A simulator-created bank additionally records resolved joint names. Saved
    env.yaml supplies observation terms, action selectors/scales, commands,
    gravity and control timing; those are checked without importing task code.
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    config = config or read_agent_config(
        checkpoint_path.parent / "params" / "agent.yaml"
    )
    env_path = checkpoint_path.parent / "params" / "env.yaml"
    if not env_path.is_file():
        raise ValueError(
            "Checkpoint requires params/env.yaml to verify its environment contract"
        )
    env = read_agent_config(env_path)
    contract = bank["environment_contract"]
    fields = {
        "policy_observation_cfg": env.get("observations", {}).get("policy"),
        "actions_cfg": env.get("actions"),
        "commands_cfg": env.get("commands"),
        "sim_dt": env.get("sim", {}).get("dt"),
        "gravity": env.get("sim", {}).get("gravity"),
        "decimation": env.get("decimation"),
        "wrapper_action_clip": config.get("clip_actions"),
    }
    differing = [
        key
        for key, value in fields.items()
        if _canonical(contract.get(key)) != _canonical(value)
    ]
    if differing:
        raise ValueError(
            f"State bank/checkpoint environment contract mismatch: {', '.join(differing)}"
        )
    return {
        "env_config_path": str(env_path),
        "checked_fields": list(fields),
        "runtime_joint_order": "recorded by bank; saved action selectors and preserve_order checked",
    }


def load_raw_state_bank(path: str | Path) -> dict[str, Any]:
    """Load the existing validation-bank format, rejecting normalized states."""
    bank = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(bank, dict) or not isinstance(
        bank.get("environment_contract"), dict
    ):
        raise TypeError("State bank requires an environment_contract")
    _validate_bank(bank, bank["environment_contract"])
    return bank


@dataclass
class GaussianTeacher:
    """Frozen native RSL-RL MLP actor with its own observation normalizer."""

    actor: nn.Module
    config: dict[str, Any]
    checkpoint_path: Path
    num_actor_obs: int
    num_actions: int

    @torch.no_grad()
    def moments(
        self, raw_observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Gaussian mean/std in public action coordinates."""
        if (
            raw_observations.ndim != 2
            or raw_observations.shape[-1] != self.num_actor_obs
        ):
            raise ValueError(
                "Teacher observation dimensions do not match the raw state bank"
            )
        parameter = next(self.actor.parameters())
        values = raw_observations.to(parameter)
        self.actor.distribution.update(
            self.actor.mlp(self.actor.obs_normalizer(values))
        )
        return self.actor.output_mean, self.actor.output_std

    @torch.no_grad()
    def sample(
        self, raw_observations: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample the full teacher Gaussian, optionally with prescribed noise."""
        mean, std = self.moments(raw_observations)
        noise = torch.randn_like(mean) if noise is None else noise.to(mean)
        if noise.shape != mean.shape:
            raise ValueError("Teacher noise shape must match [batch, action_dim]")
        return mean + std * noise


@torch.no_grad()
def collect_teacher_state_bank(
    env: Any,
    teacher: GaussianTeacher,
    task_name: str,
    *,
    num_states: int = 256,
    samples_per_state: int = 8,
    warmup_steps: int = 128,
    collection_steps: int | None = None,
    seed: int = 12345,
) -> dict[str, Any]:
    """Collect raw states from a frozen teacher through a native vector env.

    This function accepts the native RSL-RL TensorDict wrapper or the external
    FPO wrapper. It does not construct/import any simulator class. Sampling
    uses teacher stochastic actions and a separately seeded state selection.
    """
    if num_states < 1 or samples_per_state < 1 or warmup_steps < 0:
        raise ValueError("Invalid teacher bank collection counts")
    collection_steps = (
        max(16, math.ceil(num_states / env.num_envs))
        if collection_steps is None
        else collection_steps
    )
    if collection_steps < 1 or collection_steps * env.num_envs < num_states:
        raise ValueError("collection_steps must provide enough distinct vector states")
    runner = SimpleNamespace(env=env, alg=SimpleNamespace(policy=teacher))
    contract = _environment_contract(runner, task_name)
    prototype = {
        "version": 1,
        "observation_coordinates": "raw",
        "noise_distribution": "standard_normal",
        "environment_contract": contract,
        "raw_observations": torch.zeros(1, teacher.num_actor_obs),
        "initial_noise": torch.zeros(1, samples_per_state, teacher.num_actions),
    }
    validate_checkpoint_bank_contract(
        teacher.checkpoint_path, prototype, config=teacher.config
    )
    if env.num_actions != teacher.num_actions:
        raise ValueError("Teacher action dimension differs from environment")
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def raw_policy(observations: Any) -> torch.Tensor:
        observations = (
            observations[0] if isinstance(observations, tuple) else observations
        )
        values = (
            observations
            if isinstance(observations, torch.Tensor)
            else observations["policy"]
        )
        if values.ndim != 2 or values.shape[1] != teacher.num_actor_obs:
            raise ValueError("Teacher observation dimension differs from environment")
        return values

    if callable(getattr(env, "seed", None)):
        env.seed(seed)
    observations = raw_policy(env.reset())
    states, sampled_ids, sampled_steps = [], [], []
    collected = 0
    for step in range(warmup_steps + collection_steps):
        if step >= warmup_steps and collected < num_states:
            remaining_steps = warmup_steps + collection_steps - step
            count = min(
                env.num_envs, math.ceil((num_states - collected) / remaining_steps)
            )
            indices = torch.randperm(env.num_envs, generator=generator)[:count]
            states.append(observations[indices.to(observations.device)].detach().cpu())
            sampled_ids.extend(indices.tolist())
            sampled_steps.extend([step] * count)
            collected += count
        noise = torch.randn(env.num_envs, teacher.num_actions, generator=generator)
        observations = raw_policy(env.step(teacher.sample(observations, noise))[0])
    bank = {
        "version": 1,
        "observation_coordinates": "raw",
        "noise_distribution": "standard_normal",
        "environment_contract": contract,
        "raw_observations": torch.cat(states),
        "initial_noise": torch.randn(
            num_states, samples_per_state, teacher.num_actions, generator=generator
        ),
        "seed": seed,
        "source_checkpoint": str(teacher.checkpoint_path),
        "sampled_env_ids": sampled_ids,
        "sampled_vector_steps": sampled_steps,
        "collection": {
            "behavior": "frozen_native_ppo_full_gaussian",
            "warmup_steps": warmup_steps,
            "collection_steps": collection_steps,
            "vector_env_count": env.num_envs,
            "task_performance_evaluation": False,
        },
    }
    _validate_bank(bank, contract)
    return bank


def load_native_ppo_teacher(
    checkpoint_path: str | Path, device: str = "cpu"
) -> GaussianTeacher:
    """Load the active RSL-RL 5 MLPModel/GaussianDistribution checkpoint schema."""
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    config = read_agent_config(checkpoint_path.parent / "params" / "agent.yaml")
    actor_cfg = copy.deepcopy(config.get("actor", {}))
    if actor_cfg.pop("class_name", None) not in ("MLPModel", "rsl_rl.models.MLPModel"):
        raise ValueError("Teacher must use native RSL-RL MLPModel")
    distribution_cfg = actor_cfg.get("distribution_cfg", {})
    if distribution_cfg.get("class_name") not in (
        "GaussianDistribution",
        "rsl_rl.modules.GaussianDistribution",
    ):
        raise ValueError("Teacher must use a state-independent GaussianDistribution")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("actor_state_dict")
    if not isinstance(state, dict) or "mlp.0.weight" not in state:
        raise ValueError("Teacher checkpoint is missing native actor_state_dict")
    obs_dim = state["mlp.0.weight"].shape[1]
    std_key = (
        "distribution.std_param"
        if distribution_cfg.get("std_type", "scalar") == "scalar"
        else "distribution.log_std_param"
    )
    if std_key not in state:
        raise ValueError(f"Teacher Gaussian checkpoint is missing {std_key}")
    action_dim = state[std_key].numel()
    groups = config.get("obs_groups", {}).get("actor")
    if groups != ["policy"]:
        raise ValueError(
            "Raw policy state bank currently requires teacher obs_groups.actor == ['policy']"
        )
    actor = MLPModel(
        TensorDict({"policy": torch.zeros(1, obs_dim)}, batch_size=[1]),
        config["obs_groups"],
        "actor",
        action_dim,
        **actor_cfg,
    ).to(device)
    actor.load_state_dict(state, strict=True)
    actor.eval().requires_grad_(False)
    return GaussianTeacher(actor, config, checkpoint_path, obs_dim, action_dim)


@dataclass
class PMFCheckpoint:
    """pMF actor and its independent frozen observation coordinates."""

    policy: PMFActorCritic
    normalizer: nn.Module
    config: dict[str, Any]
    checkpoint_path: Path
    weight_selection: str
    terminal_sigma: float
    completed_env_transitions: int | None
    terminal_std: torch.Tensor


def _pmf_dimensions(
    state: dict[str, torch.Tensor], cfg: dict[str, Any]
) -> tuple[int, int, int]:
    state = {
        key.replace("actor.flow.", "actor."): value for key, value in state.items()
    }
    architecture = cfg.get("actor_architecture", "mlp")
    if architecture == "mlp":
        layers = sorted(
            (int(key.split(".")[1]), value)
            for key, value in state.items()
            if key.startswith("actor.") and key.endswith(".weight")
        )
        action_dim = layers[-1][1].shape[0] // 2
        obs_dim = layers[0][1].shape[1] - int(cfg["timestep_embed_dim"]) - action_dim
    elif architecture == "time_conditioned_mlp":
        action_dim = state["actor.output_layer.weight"].shape[0] // 2
        obs_dim = state["actor.input_layer.weight"].shape[1] - action_dim
    else:
        raise ValueError(f"Unsupported PMF backbone: {architecture}")
    return obs_dim, state["critic.0.weight"].shape[1], action_dim


def load_pmf_checkpoint(
    checkpoint_path: str | Path, device: str = "cpu", *, weights: str = "online"
) -> PMFCheckpoint:
    """Restore pMF weights, normalizer and saved next-rollout noise schedule."""
    if weights not in ("online", "evaluation"):
        raise ValueError("weights must be online or evaluation")
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    config = read_agent_config(checkpoint_path.parent / "params" / "agent.yaml")
    policy_cfg = config["policy"]
    class_name = policy_cfg.get("class_name")
    if class_name not in ("PMFActorCritic", "LearnedStdPMFActorCritic"):
        raise ValueError("Action diagnostics require a PMFActorCritic transport policy")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    key = "online_model_state_dict" if weights == "online" else "model_state_dict"
    if key not in checkpoint:
        if weights == "online" and "ema_state_dict" in checkpoint:
            raise ValueError(
                "Checkpoint has EMA but no saved online weights; select evaluation explicitly"
            )
        key = "model_state_dict"
    state = checkpoint[key]
    obs_dim, critic_dim, action_dim = _pmf_dimensions(state, policy_cfg)
    policy_class = PMFActorCritic
    if class_name == "LearnedStdPMFActorCritic":
        from isaaclab_fpo.modules.learned_std_actor_critic import (
            LearnedStdPMFActorCritic,
        )

        policy_class = LearnedStdPMFActorCritic
    policy = policy_class(
        obs_dim, critic_dim, action_dim, SimpleNamespace(**policy_cfg)
    ).to(device)
    policy.load_state_dict(state, strict=True)
    normalizer = (
        EmpiricalNormalization(obs_dim).to(device)
        if config.get("empirical_normalization")
        else nn.Identity()
    )
    if config.get("empirical_normalization"):
        if "obs_norm_state_dict" not in checkpoint:
            raise ValueError(
                "Normalized pMF checkpoint lacks its observation normalizer"
            )
        normalizer.load_state_dict(checkpoint["obs_norm_state_dict"], strict=True)
    elif "obs_norm_state_dict" in checkpoint:
        raise ValueError(
            "pMF checkpoint/config disagree about observation normalization"
        )
    transitions = checkpoint.get("fsppo_joint_runner_state_dict", {}).get(
        "tot_timesteps"
    )
    sigma = float(policy_cfg["action_perturb_std"])
    algorithm = config["algorithm"]
    decay = int(algorithm.get("action_perturb_std_decay_env_steps", 0))
    final = algorithm.get("action_perturb_std_final")
    if decay > 0 and final is not None:
        if transitions is None:
            raise ValueError(
                "Annealed-noise checkpoint lacks saved completed environment transitions"
            )
        sigma += (float(final) - sigma) * min(1.0, transitions / decay)
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("Invalid terminal Gaussian noise scale")
    policy.action_perturb_std = sigma
    terminal_std = (
        policy.conditional_action_std().detach().cpu()
        if class_name == "LearnedStdPMFActorCritic"
        else torch.full((action_dim,), sigma)
    )
    sigma = terminal_std.mean().item()
    policy.eval().requires_grad_(False)
    normalizer.eval().requires_grad_(False)
    return PMFCheckpoint(
        policy,
        normalizer,
        config,
        checkpoint_path,
        key,
        sigma,
        transitions,
        terminal_std,
    )


def conditional_action_covariance(actions: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute unbiased covariance within each fixed state, shape [S, K, A]."""
    if actions.ndim != 3 or actions.shape[1] < 2 or not torch.isfinite(actions).all():
        raise ValueError("Action samples must be finite [states, samples>=2, actions]")
    actions = actions.double()
    mean = actions.mean(dim=1)
    centered = actions - mean[:, None]
    covariance = centered.transpose(1, 2) @ centered / (actions.shape[1] - 1)
    # Independent same-state samples, not consecutive simulation control actions.
    pairwise_delta_rms = (2 * covariance.diagonal(dim1=-2, dim2=-1).mean()).sqrt()
    return {
        "mean": mean,
        "covariance": covariance,
        "std": covariance.diagonal(dim1=-2, dim2=-1).clamp_min(0).sqrt(),
        "independent_action_delta_rms": pairwise_delta_rms,
    }


def _summarize_samples(actions: torch.Tensor) -> dict[str, Any]:
    stats = conditional_action_covariance(actions)
    covariance = stats["covariance"].mean(dim=0)
    variance = covariance.diagonal()
    denominator = variance.sqrt()[:, None] * variance.sqrt()[None, :]
    correlation = covariance / denominator.clamp_min(1e-15)
    off_diagonal = correlation - torch.diag(correlation.diagonal())
    return {
        "mean_action": stats["mean"].mean(dim=0).tolist(),
        "rms_conditional_std_per_action": variance.clamp_min(0).sqrt().tolist(),
        "mean_conditional_covariance": covariance.tolist(),
        "mean_covariance_trace": covariance.trace().item(),
        "max_abs_off_diagonal_correlation": off_diagonal.abs().max().item(),
        "independent_same_state_action_delta_rms": stats[
            "independent_action_delta_rms"
        ].item(),
    }


@torch.no_grad()
def covariance_report(
    teacher: GaussianTeacher,
    pmf: PMFCheckpoint,
    bank: dict[str, Any],
    *,
    samples_per_state: int = 256,
    seed: int = 12345,
    batch_size: int = 64,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Compare PPO's full Gaussian with pMF latent and terminal noise sources."""
    if samples_per_state < 2 or batch_size < 1:
        raise ValueError("samples_per_state >= 2 and batch_size > 0 are required")
    obs = bank["raw_observations"]
    if (obs.shape[1], pmf.policy.num_actions) != (
        teacher.num_actor_obs,
        teacher.num_actions,
    ):
        raise ValueError("Teacher/pMF observation and action dimensions differ")
    if pmf.policy.num_actor_obs != teacher.num_actor_obs:
        raise ValueError("pMF observation dimensions differ from teacher")
    if bank["environment_contract"]["action_dim"] != teacher.num_actions:
        raise ValueError("State bank action dimension differs from checkpoints")
    contracts = {
        "teacher": validate_checkpoint_bank_contract(
            teacher.checkpoint_path, bank, config=teacher.config
        ),
        "pmf": validate_checkpoint_bank_contract(
            pmf.checkpoint_path, bank, config=pmf.config
        ),
    }
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ppo_actions, transports, joint_actions = [], [], []
    ppo_stds = []
    for chunk in obs.split(batch_size):
        shape = (len(chunk), samples_per_state, teacher.num_actions)
        latent = torch.randn(shape, generator=generator)
        terminal = torch.randn(shape, generator=generator)
        teacher_noise = torch.randn(shape, generator=generator)
        mean, std = teacher.moments(chunk)
        ppo_actions.append(
            (mean[:, None] + std[:, None] * teacher_noise.to(mean)).cpu()
        )
        ppo_stds.append(std.cpu())
        parameter = next(pmf.policy.parameters())
        mapped = pmf.policy.transport_actions(
            pmf.normalizer(chunk.to(parameter)), latent.to(parameter)
        )
        transports.append(mapped.cpu())
        joint_actions.append(
            (mapped + pmf.terminal_std.to(parameter) * terminal.to(parameter)).cpu()
        )
    samples = {
        "ppo_full_gaussian": torch.cat(ppo_actions),
        "pmf_latent_transport": torch.cat(transports),
        "pmf_executed_raw": torch.cat(joint_actions),
    }
    latent_cov = conditional_action_covariance(samples["pmf_latent_transport"])[
        "covariance"
    ]
    full_cov = conditional_action_covariance(samples["pmf_executed_raw"])["covariance"]
    terminal_cov = torch.diag(pmf.terminal_std.double().square())
    predicted = latent_cov + terminal_cov
    clip = bank["environment_contract"].get("wrapper_action_clip")
    report = {
        "version": 1,
        "state_bank_fingerprint": _bank_fingerprint(bank),
        "num_states": len(obs),
        "samples_per_state": samples_per_state,
        "seed": seed,
        "observation_coordinates": "raw, normalized independently inside each checkpoint",
        "teacher_checkpoint": str(teacher.checkpoint_path),
        "pmf_checkpoint": str(pmf.checkpoint_path),
        "pmf_weights": pmf.weight_selection,
        "pmf_actor_scale": float(pmf.policy.actor_scale),
        "wrapper_action_clip": clip,
        "terminal_sigma": pmf.terminal_sigma,
        "terminal_std_per_action": pmf.terminal_std.tolist(),
        "terminal_sigma_summary": "mean of per-action terminal standard deviations",
        "sigma_coordinates": "public actions after actor_scale, before wrapper clipping",
        "sigma_schedule_position": (
            "learned per-action diagonal scale from selected checkpoint weights"
            if pmf.config["policy"].get("learn_action_perturb_std", False)
            else "next rollout boundary at saved completed environment transitions"
        ),
        "completed_env_transitions": pmf.completed_env_transitions,
        "ppo_gaussian_std_per_action": torch.cat(ppo_stds).mean(dim=0).tolist(),
        "pmf_terminal_covariance": terminal_cov.tolist(),
        "pmf_latent_plus_terminal_covariance": predicted.mean(dim=0).tolist(),
        "empirical_covariance_law_residual_frobenius": (full_cov - predicted)
        .norm(dim=(-2, -1))
        .mean()
        .item(),
        "contracts": contracts,
        "limitations": [
            "Monte Carlo conditional covariance; finite-sample covariance-law residual is expected",
            "Independent same-state action delta does not measure temporal action-rate in simulation",
            "Fixed state bank does not establish closed-loop task performance",
            "Zero latent is a designated sample, not the marginal policy expectation",
        ],
        "distributions": {
            name: _summarize_samples(values) for name, values in samples.items()
        },
    }
    if clip is not None:
        for name in ("ppo_full_gaussian", "pmf_executed_raw"):
            samples[name + "_wrapper_clipped"] = samples[name].clamp(
                -float(clip), float(clip)
            )
            report["distributions"][name + "_wrapper_clipped"] = _summarize_samples(
                samples[name + "_wrapper_clipped"]
            )
            report["distributions"][name]["wrapper_clipped_fraction"] = (
                (samples[name].abs() > float(clip)).float().mean().item()
            )
    return report, samples


def distill_teacher(
    teacher: GaussianTeacher,
    bank: dict[str, Any],
    student_config: dict[str, Any],
    output_dir: str | Path,
    *,
    updates: int = 5000,
    batch_size: int = 256,
    flow_samples: int = 4,
    learning_rate: float = 1e-4,
    seed: int = 42,
    device: str = "cpu",
    ode_steps: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Fit a fresh pMF to stochastic PPO actions using only plain pMF loss.

    Teacher samples, flow endpoint noise and training times are independently
    drawn on every update. There are no PPO ratios, critic updates, advantage
    weights or environment interactions in this diagnostic experiment.
    """
    if (
        updates < 1
        or batch_size < 1
        or flow_samples < 1
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError(
            "Positive updates, batch size, flow samples and learning rate required"
        )
    output_dir = Path(output_dir).expanduser().resolve()
    if (output_dir / "model_distilled.pt").exists():
        raise FileExistsError(f"Distilled checkpoint already exists in {output_dir}")
    contract_check = validate_checkpoint_bank_contract(
        teacher.checkpoint_path, bank, config=teacher.config
    )
    raw = bank["raw_observations"]
    if (
        raw.shape[1] != teacher.num_actor_obs
        or bank["environment_contract"]["action_dim"] != teacher.num_actions
    ):
        raise ValueError("Teacher dimensions do not match state bank")
    config = copy.deepcopy(student_config)
    policy_cfg = config["policy"]
    # Hand-written experiment YAML omits fields normally supplied by rl_cfg.
    # Resolve the small Torch actor defaults here without importing Isaac Lab.
    for name, value in {
        "timestep_embed_dim": 8,
        "cfm_loss_t_inverse_cdf_beta": 1.0,
        "cfm_loss_reduction": "sqrt",
        "actor_mlp_output_scale": 1.0,
        "actor_final_layer_weight_scale": None,
        "actor_scale": 1.0,
        "pmf_logit_normal_mean": -0.4,
        "pmf_logit_normal_std": 1.0,
        "pmf_fm_proportion": 0.5,
        "pmf_aux_v_loss_coef": 1.0,
        "pmf_time_eps": 0.05,
        "pmf_adaptive_gradient_norm_p": 0.0,
        "pmf_adaptive_gradient_norm_eps": 0.01,
    }.items():
        policy_cfg.setdefault(name, value)
    if policy_cfg.get("class_name") not in (
        "PMFActorCritic",
        "LearnedStdPMFActorCritic",
    ):
        raise ValueError("Student must be PMFActorCritic")
    policy_cfg["class_name"] = "PMFActorCritic"
    policy_cfg["sampling_steps"] = 1
    policy_cfg["training_sampling_steps"] = 1
    # Teacher targets already include the full Gaussian exploration. Readding
    # terminal exploration would double count it during student playback.
    policy_cfg["action_perturb_std"] = 0.0
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        student = PMFActorCritic(
            teacher.num_actor_obs,
            teacher.num_actor_obs,
            teacher.num_actions,
            SimpleNamespace(**policy_cfg),
        ).to(device)
    normalizer = (
        EmpiricalNormalization(teacher.num_actor_obs).to(device)
        if config.get("empirical_normalization")
        else nn.Identity()
    )
    if config.get("empirical_normalization"):
        with torch.no_grad():
            normalizer(raw.to(device))
    normalizer.eval().requires_grad_(False)
    optimizer = torch.optim.Adam(student.actor.parameters(), lr=learning_rate)
    history = []
    for update in range(updates):
        indices = torch.randint(len(raw), (batch_size,), generator=generator)
        state = raw[indices]
        with torch.no_grad():
            mean, std = teacher.moments(state)
            target_noise = torch.randn(mean.shape, generator=generator).to(mean)
            targets = (mean + std * target_noise).to(device)
        eps = torch.randn(
            batch_size, flow_samples, teacher.num_actions, generator=generator
        ).to(device)
        times = torch.sigmoid(
            torch.randn(2, batch_size, flow_samples, 1, generator=generator)
            * student.pmf_logit_normal_std
            + student.pmf_logit_normal_mean
        )
        count = batch_size * flow_samples
        anchors = torch.randperm(count, generator=generator)[
            : int(count * student.pmf_fm_proportion)
        ]
        times[1].view(-1)[anchors] = times[0].view(-1)[anchors]
        r, t = times.min(dim=0).values.to(device), times.max(dim=0).values.to(device)
        score, _, _, components = student.get_pmf_loss(
            normalizer(state.to(device)), targets, eps, r, t, return_components=True
        )
        loss = score.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite pMF loss at update {update + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(student.actor.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(
                f"Nonfinite student gradient at update {update + 1}"
            )
        optimizer.step()
        if update == 0 or (update + 1) % 100 == 0 or update + 1 == updates:
            history.append(
                {
                    "update": update + 1,
                    "pmf_loss": loss.item(),
                    "u_loss": components["u_loss"].mean().item(),
                    "v_loss": components["v_loss"].mean().item(),
                    "gradient_norm": gradient_norm.item(),
                }
            )
    student.eval()
    checkpoint = {
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iter": updates,
        "infos": {
            "experiment": "ppo_teacher_distillation",
            "rl_resume_supported": False,
        },
        "policy_class_name": "PMFActorCritic",
        "algorithm_class_name": "PMFFPO",
    }
    if config.get("empirical_normalization"):
        checkpoint["obs_norm_state_dict"] = normalizer.state_dict()
        checkpoint["privileged_obs_norm_state_dict"] = normalizer.state_dict()
    config.update(
        seed=seed,
        device=device,
        resume=False,
        max_iterations=updates,
        enable_post_training_eval=False,
        run_name="ppo_teacher_distillation",
        flow_eval_modes=["random"],
        flow_eval_fixed_seed=seed,
    )
    config["algorithm"]["class_name"] = "PMFFPO"
    config["algorithm"]["ema_decay"] = 0.0
    config["algorithm"]["action_perturb_std_final"] = None
    config["algorithm"]["action_perturb_std_decay_env_steps"] = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "params").mkdir(exist_ok=True)
    torch.save(checkpoint, output_dir / "model_distilled.pt")
    with (output_dir / "params" / "agent.yaml").open("w") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    shutil.copyfile(
        teacher.checkpoint_path.parent / "params" / "env.yaml",
        output_dir / "params" / "env.yaml",
    )
    report = {
        "version": 1,
        "experiment": "frozen_gaussian_ppo_teacher_plain_pmf_distillation",
        "teacher_checkpoint": str(teacher.checkpoint_path),
        "teacher_checkpoint_sha256": hashlib.sha256(
            teacher.checkpoint_path.read_bytes()
        ).hexdigest(),
        "state_bank_fingerprint": _bank_fingerprint(bank),
        "state_bank_source_checkpoint": bank.get("source_checkpoint"),
        "environment_contract": bank["environment_contract"],
        "contract_check": contract_check,
        "updates": updates,
        "batch_size": batch_size,
        "flow_samples": flow_samples,
        "seed": seed,
        "learning_rate": learning_rate,
        "teacher_target": "stochastic full Gaussian public action, before wrapper clipping",
        "flow_noise": "fresh independent standard normal each update, independent of teacher target noise",
        "student_normalization": "student's own bank-fitted frozen statistics"
        if config.get("empirical_normalization")
        else "identity/raw observations",
        "checkpoint": str(output_dir / "model_distilled.pt"),
        "history": history,
        "limitations": [
            "Uses a pretrained PPO teacher and its state distribution; not a from-scratch sample-efficiency baseline",
            "Student critic is untrained; checkpoint supports policy evaluation, not RL resume",
            "Offline distribution fit does not establish closed-loop survival or tracking",
            "Evaluate stochastic pMF latent samples; latent zero is not the policy expectation",
        ],
    }
    with torch.no_grad():
        shape = (len(raw), 256, teacher.num_actions)
        latent = torch.randn(shape, generator=generator).to(device)
        generated = student.transport_actions(normalizer(raw.to(device)), latent).cpu()
        mean, std = teacher.moments(raw)
        sampled = (
            mean[:, None]
            + std[:, None] * torch.randn(shape, generator=generator).to(mean)
        ).cpu()
        report["fixed_state_distribution"] = {
            "teacher": _summarize_samples(sampled),
            "student": _summarize_samples(generated),
        }
        generated_stats = conditional_action_covariance(generated)
        report["fixed_state_distribution"]["mean_squared_mean_error"] = (
            (generated_stats["mean"] - mean.cpu()).square().mean().item()
        )
        report["fixed_state_distribution"]["mean_squared_covariance_error"] = (
            (
                generated_stats["covariance"]
                - torch.diag_embed(std.cpu().double().square())
            )
            .square()
            .mean()
            .item()
        )
        if ode_steps:
            from isaaclab_fpo.modules.pmf_consistency import evaluate_pmf_consistency

            report["flow_consistency"] = evaluate_pmf_consistency(
                student,
                normalizer(raw.to(device)),
                bank["initial_noise"].to(device),
                ode_steps=ode_steps,
                batch_size=batch_size,
                action_clip=config.get("clip_actions"),
            )
    with (output_dir / "distillation_report.json").open("w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report
