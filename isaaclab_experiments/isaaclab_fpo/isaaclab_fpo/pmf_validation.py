"""Checkpoint diagnostics with reusable raw-state and Gaussian-noise banks.

This module imports no simulator APIs.  A bank deliberately stores observations
before empirical normalization: each checkpoint must use its own saved actor
normalizer when comparing one-step transport with its instantaneous ODE field.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _json_value(value: Any, depth: int = 0) -> Any:
    """Represent resolved configs without importing their simulator classes."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if depth >= 20:
        return f"{type(value).__module__}.{type(value).__qualname__}"
    if isinstance(value, Mapping):
        return {str(key): _json_value(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item, depth + 1) for item in value]
    if callable(value):
        return f"{value.__module__}.{getattr(value, '__qualname__', type(value).__qualname__)}"
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_value(to_dict(), depth + 1)
    if hasattr(value, "__dict__"):
        return {
            str(key): _json_value(item, depth + 1)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment_contract(runner: Any, task_name: str | None) -> dict[str, Any]:
    """Capture observation order and action coordinates, beyond tensor widths."""
    env = runner.env
    base = getattr(env, "unwrapped", env)
    cfg = getattr(base, "cfg", None)
    spec = getattr(base, "spec", None)
    manager = getattr(base, "observation_manager", None)
    names = getattr(manager, "active_terms", {})
    dimensions = getattr(manager, "group_obs_term_dim", {})
    action_manager = getattr(base, "action_manager", None)
    sim_cfg = getattr(cfg, "sim", None)
    observation_cfg = getattr(getattr(cfg, "observations", None), "policy", None)
    contract = {
        "task_name": task_name or getattr(spec, "id", None),
        "environment_class": f"{type(base).__module__}.{type(base).__qualname__}",
        "actor_observation_dim": runner.alg.policy.num_actor_obs,
        "action_dim": runner.alg.policy.num_actions,
        "policy_observation_terms": _json_value(names.get("policy")),
        "policy_observation_term_dims": _json_value(dimensions.get("policy")),
        "policy_observation_cfg": _json_value(observation_cfg),
        "action_terms": _json_value(getattr(action_manager, "active_terms", None)),
        "actions_cfg": _json_value(getattr(cfg, "actions", None)),
        "commands_cfg": _json_value(getattr(cfg, "commands", None)),
        "sim_dt": _json_value(getattr(sim_cfg, "dt", None)),
        "gravity": _json_value(getattr(sim_cfg, "gravity", None)),
        "decimation": _json_value(getattr(cfg, "decimation", None)),
        "wrapper_action_clip": _json_value(getattr(env, "clip_actions", None)),
    }
    scene = getattr(base, "scene", None)
    if scene is not None:
        try:
            contract["robot_joint_names"] = list(scene["robot"].joint_names)
        except (KeyError, TypeError, AttributeError):
            pass
    return contract


def _bank_fingerprint(bank: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(bank["environment_contract"], sort_keys=True).encode())
    for key in ("raw_observations", "initial_noise"):
        tensor = bank[key].detach().cpu().contiguous()
        digest.update(f"{key}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _validate_bank(bank: Any, contract: dict[str, Any]) -> None:
    if not isinstance(bank, dict) or bank.get("version") != 1:
        raise ValueError("Unsupported pMF validation state-bank format")
    if bank.get("observation_coordinates") != "raw":
        raise ValueError("State bank must contain raw, unnormalized observations")
    saved_contract = bank.get("environment_contract")
    if saved_contract != contract:
        differing = [
            key
            for key in contract
            if not isinstance(saved_contract, dict)
            or saved_contract.get(key) != contract[key]
        ]
        raise ValueError(
            f"State-bank environment contract mismatch: {', '.join(differing)}"
        )
    obs, noise = bank.get("raw_observations"), bank.get("initial_noise")
    for key, tensor in (("raw_observations", obs), ("initial_noise", noise)):
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            raise ValueError(f"State-bank {key} must be a floating tensor")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"State-bank {key} contains nonfinite values")
    if (
        obs.ndim != 2
        or obs.shape[0] < 1
        or obs.shape[1] != contract["actor_observation_dim"]
    ):
        raise ValueError("State-bank actor observation dimensions do not match")
    if noise.ndim != 3 or noise.shape[0] != obs.shape[0] or noise.shape[1] < 1:
        raise ValueError("State-bank noise must have shape [states, samples, actions]")
    if noise.shape[2] != contract["action_dim"]:
        raise ValueError("State-bank action dimensions do not match")
    if bank.get("noise_distribution") != "standard_normal":
        raise ValueError(
            "State bank must declare its standard-normal noise distribution"
        )


def _collect_bank(
    runner: Any,
    contract: dict[str, Any],
    checkpoint_path: Path,
    *,
    num_states: int,
    samples_per_state: int,
    seed: int,
    warmup_steps: int,
    collection_steps: int | None,
) -> dict[str, Any]:
    env, policy = runner.env, runner.alg.policy
    parameter = next(policy.parameters())
    device, dtype = parameter.device, parameter.dtype
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rollout_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    num_envs = int(env.num_envs)
    if collection_steps is None:
        collection_steps = max(16, math.ceil(num_states / min(num_envs, 16)))
    if (
        isinstance(collection_steps, bool)
        or not isinstance(collection_steps, int)
        or collection_steps < 1
        or collection_steps * num_envs < num_states
    ):
        raise ValueError("collection_steps * num_envs must cover num_states")

    obs, _ = env.reset()
    episode_returns = torch.zeros(num_envs, device=device, dtype=torch.float64)
    episode_lengths = torch.zeros(num_envs, device=device, dtype=torch.long)
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    completed_timeout_count = 0
    episodes_with_timeout_info = 0
    states: list[torch.Tensor] = []
    sampled_env_ids: list[int] = []
    sampled_steps: list[int] = []
    captured = 0
    rotation = 0

    for step in range(warmup_steps + collection_steps):
        obs = obs.to(device=device, dtype=dtype)
        if step >= warmup_steps:
            index = step - warmup_steps
            # Spread the exact requested state count over the whole finite
            # window; rotate environment indices instead of taking env 0 only.
            target = (index + 1) * num_states // collection_steps
            count = target - captured
            if count:
                ids = (torch.arange(count, device=device) + rotation) % num_envs
                states.append(obs[ids].detach().cpu().clone())
                sampled_env_ids.extend(ids.cpu().tolist())
                sampled_steps.extend([step] * count)
                rotation = (rotation + count) % num_envs
                captured = target

        noise = torch.randn((num_envs, policy.num_actions), generator=rollout_generator)
        noise = noise.to(device=device, dtype=dtype)
        # This exact one-step map excludes conditional residual action noise.
        actions = policy.transport_actions(runner.obs_normalizer(obs), noise)
        if not bool(torch.isfinite(actions).all()):
            raise ValueError(
                "Checkpoint produced nonfinite actions during state collection"
            )
        obs, rewards, dones, infos = env.step(actions.to(env.device))
        episode_returns += rewards.to(device=device, dtype=torch.float64).reshape(-1)
        episode_lengths += 1
        done = dones.to(device=device).reshape(-1).bool()
        if bool(done.any()):
            completed_returns.extend(episode_returns[done].cpu().tolist())
            completed_lengths.extend(episode_lengths[done].cpu().tolist())
            timeout = infos.get("time_outs")
            if timeout is None:
                timeout = getattr(
                    getattr(env, "unwrapped", env), "reset_time_outs", None
                )
            if timeout is not None:
                timeout = torch.as_tensor(timeout, device=device).reshape(-1).bool()
                completed_timeout_count += int((done & timeout).sum().item())
                episodes_with_timeout_info += int(done.sum().item())
            episode_returns[done] = 0
            episode_lengths[done] = 0

    count = len(completed_returns)
    collection = {
        "policy": "one_step_transport_without_residual_action_noise",
        "warmup_steps": warmup_steps,
        "collection_steps": collection_steps,
        "total_vector_steps": warmup_steps + collection_steps,
        "total_environment_steps": (warmup_steps + collection_steps) * num_envs,
        "num_envs": num_envs,
        "completed_episodes": count,
        "mean_completed_episode_return": sum(completed_returns) / count
        if count
        else None,
        "mean_completed_episode_length": sum(completed_lengths) / count
        if count
        else None,
        "completed_episode_timeout_rate": (
            completed_timeout_count / episodes_with_timeout_info
            if episodes_with_timeout_info
            else None
        ),
        "episodes_with_timeout_info": episodes_with_timeout_info,
        "unfinished_episodes": int((episode_lengths > 0).sum().item()),
        "episode_metrics_include_warmup": True,
        "completion_rule": "environment_done_only",
    }
    return {
        "version": 1,
        "observation_coordinates": "raw",
        "noise_distribution": "standard_normal",
        "environment_contract": contract,
        "raw_observations": torch.cat(states),
        "initial_noise": torch.randn(
            (num_states, samples_per_state, policy.num_actions), generator=generator
        ),
        "seed": seed,
        "source_checkpoint": str(checkpoint_path),
        "sampled_env_ids": sampled_env_ids,
        "sampled_vector_steps": sampled_steps,
        "collection": collection,
    }


def validate_pmf_checkpoint(
    runner: Any,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    state_bank_path: str | Path | None = None,
    num_states: int = 256,
    samples_per_state: int = 8,
    seed: int = 12345,
    ode_steps: Sequence[int] = (8, 32, 128),
    batch_size: int = 256,
    warmup_steps: int = 32,
    collection_steps: int | None = None,
    task_name: str | None = None,
    state_bank_output_path: str | Path | None = None,
    solver: str = "heun",
    time_eps: float | None = None,
    fields: Sequence[str] = ("diagonal_mean", "auxiliary"),
) -> dict[str, Any]:
    """Write ``report.json`` and a reusable ``state_bank.pt`` for one checkpoint.

    An existing bank supplies the authoritative state/sample counts and does
    not step or reset the environment.  Newly collected banks use a bounded
    rollout; incomplete episodes never enter the reported return or length.
    The model and both normalizers, including their training modes, are
    restored even if loading, collection, or diagnostics raises.  Python,
    NumPy, and Torch random states are also preserved; newly collected banks
    seed the environment before reset.  Environment state itself is advanced
    when collecting a new bank.
    """
    from isaaclab_fpo.modules.pmf_consistency import evaluate_pmf_consistency

    for name, value in (
        ("num_states", num_states),
        ("samples_per_state", samples_per_state),
        ("batch_size", batch_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (
        isinstance(warmup_steps, bool)
        or not isinstance(warmup_steps, int)
        or warmup_steps < 0
    ):
        raise ValueError("warmup_steps must be a nonnegative integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63 - 1:
        raise ValueError("seed must be an integer in [0, 2**63 - 1)")
    if not callable(getattr(runner.alg.policy, "transport_actions", None)):
        raise TypeError("pMF checkpoint validation requires a pMF transport policy")

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing its evaluation model_state_dict")
    compatibility_check = getattr(runner, "_validate_checkpoint_compatibility", None)
    if callable(compatibility_check):
        compatibility_check(checkpoint, require_online_state=False)
    policy = runner.alg.policy
    modules = (policy, runner.obs_normalizer, runner.privileged_obs_normalizer)
    original_states = [copy.deepcopy(module.state_dict()) for module in modules]
    original_modes = [
        (child, child.training) for module in modules for child in module.modules()
    ]
    contract = _environment_contract(runner, task_name)
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state()
    # Snapshot only already initialized CUDA generators.  CPU/offline
    # validation must not initialize CUDA merely to preserve random state.
    cuda_rng_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    )

    try:
        # model_state_dict is the saved evaluation artifact; EMA may already
        # be baked into its actor.  Loading live EMA/optimizer state here would
        # silently replace the desired checkpoint actor or sampler settings.
        policy.load_state_dict(checkpoint["model_state_dict"])
        if runner.empirical_normalization:
            if "obs_norm_state_dict" not in checkpoint:
                raise ValueError(
                    "Normalized checkpoint is missing actor observation normalization"
                )
            runner.obs_normalizer.load_state_dict(checkpoint["obs_norm_state_dict"])
            if "privileged_obs_norm_state_dict" in checkpoint:
                runner.privileged_obs_normalizer.load_state_dict(
                    checkpoint["privileged_obs_norm_state_dict"]
                )
        elif "obs_norm_state_dict" in checkpoint:
            raise ValueError(
                "Checkpoint requires empirical normalization; enable it for validation"
            )
        runner.eval_mode()
        # Explicitly freeze these even for runner implementations whose eval
        # method does not manage normalizers.
        for module in modules:
            module.eval()

        with torch.no_grad():
            if state_bank_path is not None:
                source_bank_path = Path(state_bank_path).expanduser().resolve()
                bank = torch.load(
                    source_bank_path, map_location="cpu", weights_only=True
                )
            else:
                source_bank_path = None
                environment_seed = getattr(runner.env, "seed", None)
                if callable(environment_seed):
                    environment_seed(seed)
                random.seed(seed)
                np.random.seed(seed % 2**32)
                torch.random.default_generator.manual_seed(seed)
                if torch.cuda.is_initialized():
                    torch.cuda.manual_seed_all(seed)
                bank = _collect_bank(
                    runner,
                    contract,
                    checkpoint_path,
                    num_states=num_states,
                    samples_per_state=samples_per_state,
                    seed=seed,
                    warmup_steps=warmup_steps,
                    collection_steps=collection_steps,
                )
            _validate_bank(bank, contract)
            parameter = next(policy.parameters())
            normalized = runner.obs_normalizer(bank["raw_observations"].to(parameter))
            noise = bank["initial_noise"].to(parameter)
            metrics = evaluate_pmf_consistency(
                policy,
                normalized,
                noise,
                ode_steps=ode_steps,
                solver=solver,
                time_eps=time_eps,
                batch_size=batch_size,
                fields=fields,
                action_clip=getattr(runner.env, "clip_actions", None),
            )

        params_artifacts = {}
        for name in ("agent.yaml", "env.yaml"):
            path = checkpoint_path.parent / "params" / name
            if path.is_file():
                params_artifacts[name] = {
                    "path": str(path),
                    "sha256": _file_sha256(path),
                    "content": path.read_text(),
                }
        report = {
            "version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": _file_sha256(checkpoint_path),
                "iteration": _json_value(checkpoint.get("iter")),
                "model_state_key": "model_state_dict",
                "ema_policy": "use_saved_evaluation_artifact_without_reapplying_ema",
                "ema_state_present": "ema_state_dict" in checkpoint,
                "online_state_present": "online_model_state_dict" in checkpoint,
                "policy_class_name": checkpoint.get(
                    "policy_class_name", type(policy).__name__
                ),
                "algorithm_class_name": checkpoint.get("algorithm_class_name"),
                "algorithm_state": _json_value(checkpoint.get("algorithm_state_dict")),
                "actor_observation_dim": policy.num_actor_obs,
                "action_dim": policy.num_actions,
                "actor_scale": policy.actor_scale,
                "actor_mlp_output_scale": policy.mlp_output_scale,
                "pmf_time_eps": policy.pmf_time_eps,
                "empirical_normalization": runner.empirical_normalization,
                "actor_normalizer_sha256": _state_dict_fingerprint(
                    runner.obs_normalizer.state_dict()
                ),
                "runner_config": _json_value(runner.cfg),
                "training_objectives": {
                    "fsppo_joint_aux_loss_coef": getattr(
                        runner.cfg.algorithm, "fsppo_joint_aux_loss_coef", None
                    ),
                    "fsppo_joint_adv_v_loss_coef": getattr(
                        runner.cfg.algorithm, "fsppo_joint_adv_v_loss_coef", None
                    ),
                    "pmf_aux_v_loss_coef": getattr(policy, "pmf_aux_v_loss_coef", None),
                },
                "run_config_artifacts": params_artifacts,
            },
            "state_bank": {
                "input_path": str(source_bank_path) if source_bank_path else None,
                "output_path": str(output_dir / "state_bank.pt"),
                "fingerprint": _bank_fingerprint(bank),
                "num_states": bank["raw_observations"].shape[0],
                "samples_per_state": bank["initial_noise"].shape[1],
                "counts_from_loaded_bank": source_bank_path is not None,
                "seed": bank.get("seed"),
                "source_checkpoint": bank.get("source_checkpoint"),
                "environment_contract": contract,
                "observation_coordinates": "raw",
                "noise_distribution": "standard_normal",
            },
            "collection": bank.get("collection") if source_bank_path is None else None,
            "source_bank_collection": bank.get("collection")
            if source_bank_path is not None
            else None,
            "diagnostics": metrics,
            "interpretation": {
                "consistency_error_is_distribution_matching_proof": False,
                "conditional_action_variance_is_multimodality_proof": False,
                "reused_bank_provides_current_checkpoint_return": False,
            },
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(bank, output_dir / "state_bank.pt")
        if state_bank_output_path is not None:
            destination = Path(state_bank_output_path).expanduser().resolve()
            if destination != output_dir / "state_bank.pt":
                destination.parent.mkdir(parents=True, exist_ok=True)
                torch.save(bank, destination)
            report["state_bank"]["export_path"] = str(destination)
        (output_dir / "report.json").write_text(
            json.dumps(_json_value(report), indent=2, allow_nan=False) + "\n"
        )
        return report
    finally:
        try:
            for module, state in zip(modules, original_states):
                module.load_state_dict(state)
            for module, training in original_modes:
                module.training = training
        finally:
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
            torch.random.set_rng_state(torch_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)


def _state_dict_fingerprint(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(f"{tensor.dtype}:{tuple(tensor.shape)}".encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            digest.update(json.dumps(_json_value(value), sort_keys=True).encode())
    return digest.hexdigest()
