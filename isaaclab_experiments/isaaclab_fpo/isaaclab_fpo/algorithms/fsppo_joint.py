"""One-step latent Gaussian PPO with optional conditional-policy budgets."""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

import torch

from isaaclab_fpo.algorithms.fpo import FPO
from isaaclab_fpo.modules import (
    GaussianResidualPMFActorCritic,
    LearnedStdPMFActorCritic,
    PMFActorCritic,
)
from isaaclab_fpo.storage.joint_rollout_storage import JointRolloutStorage

if TYPE_CHECKING:
    from isaaclab_fpo.rl_cfg import FpoRslRlPpoAlgorithmCfg


class FSPPOJoint(FPO):
    """Optimize the joint policy ``p(xi) Normal(a; F_theta(obs, xi), sigma^2 I)``.

    Rollouts retain the actual generating latent and raw, perturbed action.
    The Gaussian conditional log-ratio is therefore exact for the joint policy;
    it is not the marginal action log-ratio. With sigma held positive and
    fixed for each rollout and its update, the squared same-latent map cost
    divided by ``2 * sigma**2`` is the joint KL. The Gaussian-residual policy
    and optional learned-std flow policies permit a learned diagonal scale
    and use its full Gaussian KL.
    Both KLs bound marginal action KL in expectation over the prior. The
    legacy PMF policy retains its historical L1 map budget for compatibility.

    When the budget is enabled, every candidate optimizer step is checked on
    one fixed probe set for this rollout.  A rejected candidate restores BOTH
    model and optimizer state and retries with a smaller step size.  This
    enforces only the empirical probe mean budget, not a population or all-state
    bound.  A fresh probe at the end measures generalization of that budget.
    The budget can be disabled for a controlled no-budget ablation.  Optional
    pMF regression is a positive auxiliary loss and never participates in the
    policy ratio.
    """

    # Logging-only map-distance estimate used by the cheap diagnostic mode.
    # The budgeted path keeps the full ``fsppo_joint_map_samples`` width
    # because its value enters gradients.
    diagnostic_obs_samples = 256
    diagnostic_map_samples = 2

    def __init__(
        self,
        policy: PMFActorCritic,
        cfg: FpoRslRlPpoAlgorithmCfg,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
    ):
        if not isinstance(policy, PMFActorCritic):
            raise TypeError("FSPPOJoint requires a PMFActorCritic policy")
        if multi_gpu_cfg is not None:
            raise ValueError(
                "FSPPOJoint currently supports a single learner device only"
            )
        if policy.sampling_steps != 1 or not 0 < policy.pmf_time_eps <= 1:
            raise ValueError(
                "FSPPOJoint requires sampling_steps=1 and 0 < pmf_time_eps <= 1"
            )
        for name, value in {
            "action_perturb_std": policy.action_perturb_std,
            "actor_scale": policy.actor_scale,
            "learning_rate": cfg.learning_rate,
            "max_grad_norm": cfg.max_grad_norm,
            "fsppo_joint_kl_target": cfg.fsppo_joint_kl_target,
        }.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(policy.mlp_output_scale):
            raise ValueError("actor_mlp_output_scale must be finite")
        if cfg.schedule != "fixed" or cfg.trust_region_mode != "ppo":
            raise ValueError(
                "FSPPOJoint requires schedule='fixed' and trust_region_mode='ppo'"
            )
        decay_env_steps = cfg.action_perturb_std_decay_env_steps
        self.gaussian_residual_policy = isinstance(
            policy, (GaussianResidualPMFActorCritic, LearnedStdPMFActorCritic)
        )
        self.learn_action_std = (
            self.gaussian_residual_policy and policy.learn_action_perturb_std
        )
        if self.learn_action_std and (
            decay_env_steps != 0 or cfg.action_perturb_std_final is not None
        ):
            raise ValueError(
                "Learned action noise requires action_perturb_std_decay_env_steps=0 "
                "and action_perturb_std_final=None"
            )
        if (
            isinstance(decay_env_steps, bool)
            or not isinstance(decay_env_steps, int)
            or decay_env_steps < 0
        ):
            raise ValueError(
                "action_perturb_std_decay_env_steps must be a non-negative integer"
            )
        final_action_std = cfg.action_perturb_std_final
        if final_action_std is None:
            if decay_env_steps > 0:
                raise ValueError(
                    "action_perturb_std_final must be set when action-noise decay is enabled"
                )
            final_action_std = float(policy.action_perturb_std)
        if not math.isfinite(final_action_std) or final_action_std <= 0:
            raise ValueError("action_perturb_std_final must be finite and positive")
        if final_action_std > policy.action_perturb_std:
            raise ValueError(
                "action_perturb_std_final must not exceed the initial action_perturb_std"
            )
        if decay_env_steps == 0 and final_action_std != policy.action_perturb_std:
            raise ValueError(
                "set action_perturb_std_decay_env_steps to anneal action noise"
            )
        if cfg.knn_entropy_coef != 0 or cfg.storage_action_noise_std != 0:
            raise ValueError(
                "FSPPOJoint requires knn_entropy_coef=0 and storage_action_noise_std=0"
            )
        if not math.isfinite(cfg.clip_param) or not 0 < cfg.clip_param < 1:
            raise ValueError("clip_param must be in (0, 1)")
        for name in (
            "fsppo_joint_kl_coef",
            "fsppo_joint_dual_lr",
            "fsppo_joint_kl_coef_max",
            "fsppo_joint_aux_loss_coef",
            "fsppo_joint_entropy_coef",
        ):
            value = getattr(cfg, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if cfg.fsppo_joint_kl_coef > cfg.fsppo_joint_kl_coef_max:
            raise ValueError("fsppo_joint_kl_coef exceeds fsppo_joint_kl_coef_max")
        for name in (
            "num_learning_epochs",
            "num_mini_batches",
            "fsppo_joint_map_samples",
            "fsppo_joint_probe_size",
            "fsppo_joint_replay_check_interval",
            "n_samples_per_action",
        ):
            value = getattr(cfg, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(cfg.fsppo_joint_max_backtracks, bool)
            or not isinstance(cfg.fsppo_joint_max_backtracks, int)
            or cfg.fsppo_joint_max_backtracks < 0
        ):
            raise ValueError(
                "fsppo_joint_max_backtracks must be a non-negative integer"
            )
        if not 0 < cfg.fsppo_joint_backtrack_factor < 1:
            raise ValueError("fsppo_joint_backtrack_factor must be in (0, 1)")
        if cfg.fsppo_joint_aux_weighting not in ("uniform", "advantage_exp"):
            raise ValueError(
                "fsppo_joint_aux_weighting must be 'uniform' or 'advantage_exp'"
            )
        if cfg.fsppo_joint_aux_target not in ("actions", "means"):
            raise ValueError("fsppo_joint_aux_target must be 'actions' or 'means'")
        if (
            not math.isfinite(cfg.fsppo_joint_aux_temperature)
            or cfg.fsppo_joint_aux_temperature <= 0
        ):
            raise ValueError("fsppo_joint_aux_temperature must be finite and positive")
        if (
            not math.isfinite(cfg.fsppo_joint_aux_weight_clip)
            or cfg.fsppo_joint_aux_weight_clip < 1
        ):
            raise ValueError(
                "fsppo_joint_aux_weight_clip must be finite and at least 1"
            )

        # The shared launcher enables TF32 ("high"). Different rollout and
        # minibatch GEMM shapes then change the same actor's means enough to
        # perturb a small-sigma log density. Use full FP32 consistently in
        # this single-learner process, including checkpoint playback.
        torch.set_float32_matmul_precision("highest")
        super().__init__(policy, cfg, device=device)
        self.transition = JointRolloutStorage.Transition()
        self.initial_sigma = float(policy.action_perturb_std)
        self.final_sigma = float(final_action_std)
        self.sigma_decay_env_steps = decay_env_steps
        self.sigma = self.initial_sigma
        self.action_noise_schedule_progress = 0.0
        self.action_noise_completed_env_steps = 0
        self.enable_budget = bool(cfg.fsppo_joint_enable_budget)
        self.kl_target = cfg.fsppo_joint_kl_target
        self.kl_coefficient = cfg.fsppo_joint_kl_coef if self.enable_budget else 0.0
        self.dual_lr = cfg.fsppo_joint_dual_lr
        self.kl_coefficient_max = cfg.fsppo_joint_kl_coef_max
        self.map_samples = cfg.fsppo_joint_map_samples
        self.probe_size = cfg.fsppo_joint_probe_size
        self.max_backtracks = cfg.fsppo_joint_max_backtracks
        self.backtrack_factor = cfg.fsppo_joint_backtrack_factor
        self.aux_loss_coef = cfg.fsppo_joint_aux_loss_coef
        self.aux_weighting = cfg.fsppo_joint_aux_weighting
        self.aux_temperature = cfg.fsppo_joint_aux_temperature
        self.aux_weight_clip = cfg.fsppo_joint_aux_weight_clip
        self.aux_target = cfg.fsppo_joint_aux_target
        self.entropy_coef = cfg.fsppo_joint_entropy_coef
        self.cheap_map_diagnostics = bool(cfg.fsppo_joint_cheap_map_diagnostics)
        self.replay_check_interval = cfg.fsppo_joint_replay_check_interval
        self._sampler_contract = self._current_sampler_contract()
        self._old_actor = copy.deepcopy(self.policy.actor).requires_grad_(False).eval()

    def begin_rollout(self, completed_env_steps: int) -> None:
        """Set the scheduled scale or retain the learned scale for a rollout.

        The schedule advances only at a rollout boundary. Each collected batch
        and its subsequent PPO update use that same fixed Gaussian scale.
        A learned Gaussian scale instead changes through the PPO update.
        """
        if (
            isinstance(completed_env_steps, bool)
            or not isinstance(completed_env_steps, int)
            or completed_env_steps < 0
        ):
            raise ValueError("completed_env_steps must be a non-negative integer")
        self._check_sampler_contract()
        self.action_noise_completed_env_steps = completed_env_steps
        if self.learn_action_std:
            # The scale is part of the actor and follows PPO gradients, not an
            # external schedule. Sampling and replay use the same saved actor.
            self.action_noise_schedule_progress = 0.0
            return
        if self.sigma_decay_env_steps > 0:
            progress = min(completed_env_steps / self.sigma_decay_env_steps, 1.0)
        else:
            progress = 0.0
        self.action_noise_schedule_progress = progress
        self.sigma = self.initial_sigma + progress * (
            self.final_sigma - self.initial_sigma
        )
        self.policy.action_perturb_std = self.sigma

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
    ):
        """Allocate rollout storage with separate generating-latent fields."""
        self.storage = JointRolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            device=self.device,
        )

    @torch.no_grad()
    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        """Sample and retain the raw action and its actual generating latent."""
        self._check_sampler_contract()
        actions, latent, mean, log_prob = self.policy.sample_transport(obs)
        # Keep the raw Gaussian sample even if an environment transforms its
        # input tensor in place (for example, action saturation).
        self.transition.actions = actions.clone()
        self.transition.action_latent = latent
        self.transition.action_mean = mean
        self.transition.action_log_prob = log_prob
        self.transition.action_std = self._action_std().expand_as(mean).clone()
        self.transition.values = self.policy.evaluate(critic_obs)
        # The runner may reuse observation buffers before process_env_step.
        self.transition.observations = obs.clone()
        self.transition.privileged_observations = critic_obs.clone()
        return actions

    def state_dict(self) -> dict:
        """Save adaptive penalty and the fixed sampling contract for resume."""
        return {
            "version": 1,
            "budget_enabled": self.enable_budget,
            "kl_coefficient": self.kl_coefficient,
            "update_counter": self.update_counter,
            "tot_timesteps": self.tot_timesteps,
            "sampler": dict(self._sampler_contract),
        }

    def load_state_dict(self, state: dict) -> None:
        """Reject incompatible sampling distributions before resuming."""
        if state.get("version") != 1 or state.get("sampler") != self._sampler_contract:
            raise ValueError("FSPPOJoint checkpoint sampling contract mismatch")
        # Checkpoints written before this flag existed used the budget path.
        if bool(state.get("budget_enabled", True)) != self.enable_budget:
            raise ValueError("FSPPOJoint checkpoint budget mode mismatch")
        coefficient = float(state["kl_coefficient"])
        if (
            not math.isfinite(coefficient)
            or not 0 <= coefficient <= self.kl_coefficient_max
        ):
            raise ValueError("Invalid FSPPOJoint checkpoint KL coefficient")
        for name in ("update_counter", "tot_timesteps"):
            value = state[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Invalid FSPPOJoint checkpoint {name}")
        self.kl_coefficient = coefficient if self.enable_budget else 0.0
        self.update_counter = state["update_counter"]
        self.tot_timesteps = state["tot_timesteps"]

    def update(self, obs_normalizer=None, privileged_obs_normalizer=None) -> dict:
        """Run PPO updates with optional transactional probe-budget checks."""
        del (
            obs_normalizer,
            privileged_obs_normalizer,
        )  # Stored inputs already include normalization.
        self._check_sampler_contract()
        if self.storage.step != self.storage.num_transitions_per_env:
            raise RuntimeError("FSPPOJoint.update requires a complete rollout")
        self._old_actor.load_state_dict(self.policy.actor.state_dict())
        probe = self._make_probe()
        # Prove that replay still refers to the behavior actor, not an EMA or
        # a policy changed between collection and update.  The sweep costs
        # about one rollout of actor replays, so thin it with
        # fsppo_joint_replay_check_interval (update counter 0 always runs).
        if self.update_counter % self.replay_check_interval == 0:
            self._validate_replay()
        aux_weight_metrics = self._prepare_aux_weights()
        totals: dict[str, torch.Tensor] = {}
        accepted = rejected = attempted = 0
        last_step_lr = 0.0
        coefficient_used = self.kl_coefficient if self.enable_budget else 0.0
        for batch in self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        ):
            attempted += 1
            loss, statistics = self._batch_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite FSPPOJoint loss; inspect raw observations/actions/returns"
                )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = self._clip_gradients(batch, statistics, batch_index=attempted)
            was_accepted, rejected_count, candidate_lr = self._step_with_budget(probe)
            rejected += rejected_count
            if not was_accepted:
                break
            accepted += 1
            last_step_lr = candidate_lr
            statistics["mean_grad_norm_before_clip"] = norm.detach()
            statistics["mean_grad_norm_after_clip"] = norm.detach() * (
                self.max_grad_norm / (norm.detach() + 1e-6)
            ).clamp(max=1.0)
            for name, value in statistics.items():
                value = value.detach()
                if name == "log_ratio_min":
                    totals[name] = torch.minimum(totals.get(name, value), value)
                elif name == "log_ratio_max":
                    totals[name] = torch.maximum(totals.get(name, value), value)
                else:
                    totals[name] = totals.get(name, 0.0) + value

        with torch.no_grad():
            if self.enable_budget:
                probe_kl = self._probe_distance(probe)
                # New states/noises are drawn AFTER all updates; these diagnostics
                # are not reused for acceptance or described as a hard guarantee.
                audit_kl = self._probe_distance(self._make_probe())
                if not torch.isfinite(audit_kl).all():
                    raise FloatingPointError(
                        "Non-finite FSPPOJoint fresh rollout map audit"
                    )
                audit_mean = float(audit_kl.mean())
                self.kl_coefficient = min(
                    self.kl_coefficient_max,
                    max(
                        0.0,
                        self.kl_coefficient
                        + self.dual_lr * (audit_mean / self.kl_target - 1.0),
                    ),
                )
            else:
                # self.kl_coefficient = 0.0
                pass
            variance = self.storage.returns.var(unbiased=False)
            explained = torch.where(
                variance > 1e-8,
                1
                - (self.storage.returns - self.storage.values).var(unbiased=False)
                / variance,
                variance.new_zeros(()),
            )
            action_std = self.storage.actions.std(dim=(0, 1), unbiased=False).mean()
            noise_std = self._action_std().detach()
        means = {
            key: value
            if key in ("log_ratio_min", "log_ratio_max")
            else value / max(accepted, 1)
            for key, value in totals.items()
        }
        metrics = {
            "clip_param": self.clip_param,
            # This k3 diagnostic uses a genuine joint likelihood ratio. It is
            # separate from the analytically integrated map KL below.
            "approx_kl": means.get("sample_joint_kl", 0.0),
            "clip_fraction": means.get("clip_fraction", 0.0),
            "explained_variance": explained,
            "action_std": action_std,
            "joint/action_noise_std": noise_std.mean(),
            "joint/action_noise_std_min": noise_std.min(),
            "joint/action_noise_std_max": noise_std.max(),
            "joint/action_noise_variance": noise_std.square().mean(),
            "joint/learn_action_noise_std": float(self.learn_action_std),
            "joint/conditional_entropy": self._conditional_entropy().detach(),
            "joint/action_noise_schedule_progress": self.action_noise_schedule_progress,
            "joint/action_noise_completed_env_steps": (
                self.action_noise_completed_env_steps
            ),
            "joint/budget_enabled": float(self.enable_budget),
            "joint/kl_coefficient_used": coefficient_used,
            "joint/kl_coefficient_next": self.kl_coefficient,
            "joint/kl_target": self.kl_target,
            "joint/accepted_updates": accepted,
            "joint/attempted_batches": attempted,
            "joint/rejected_candidates": rejected,
            "joint/early_stop": float(
                accepted < self.num_learning_epochs * self.num_mini_batches
            ),
            "joint/last_step_learning_rate": last_step_lr,
            "joint/ratio_mean": means.get("ratio_mean", 1.0),
            "joint/log_ratio_min": means.get("log_ratio_min", 0.0),
            "joint/log_ratio_max": means.get("log_ratio_max", 0.0),
            "joint/map_kl_train": means.get("map_kl", 0.0),
            "joint/budget_uses_gaussian_kl": float(self.gaussian_residual_policy),
            **aux_weight_metrics,
            "mean_grad_norm_before_clip": means.get("mean_grad_norm_before_clip", 0.0),
            "mean_grad_norm_after_clip": means.get("mean_grad_norm_after_clip", 0.0),
        }
        if self.enable_budget:
            for name, samples in (("probe", probe_kl), ("audit", audit_kl)):
                mean = samples.mean()
                # Preserve legacy names, and expose the actual metric used.
                kind = (
                    "conditional_joint_kl"
                    if self.gaussian_residual_policy
                    else "legacy_map_l1"
                )
                for label, value in (
                    ("mean", mean),
                    ("p95", torch.quantile(samples.flatten(), 0.95)),
                    ("max", samples.max()),
                ):
                    metrics[f"joint/{name}_kl_{label}"] = value
                    metrics[f"joint/{name}_{kind}_{label}"] = value
                if not self.gaussian_residual_policy:
                    metrics[f"joint/{name}_map_distance"] = mean
        if self.gaussian_residual_policy:
            metrics["joint/conditional_joint_kl_train"] = means.get("map_kl", 0.0)
        else:
            metrics["joint/legacy_map_l1_train"] = means.get("map_kl", 0.0)
        for name in ("aux_u_loss", "aux_v_loss", "aux_jvp_norm"):
            if name in means:
                metrics[f"joint/{name}"] = means[name]
        metrics.update(self._extra_update_metrics(means))
        self.storage.clear()
        self.update_counter += 1
        self.tot_timesteps += 1
        loss_dict = {
            "surrogate_loss": means.get("surrogate_loss", 0.0),
            "value_loss": means.get("value_loss", 0.0),
            "map_kl_loss": means.get("map_kl_loss", 0.0),
            "pmf_aux_loss": means.get("pmf_aux_loss", 0.0),
            "conditional_entropy_loss": means.get("conditional_entropy_loss", 0.0),
            "total_loss": means.get("total_loss", 0.0),
        }
        self._metrics_to_cpu(loss_dict, metrics)
        loss_dict["metrics"] = metrics
        return loss_dict

    def _clipped_surrogate(
        self, log_ratio: torch.Tensor, advantages: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate the exact PPO clipped objective without inactive exponentials.

        Positive advantages cap the ratio from above; negative advantages cap
        it only from below. Clipping in log space prevents the unused positive
        branch from overflowing and producing ``0 * inf`` in backward. The
        negative-advantage branch deliberately retains its unbounded penalty.
        A zero advantage contributes zero regardless of the likelihood ratio.
        """
        clipped_log_ratio = torch.where(
            advantages > 0,
            log_ratio.clamp_max(math.log1p(self.clip_param)),
            torch.where(
                advantages < 0,
                log_ratio.clamp_min(math.log1p(-self.clip_param)),
                torch.zeros_like(log_ratio),
            ),
        )
        return -(clipped_log_ratio.exp() * advantages).mean()

    @torch.no_grad()
    def _clip_gradients(
        self,
        batch: dict[str, torch.Tensor],
        statistics: dict[str, torch.Tensor],
        *,
        batch_index: int = 0,
    ) -> torch.Tensor:
        """Clip finite gradients, recovering an overflowed FP32 norm in FP64.

        Measure the norm before any in-place scaling: the standard clipping
        helper would otherwise turn finite gradients into zeros when its norm
        overflows. Real non-finite gradient entries remain a hard error and
        are reported with the offending parameters and minibatch ranges.
        """
        parameters = [p for p in self.policy.parameters() if p.grad is not None]
        gradients = [p.grad for p in parameters]
        norm = torch.nn.utils.get_total_norm(gradients)
        if not torch.isfinite(norm):
            bad_parameters = [
                name
                for name, parameter in self.policy.named_parameters()
                if parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
            ]
            if bad_parameters:
                raise FloatingPointError(
                    "Non-finite FSPPOJoint gradient before optimizer step; "
                    f"iteration={self.update_counter}, minibatch={batch_index}; "
                    f"parameters={', '.join(bad_parameters)}; "
                    + self._gradient_failure_ranges(batch, statistics)
                )
            norm = torch.nn.utils.get_total_norm(
                [gradient.to(dtype=torch.float64) for gradient in gradients]
            )
            if not torch.isfinite(norm):
                raise FloatingPointError(
                    "Non-finite FSPPOJoint FP64 gradient norm; "
                    f"iteration={self.update_counter}, minibatch={batch_index}; "
                    + self._gradient_failure_ranges(batch, statistics)
                )
        torch.nn.utils.clip_grads_with_norm_(parameters, self.max_grad_norm, norm)
        return norm

    @staticmethod
    def _gradient_failure_ranges(
        batch: dict[str, torch.Tensor], statistics: dict[str, torch.Tensor]
    ) -> str:
        """Summarize an exceptional minibatch without printing entire tensors."""
        ranges = []
        for name in ("obs", "actions", "returns"):
            tensor = batch[name].detach()
            finite = torch.isfinite(tensor)
            finite_values = tensor[finite]
            bounds = (
                f"[{float(finite_values.min()):.6g}, {float(finite_values.max()):.6g}]"
                if finite_values.numel()
                else "[no finite values]"
            )
            ranges.append(f"{name}={bounds}, nonfinite={int((~finite).sum())}")
        for name in ("log_ratio_min", "log_ratio_max"):
            if name in statistics:
                ranges.append(f"{name}={float(statistics[name].detach()):.6g}")
        return "; ".join(ranges)

    def _extra_update_metrics(self, means: dict[str, torch.Tensor]) -> dict:
        """Return variant-specific diagnostics before conversion to CPU scalars."""
        return {}

    def _current_sampler_contract(self) -> dict:
        contract = {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "num_actor_obs": self.policy.num_actor_obs,
            "num_actions": self.policy.num_actions,
            "timestep_embed_dim": self.policy.timestep_embed_dim,
            "action_perturb_std": self.initial_sigma,
            "actor_scale": float(self.policy.actor_scale),
            "actor_mlp_output_scale": float(self.policy.mlp_output_scale),
            "sampling_steps": self.policy.sampling_steps,
            "pmf_time_eps": float(self.policy.pmf_time_eps),
        }
        if self.sigma_decay_env_steps > 0:
            contract["action_perturb_std_final"] = self.final_sigma
            contract["action_perturb_std_decay_env_steps"] = self.sigma_decay_env_steps
        if isinstance(self.policy, GaussianResidualPMFActorCritic):
            contract.update(
                policy_class=type(self.policy).__name__,
                learn_action_perturb_std=self.learn_action_std,
                action_perturb_std_min=float(self.policy.action_perturb_std_min),
                action_perturb_std_max=float(self.policy.action_perturb_std_max),
                gaussian_residual_scale=float(self.policy.gaussian_residual_scale),
                actor_residual_scale=float(self.policy.actor.residual_scale),
                actor_public_output_scale=float(self.policy.actor.public_output_scale),
                budget_metric="conditional_gaussian_kl",
            )
        elif isinstance(self.policy, LearnedStdPMFActorCritic):
            contract.update(
                policy_class=type(self.policy).__name__,
                learn_action_perturb_std=self.learn_action_std,
                action_perturb_std_min=float(self.policy.action_perturb_std_min),
                action_perturb_std_max=float(self.policy.action_perturb_std_max),
                budget_metric="conditional_gaussian_kl",
            )
        return contract

    def _check_sampler_contract(self) -> None:
        if self._current_sampler_contract() != self._sampler_contract:
            raise ValueError(
                "FSPPOJoint sampler dimensions and noise schedule must remain fixed"
            )
        if (
            not self.learn_action_std
            and float(self.policy.action_perturb_std) != self.sigma
        ):
            raise ValueError(
                "FSPPOJoint action noise may only change at a rollout boundary"
            )

    def _action_std(self, actor: torch.nn.Module | None = None) -> torch.Tensor:
        """Read the current or frozen conditional scale in action coordinates."""
        if self.gaussian_residual_policy:
            return self.policy.conditional_action_std(actor=actor)
        return next(self.policy.parameters()).new_full(
            (self.policy.num_actions,), self.sigma
        )

    def _conditional_entropy(self) -> torch.Tensor:
        """Gaussian conditional entropy; this excludes latent-action mutual information."""
        return (self._action_std().log() + 0.5 * math.log(2 * math.pi * math.e)).sum()

    @torch.no_grad()
    def _prepare_aux_weights(self) -> dict[str, torch.Tensor]:
        """Freeze capped positive distillation weights for the entire PPO update.

        Standardization and normalization use the complete rollout. These
        weights are reused unchanged by all epochs and mini-batches. Capping
        after mean normalization can reduce their final mean below one.
        """
        weights = self.storage.aux_weights
        weights.fill_(1.0)
        if self.aux_loss_coef > 0 and self.aux_weighting == "advantage_exp":
            advantages = self.storage.advantages.detach()
            standardized = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )
            log_weights = standardized / self.aux_temperature
            if not torch.isfinite(log_weights).all():
                raise FloatingPointError("Non-finite FSPPOJoint auxiliary log weights")
            log_weights = (log_weights - log_weights.max()).clamp_min(
                math.log(torch.finfo(log_weights.dtype).tiny)
            )
            positive_weights = log_weights.exp()
            positive_weights = positive_weights / positive_weights.mean()
            weights.copy_(positive_weights.clamp_max(self.aux_weight_clip))
        flat = weights.flatten()
        # Fraction of an equivalent uniformly weighted rollout, in (0, 1].
        ess_fraction = flat.sum().square() / (flat.square().sum() * flat.numel())
        return {
            "joint/aux_weight_mean": flat.mean(),
            "joint/aux_weight_min": flat.min(),
            "joint/aux_weight_max": flat.max(),
            "joint/aux_weight_ess_fraction": ess_fraction,
            "joint/aux_reward_weighting": flat.new_tensor(
                float(self.aux_weighting == "advantage_exp")
            ),
            "joint/aux_target_is_mean": flat.new_tensor(
                float(self.aux_target == "means")
            ),
        }

    @torch.no_grad()
    def _validate_replay(self) -> None:
        # Rollout sampling evaluates the actor once per environment step with
        # batch size num_envs. Replay at that same shape: splitting the flattened
        # rollout by probe_size can change GEMM kernels (for example, 256 rows
        # during sampling versus 1024 here) and introduce enough FP32 rounding
        # noise to fail the strict mean check despite unchanged policy weights.
        for step in range(self.storage.num_transitions_per_env):
            obs = self.storage.observations[step]
            latents = self.storage.action_latents[step]
            means = self.storage.action_means[step]
            actions = self.storage.actions[step]
            log_probs = self.storage.action_log_probs[step]
            action_stds = self.storage.action_stds[step]
            replay = self.policy.transport_actions(obs, latents)
            if self.gaussian_residual_policy:
                if not torch.allclose(
                    self._action_std().expand_as(action_stds),
                    action_stds,
                    rtol=1e-5,
                    atol=1e-7,
                ):
                    raise RuntimeError(
                        "FSPPOJoint behavior replay mismatch: action scale changed"
                    )
                log_prob = self.policy.conditional_action_log_prob(
                    actions, replay, std=action_stds
                )
            else:
                log_prob = self.policy.conditional_action_log_prob(
                    actions, replay
                )
            if not torch.allclose(
                replay, means, rtol=1e-4, atol=1e-6
            ) or not torch.allclose(log_prob, log_probs, rtol=1e-4, atol=1e-4):
                mean_error = float((replay - means).abs().max())
                log_prob_error = float((log_prob - log_probs).abs().max())
                raise RuntimeError(
                    "FSPPOJoint behavior replay mismatch: actor, latent, or raw action changed; "
                    f"max mean error={mean_error:.3g}, max log-prob error={log_prob_error:.3g}"
                )

    @torch.no_grad()
    def _make_probe(self) -> dict[str, torch.Tensor]:
        if self.enable_budget:
            observations = self.storage.observations.flatten(0, 1)
            indices = torch.randperm(observations.shape[0], device=self.device)[
                : self.probe_size
            ]
            obs = observations[indices]
            noise = torch.randn(
                obs.shape[0],
                self.map_samples,
                self.policy.num_actions,
                device=self.device,
            )
            old_actions = self.policy.transport_actions(
                obs, noise, actor=self._old_actor
            )
            return {
                "obs": obs,
                "noise": noise,
                "old_actions": old_actions,
                "old_std": self._action_std(actor=self._old_actor).clone(),
            }
        else:
            return {
                "obs": torch.empty(0),
                "noise": torch.empty(0),
                "old_actions": torch.empty(0),
            }

    @torch.no_grad()
    def _probe_l1(self, probe: dict[str, torch.Tensor]) -> torch.Tensor:
        """Legacy L1 same-latent map cost; it is not a Gaussian KL."""
        actions: torch.Tensor = self.policy.transport_actions(
            probe["obs"], probe["noise"]
        )
        return (actions - probe["old_actions"]).absolute().sum(dim=-1)

    def _gaussian_kl(
        self,
        current_mean: torch.Tensor,
        old_mean: torch.Tensor,
        old_std: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``KL(old || current)`` including learned scale changes."""
        current_std = self._action_std()
        log_variance_ratio = 2 * (old_std / current_std).log()
        # expm1 avoids losing a small variance-only KL by subtracting 0.5
        # from another number near 0.5 in single precision.
        variance_kl = 0.5 * (torch.expm1(log_variance_ratio) - log_variance_ratio)
        mean_kl = 0.5 * ((old_mean - current_mean) / current_std).square()
        return (variance_kl + mean_kl).sum(dim=-1)

    @torch.no_grad()
    def _probe_distance(self, probe: dict[str, torch.Tensor]) -> torch.Tensor:
        """Evaluate the selected policy's budget metric on a fixed probe."""
        if not self.gaussian_residual_policy:
            return self._probe_l1(probe)
        current_mean = self.policy.transport_actions(probe["obs"], probe["noise"])
        return self._gaussian_kl(current_mean, probe["old_actions"], probe["old_std"])

    @torch.no_grad()
    def _map_kl_diagnostic(self, obs: torch.Tensor) -> torch.Tensor:
        """Estimate the plain-PMF map distance for logging only.

        Without the budget this value never enters the loss or a schedule.
        With ``fsppo_joint_cheap_map_diagnostics`` it uses a random rollout
        subset with ``diagnostic_map_samples`` same-noise samples per row;
        otherwise it keeps the exact full-batch ``map_samples`` width.
        """
        if self.cheap_map_diagnostics:
            if obs.shape[0] > self.diagnostic_obs_samples:
                indices = torch.randperm(obs.shape[0], device=obs.device)[
                    : self.diagnostic_obs_samples
                ]
                obs = obs[indices]
            map_samples = self.diagnostic_map_samples
        else:
            map_samples = self.map_samples
        noise = torch.randn(
            obs.shape[0],
            map_samples,
            self.policy.num_actions,
            device=self.device,
        )
        current_map = self.policy.transport_actions(obs, noise)
        old_map = self.policy.transport_actions(obs, noise, actor=self._old_actor)
        return (current_map - old_map).absolute().sum(dim=-1).mean()

    def _batch_loss(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        obs, actions = batch["obs"], batch["actions"]
        advantages = batch["advantages"]
        if self.normalize_advantage_per_mini_batch:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )
        positive, negative = self.advantage_clamp
        advantages = advantages.clamp(-negative, positive)
        mean = self.policy.transport_actions(obs, batch["action_latents"])
        log_prob = self.policy.conditional_action_log_prob(actions, mean)
        log_ratio = log_prob - batch["action_log_probs"]
        surrogate = self._clipped_surrogate(log_ratio, advantages)
        # Preserve the actual joint ratio for diagnostics; it is not an
        # autograd branch of the clipped surrogate.
        detached_log_ratio = log_ratio.detach()
        ratio = detached_log_ratio.exp()
        value = self.policy.evaluate(batch["critic_obs"])
        value_error = (value - batch["returns"]).square()
        if self.use_clipped_value_loss:
            clipped = batch["values"] + (value - batch["values"]).clamp(
                -self.clip_param, self.clip_param
            )
            value_error = torch.maximum(
                value_error, (clipped - batch["returns"]).square()
            )
        value_loss = value_error.mean()
        if self.gaussian_residual_policy and not self.enable_budget:
            # Saved behavior latents already give an empirical joint KL. Avoid
            # additional transport evaluations when this is diagnostic only.
            with torch.no_grad():
                map_kl = self._gaussian_kl(
                    mean, batch["action_means"], batch["action_stds"]
                ).mean()
        elif self.enable_budget:
            # The budgeted penalty is part of the loss: it differentiates
            # through the current map at the full map-sample width.
            noise = torch.randn(
                obs.shape[0],
                self.map_samples,
                self.policy.num_actions,
                device=self.device,
            )
            current_map = self.policy.transport_actions(obs, noise)
            with torch.no_grad():
                old_map = self.policy.transport_actions(
                    obs, noise, actor=self._old_actor
                )
            if self.gaussian_residual_policy:
                map_kl = self._gaussian_kl(
                    current_map, old_map, self._action_std(actor=self._old_actor)
                ).mean()
            else:
                map_kl = (current_map - old_map).absolute().sum(dim=-1).mean()
        else:
            # Without a budget the plain-PMF map_kl is log-only: never build
            # its autograd graph and optionally skip its full-width sampling.
            map_kl = self._map_kl_diagnostic(obs)
        aux_target = batch["action_means"] if self.aux_target == "means" else actions
        aux_loss, aux_metrics = self._auxiliary_loss(
            obs, aux_target.detach(), batch.get("aux_weights")
        )
        penalty = (
            self.kl_coefficient * map_kl if self.enable_budget else map_kl.new_zeros(())
        )
        # pMF regression uses non-negative weights; signed advantages would
        # reward increasing the residual on negative-advantage samples.
        weighted_aux = self.aux_loss_coef * aux_loss.mean()
        # This is H(a | obs, latent), not the marginal mixture/action entropy.
        entropy_loss = -self.entropy_coef * self._conditional_entropy()
        loss = (
            surrogate
            + self.value_loss_coef * value_loss
            + penalty
            + weighted_aux
            + entropy_loss
        )
        statistics = {
            "surrogate_loss": surrogate,
            "value_loss": value_loss,
            "map_kl_loss": penalty,
            "pmf_aux_loss": weighted_aux,
            "conditional_entropy_loss": entropy_loss,
            "total_loss": loss,
            "sample_joint_kl": (
                torch.expm1(detached_log_ratio) - detached_log_ratio
            ).mean(),
            "log_ratio_min": detached_log_ratio.min(),
            "log_ratio_max": detached_log_ratio.max(),
            "clip_fraction": ((ratio - 1).abs() > self.clip_param).float().mean(),
            "ratio_mean": ratio.mean(),
            "map_kl": map_kl,
            **aux_metrics,
        }
        return loss, statistics

    def _auxiliary_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Regress pMF on frozen behavior targets with non-negative weights.

        Mean targets distill the transport before conditional action noise is
        added. Advantage weights make this reward-aware transport distillation;
        they do not implement an exact state-wise marginal Gibbs projection.
        """
        if self.aux_loss_coef == 0:
            return obs.new_zeros(()), {}
        shape = (obs.shape[0], self.n_samples_per_action, 1)
        times = torch.sigmoid(
            torch.randn(2, *shape, device=self.device)
            * self.policy.pmf_logit_normal_std
            + self.policy.pmf_logit_normal_mean
        )
        # Uniformly assign an exact global anchor count; do not tie anchors to
        # a permanent prefix of environment IDs.
        count = shape[0] * shape[1]
        anchor_count = int(count * self.policy.pmf_fm_proportion)
        anchors = torch.randperm(count, device=self.device)[:anchor_count]
        times[1].view(-1)[anchors] = times[0].view(-1)[anchors]
        t, r = times.max(dim=0).values, times.min(dim=0).values
        eps = torch.randn(*shape[:2], self.policy.num_actions, device=self.device)
        score, _, _, components = self.policy.get_pmf_loss(
            obs, actions, eps, r, t, return_components=True
        )
        if weights is None:
            weights = score.new_ones((obs.shape[0], 1))
        weights = weights.detach()
        return score * weights, {
            "aux_u_loss": (components["u_loss"] * weights).mean(),
            "aux_v_loss": (components["v_loss"] * weights).mean(),
            "aux_jvp_norm": components["jvp_norm"].mean(),
        }

    def _step_with_budget(
        self, probe: dict[str, torch.Tensor]
    ) -> tuple[bool, int, float]:
        """Try one gradient at smaller step sizes, rolling back rejected trials."""
        if not self.enable_budget:
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            self.optimizer.step()
            return True, 0, learning_rate

        model_state = copy.deepcopy(self.policy.state_dict())
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        base_lrs = [group["lr"] for group in self.optimizer.param_groups]
        for attempt in range(self.max_backtracks + 1):
            if attempt:
                self.policy.load_state_dict(model_state)
                self.optimizer.load_state_dict(copy.deepcopy(optimizer_state))
            factor = self.backtrack_factor**attempt
            for group, lr in zip(self.optimizer.param_groups, base_lrs):
                group["lr"] = lr * factor
            self.optimizer.step()
            kl = self._probe_distance(probe)
            finite_parameters = bool(
                torch.stack(
                    [torch.isfinite(p).all() for p in self.policy.parameters()]
                ).all()
            )
            if (
                finite_parameters
                and bool(torch.isfinite(kl).all())
                and float(kl.mean()) <= self.kl_target
            ):
                for group, lr in zip(self.optimizer.param_groups, base_lrs):
                    group["lr"] = lr
                return True, attempt, base_lrs[0] * factor
        self.policy.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)
        return False, self.max_backtracks + 1, 0.0
