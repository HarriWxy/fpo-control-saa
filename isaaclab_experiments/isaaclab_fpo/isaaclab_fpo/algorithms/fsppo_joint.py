"""One-step latent Gaussian PPO with an empirical transport-map KL budget."""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

import torch

from isaaclab_fpo.algorithms.fpo import FPO
from isaaclab_fpo.modules import PMFActorCritic
from isaaclab_fpo.storage.joint_rollout_storage import JointRolloutStorage

if TYPE_CHECKING:
    from isaaclab_fpo.rl_cfg import FpoRslRlPpoAlgorithmCfg


class FSPPOJoint(FPO):
    """Optimize the joint policy ``p(xi) Normal(a; F_theta(obs, xi), sigma^2 I)``.

    Rollouts retain the actual generating latent and raw, perturbed action.
    The Gaussian conditional log-ratio is therefore exact for the joint policy;
    it is not the marginal action log-ratio. With sigma held positive and
    fixed for each rollout and its update, the same-latent map cost divided by
    ``2 * sigma**2`` is the joint KL and an
    upper bound on marginal action KL, in expectation over the prior.

    When the budget is enabled, every candidate optimizer step is checked on
    one fixed probe set for this rollout.  A rejected candidate restores BOTH
    model and optimizer state and retries with a smaller step size.  This
    enforces only the empirical probe mean budget, not a population or all-state
    bound.  A fresh probe at the end measures generalization of that budget.
    The budget can be disabled for a controlled no-budget ablation.  Optional
    pMF regression is a positive auxiliary loss and never participates in the
    policy ratio.
    """

    def __init__(
        self,
        policy: PMFActorCritic,
        cfg: FpoRslRlPpoAlgorithmCfg,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
    ):
        if not isinstance(policy, PMFActorCritic):
            raise TypeError("FSPPOJoint requires policy.class_name='PMFActorCritic'")
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
        self.kl_coefficient = (
            cfg.fsppo_joint_kl_coef if self.enable_budget else 0.0
        )
        self.dual_lr = cfg.fsppo_joint_dual_lr
        self.kl_coefficient_max = cfg.fsppo_joint_kl_coef_max
        self.map_samples = cfg.fsppo_joint_map_samples
        self.probe_size = cfg.fsppo_joint_probe_size
        self.max_backtracks = cfg.fsppo_joint_max_backtracks
        self.backtrack_factor = cfg.fsppo_joint_backtrack_factor
        self.aux_loss_coef = cfg.fsppo_joint_aux_loss_coef
        self._sampler_contract = self._current_sampler_contract()
        self._old_actor = copy.deepcopy(self.policy.actor).requires_grad_(False).eval()

    def begin_rollout(self, completed_env_steps: int) -> None:
        """Set the fixed conditional Gaussian scale for the next rollout.

        The schedule advances only at a rollout boundary. Each collected batch
        and its subsequent PPO update therefore use the same Gaussian scale.
        """
        if (
            isinstance(completed_env_steps, bool)
            or not isinstance(completed_env_steps, int)
            or completed_env_steps < 0
        ):
            raise ValueError("completed_env_steps must be a non-negative integer")
        self._check_sampler_contract()
        self.action_noise_completed_env_steps = completed_env_steps
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
        # a policy changed between collection and update.
        self._validate_replay()
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
            norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm
            )
            if not torch.isfinite(norm):
                raise FloatingPointError(
                    "Non-finite FSPPOJoint gradient before optimizer step"
                )
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
                totals[name] = totals.get(name, 0.0) + value.detach()

        with torch.no_grad():
            if self.enable_budget:
                probe_kl = self._probe_l1(probe)
                # New states/noises are drawn AFTER all updates; these diagnostics
                # are not reused for acceptance or described as a hard guarantee.
                audit_kl = self._probe_l1(self._make_probe())
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
        means = {key: value / max(accepted, 1) for key, value in totals.items()}
        metrics = {
            "clip_param": self.clip_param,
            # This k3 diagnostic uses a genuine joint likelihood ratio. It is
            # separate from the analytically integrated map KL below.
            "approx_kl": means.get("sample_joint_kl", 0.0),
            "clip_fraction": means.get("clip_fraction", 0.0),
            "explained_variance": explained,
            "action_std": action_std,
            "joint/action_noise_std": self.sigma,
            "joint/action_noise_variance": self.sigma**2,
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
            "joint/map_kl_train": means.get("map_kl", 0.0),
            "mean_grad_norm_before_clip": means.get("mean_grad_norm_before_clip", 0.0),
            "mean_grad_norm_after_clip": means.get("mean_grad_norm_after_clip", 0.0),
        }
        if self.enable_budget:
            for name, samples in (("probe", probe_kl), ("audit", audit_kl)):
                mean = samples.mean()
                metrics[f"joint/{name}_kl_mean"] = mean
                metrics[f"joint/{name}_kl_p95"] = torch.quantile(samples.flatten(), 0.95)
                metrics[f"joint/{name}_kl_max"] = samples.max()
                metrics[f"joint/{name}_map_distance"] = (
                    mean * 2 * self.sigma**2
                )
        for name in ("aux_u_loss", "aux_v_loss", "aux_jvp_norm"):
            if name in means:
                metrics[f"joint/{name}"] = means[name]
        self.storage.clear()
        self.update_counter += 1
        self.tot_timesteps += 1
        loss_dict = {
            "surrogate_loss": means.get("surrogate_loss", 0.0),
            "value_loss": means.get("value_loss", 0.0),
            "map_kl_loss": means.get("map_kl_loss", 0.0),
            "pmf_aux_loss": means.get("pmf_aux_loss", 0.0),
            "total_loss": means.get("total_loss", 0.0),
        }
        self._metrics_to_cpu(loss_dict, metrics)
        loss_dict["metrics"] = metrics
        return loss_dict

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
            contract["action_perturb_std_decay_env_steps"] = (
                self.sigma_decay_env_steps
            )
        return contract

    def _check_sampler_contract(self) -> None:
        if self._current_sampler_contract() != self._sampler_contract:
            raise ValueError(
                "FSPPOJoint sampler dimensions and noise schedule must remain fixed"
            )
        if float(self.policy.action_perturb_std) != self.sigma:
            raise ValueError(
                "FSPPOJoint action noise may only change at a rollout boundary"
            )

    @torch.no_grad()
    def _validate_replay(self) -> None:
        obs = self.storage.observations.flatten(0, 1)
        latents = self.storage.action_latents.flatten(0, 1)
        means = self.storage.action_means.flatten(0, 1)
        actions = self.storage.actions.flatten(0, 1)
        log_probs = self.storage.action_log_probs.flatten(0, 1)
        for start in range(0, obs.shape[0], self.probe_size):
            part = slice(start, start + self.probe_size)
            replay = self.policy.transport_actions(obs[part], latents[part])
            log_prob = self.policy.conditional_action_log_prob(actions[part], replay)
            if not torch.allclose(
                replay, means[part], rtol=1e-4, atol=1e-6
            ) or not torch.allclose(log_prob, log_probs[part], rtol=1e-4, atol=1e-4):
                mean_error = float((replay - means[part]).abs().max())
                log_prob_error = float((log_prob - log_probs[part]).abs().max())
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
                obs.shape[0], self.map_samples, self.policy.num_actions, device=self.device
            )
            old_actions = self.policy.transport_actions(obs, noise, actor=self._old_actor)
            return {"obs": obs, "noise": noise, "old_actions": old_actions}
        else:
            return {"obs": torch.empty(0), "noise": torch.empty(0), "old_actions": torch.empty(0)}

    @torch.no_grad()
    def _probe_l1(self, probe: dict[str, torch.Tensor]) -> torch.Tensor:
        """Wasserstein distance between current and old transport actions."""
        actions: torch.Tensor = self.policy.transport_actions(probe["obs"], probe["noise"])
        return ((actions - probe["old_actions"])).absolute().sum(dim=-1)

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
        # No CFM-score or straight-through log-ratio clamp: this is the actual
        # joint ratio. Non-finite updates fail before mutating the optimizer.
        ratio = log_ratio.exp()
        surrogate = -torch.minimum(
            ratio * advantages,
            ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * advantages,
        ).mean()
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
        noise = torch.randn(
            obs.shape[0], self.map_samples, self.policy.num_actions, device=self.device
        )
        current_map = self.policy.transport_actions(obs, noise)
        with torch.no_grad():
            old_map = self.policy.transport_actions(obs, noise, actor=self._old_actor)
        map_kl = (current_map - old_map).absolute().sum(dim=-1).mean()
        aux_loss, aux_metrics = self._auxiliary_loss(obs, actions)
        penalty = (
            self.kl_coefficient * map_kl
            if self.enable_budget
            else map_kl.new_zeros(())
        )
        weighted_aux = self.aux_loss_coef * aux_loss
        loss = surrogate + self.value_loss_coef * value_loss + penalty + weighted_aux
        statistics = {
            "surrogate_loss": surrogate,
            "value_loss": value_loss,
            "map_kl_loss": penalty,
            "pmf_aux_loss": weighted_aux,
            "total_loss": loss,
            "sample_joint_kl": (torch.expm1(log_ratio) - log_ratio).mean(),
            "clip_fraction": ((ratio - 1).abs() > self.clip_param).float().mean(),
            "ratio_mean": ratio.mean(),
            "map_kl": map_kl,
            **aux_metrics,
        }
        return loss, statistics

    def _auxiliary_loss(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
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
        return score.mean(), {
            "aux_u_loss": components["u_loss"].mean(),
            "aux_v_loss": components["v_loss"].mean(),
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
            kl = self._probe_l1(probe)
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
