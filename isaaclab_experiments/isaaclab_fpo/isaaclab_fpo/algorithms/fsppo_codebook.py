"""Controlled joint/marginal PPO comparison on one finite mixture policy."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab_fpo.modules import CodebookPMFActorCritic
from isaaclab_fpo.storage.codebook_rollout_storage import CodebookRolloutStorage

from .codebook_diagnostics import codebook_diagnostics
from .fsppo_joint import FSPPOJoint

if TYPE_CHECKING:
    from isaaclab_fpo.rl_cfg import FpoRslRlPpoAlgorithmCfg


class FSPPOCodebook(FSPPOJoint):
    """Change only the PPO ratio between a fixed-codebook joint and marginal.

    Both modes use the same uniform discrete prior and conditional Gaussian
    sampler. Full old component log densities are computed once before any
    optimizer step; the generating component's conditional density retains
    its existing, separate storage field. Budget, auxiliary regression and
    EMA are disabled to isolate the ratio in this experiment.
    """

    def __init__(
        self,
        policy: CodebookPMFActorCritic,
        cfg: FpoRslRlPpoAlgorithmCfg,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
    ):
        if not isinstance(policy, CodebookPMFActorCritic):
            raise TypeError("FSPPOCodebook requires CodebookPMFActorCritic")
        if cfg.fsppo_codebook_ratio_mode not in ("joint", "marginal"):
            raise ValueError("fsppo_codebook_ratio_mode must be joint or marginal")
        if (
            cfg.fsppo_joint_enable_budget
            or cfg.fsppo_joint_aux_loss_coef != 0
            or cfg.ema_decay != 0
            or cfg.fsppo_joint_entropy_coef != 0
        ):
            raise ValueError("The codebook ratio comparison requires budget=False, aux=0, entropy=0 and ema_decay=0")
        self.ratio_mode = cfg.fsppo_codebook_ratio_mode
        super().__init__(policy, cfg, device=device, multi_gpu_cfg=multi_gpu_cfg)
        self.transition = CodebookRolloutStorage.Transition()

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape):
        """Allocate selected-component fields and an O(K) behavior-density cache."""
        self.storage = CodebookRolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            codebook_size=self.policy.codebook_size,
            device=self.device,
        )

    def begin_rollout(self, completed_env_steps: int) -> None:
        """Advance sigma only after the previous rollout has been consumed."""
        if self.storage is not None and self.storage.step != 0:
            raise RuntimeError("Cannot change sigma while a rollout is pending")
        self.policy.validate_codebook()
        super().begin_rollout(completed_env_steps)
        self.policy.codebook_noise_std.fill_(self.sigma)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        """Retain raw actions and the actual selected discrete component."""
        self._check_sampler_contract()
        actions, index, latent, mean, log_prob = self.policy.sample_codebook_transport(obs)
        self.transition.actions = actions.clone()
        self.transition.component_index = index
        self.transition.action_latent = latent
        self.transition.action_mean = mean
        self.transition.action_log_prob = log_prob
        self.transition.action_std = self._action_std().expand_as(mean).clone()
        self.transition.values = self.policy.evaluate(critic_obs)
        self.transition.observations = obs.clone()
        self.transition.privileged_observations = critic_obs.clone()
        return actions

    def update(self, obs_normalizer=None, privileged_obs_normalizer=None) -> dict:
        """Freeze all behavior densities once, then run the shared PPO update."""
        self._check_sampler_contract()
        self.policy.validate_codebook()
        if self.storage.step != self.storage.num_transitions_per_env:
            raise RuntimeError("FSPPOCodebook.update requires a complete rollout")
        indices = self.storage.component_indices.flatten()
        if ((indices < 0) | (indices >= self.policy.codebook_size)).any():
            raise RuntimeError("Stored codebook component index is outside the fixed prior")
        if not torch.equal(self.storage.action_latents.flatten(0, 1), self.policy.latent_codebook[indices]):
            raise RuntimeError("Stored codebook component indices do not match the behavior latents")
        self._validate_replay()
        with torch.no_grad():
            observations = self.storage.observations.flatten(0, 1)
            actions = self.storage.actions.flatten(0, 1)
            cached = self.storage.old_component_log_probs.flatten(0, 1)
            for start in range(0, observations.shape[0], self.probe_size):
                part = slice(start, start + self.probe_size)
                means = self.policy.component_means(observations[part])
                cached[part].copy_(self.policy.component_action_log_probs(actions[part], means))
            if not torch.isfinite(cached).all():
                raise FloatingPointError(
                    "Non-finite cached codebook densities before optimizer update; "
                    f"iteration={self.update_counter}, sigma={self.sigma:.6g}, "
                    f"nonfinite_components={int((~torch.isfinite(cached)).sum())}"
                )
            selected = cached.gather(1, indices.unsqueeze(-1))
            behavior_log_probs = self.storage.action_log_probs.flatten(0, 1)
            dense_log_prob_error = float((selected - behavior_log_probs).abs().max())
            mismatch = ~torch.isclose(selected, behavior_log_probs, rtol=1e-4, atol=1e-4)
            fallback_steps = mismatch.reshape(self.storage.num_transitions_per_env, self.storage.num_envs).any(dim=1)
            fallback_steps = fallback_steps.nonzero().flatten().tolist()
            # All-K GEMMs can round differently from single-code collection.
            # Re-evaluate affected steps using the actual rollout batch size
            # and single-code layout; retain the original density tolerances.
            for step in fallback_steps:
                obs = self.storage.observations[step]
                raw_actions = self.storage.actions[step]
                for component, code in enumerate(self.policy.latent_codebook):
                    latents = code.expand(self.storage.num_envs, -1).contiguous()
                    mean = self.policy.transport_actions(obs, latents)
                    self.storage.old_component_log_probs[step, :, component].copy_(
                        self.policy.conditional_action_log_prob(raw_actions, mean).squeeze(-1)
                    )
            selected = cached.gather(1, indices.unsqueeze(-1))
            if not torch.isfinite(cached).all() or not torch.allclose(
                selected, behavior_log_probs, rtol=1e-4, atol=1e-4
            ):
                raise RuntimeError(
                    "Cached codebook densities do not match the behavior sampler after rollout-layout replay; "
                    f"iteration={self.update_counter}, sigma={self.sigma:.6g}, "
                    f"max_log_prob_error={float((selected - behavior_log_probs).abs().max()):.6g}, "
                    f"nonfinite_components={int((~torch.isfinite(cached)).sum())}"
                )
            self.storage.component_log_probs_ready = True
        result = super().update(obs_normalizer, privileged_obs_normalizer)
        result["metrics"]["codebook/cache_replay_fallback_steps"] = len(fallback_steps)
        result["metrics"]["codebook/cache_selected_log_prob_error"] = dense_log_prob_error
        # This variant does not compute the legacy continuous-prior L1 map cost.
        result["metrics"].pop("joint/map_kl_train", None)
        result["metrics"].pop("joint/legacy_map_l1_train", None)
        result["metrics"].pop("joint/ratio_mean", None)
        return result

    def _batch_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        advantages = batch["advantages"]
        if self.normalize_advantage_per_mini_batch:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        positive, negative = self.advantage_clamp
        advantages = advantages.clamp(-negative, positive)
        means = self.policy.component_means(batch["obs"])
        current = self.policy.component_action_log_probs(batch["actions"], means)
        old = batch["old_component_log_probs"]
        selected = batch["component_indices"].unsqueeze(-1)
        joint_log_ratio = current.gather(1, selected) - batch["action_log_probs"]
        marginal_log_ratio = torch.logsumexp(current, dim=-1, keepdim=True) - torch.logsumexp(old, dim=-1, keepdim=True)
        log_ratio = joint_log_ratio if self.ratio_mode == "joint" else marginal_log_ratio
        surrogate = self._clipped_surrogate(log_ratio, advantages)
        ratio = log_ratio.detach().exp()
        value = self.policy.evaluate(batch["critic_obs"])
        value_error = (value - batch["returns"]).square()
        if self.use_clipped_value_loss:
            clipped = batch["values"] + (value - batch["values"]).clamp(-self.clip_param, self.clip_param)
            value_error = torch.maximum(value_error, (clipped - batch["returns"]).square())
        value_loss = value_error.mean()
        loss = surrogate + self.value_loss_coef * value_loss
        diagnostics = codebook_diagnostics(old, current.detach(), advantages.detach(), self.clip_param)
        diagnostics.update(
            {
                "codebook/joint_kl_k3": (torch.expm1(joint_log_ratio.detach()) - joint_log_ratio.detach()).mean(),
                "codebook/joint_clip_fraction": ((joint_log_ratio.detach().exp() - 1).abs() > self.clip_param)
                .float()
                .mean(),
                "codebook/marginal_clip_fraction": ((marginal_log_ratio.detach().exp() - 1).abs() > self.clip_param)
                .float()
                .mean(),
                "codebook/joint_ratio_mean": joint_log_ratio.detach().exp().mean(),
                "codebook/marginal_ratio_mean": marginal_log_ratio.detach().exp().mean(),
                "codebook/selected_ratio_mean": ratio.detach().mean(),
            }
        )
        statistics = {
            "surrogate_loss": surrogate,
            "value_loss": value_loss,
            "total_loss": loss,
            "sample_joint_kl": (torch.expm1(log_ratio.detach()) - log_ratio.detach()).mean(),
            "log_ratio_min": log_ratio.detach().min(),
            "log_ratio_max": log_ratio.detach().max(),
            "clip_fraction": ((ratio.detach() - 1).abs() > self.clip_param).float().mean(),
            "ratio_mean": ratio.detach().mean(),
            **diagnostics,
        }
        return loss, statistics

    def _extra_update_metrics(self, means: dict[str, torch.Tensor]) -> dict:
        metrics = {name: value for name, value in means.items() if name.startswith("codebook/")}
        metrics["codebook/size"] = self.policy.codebook_size
        metrics["codebook/marginal_ratio_enabled"] = float(self.ratio_mode == "marginal")
        # Enumerate the finite prior exactly on a small, deterministic state
        # probe. This is joint KL averaged over probe states, not a hard bound.
        with torch.no_grad():
            obs = self.storage.observations.flatten(0, 1)[: self.probe_size]
            old_means = self.policy.component_means(obs, actor=self._old_actor)
            current_means = self.policy.component_means(obs)
            metrics["codebook/joint_kl_analytic_probe"] = (
                (current_means - old_means).square().sum(dim=-1) / (2 * self.sigma**2)
            ).mean()
        return metrics

    def _current_sampler_contract(self) -> dict:
        contract = super()._current_sampler_contract()
        contract.update(
            {
                "latent_prior": "uniform_codebook",
                "codebook_size": self.policy.codebook_size,
                "codebook_seed": self.policy.codebook_seed,
                "codebook_sha256": self.policy.codebook_sha256,
                "ratio_mode": self.ratio_mode,
            }
        )
        return contract

    def state_dict(self) -> dict:
        """Persist the annealed scale as well as the immutable sampling contract."""
        state = super().state_dict()
        state["action_noise_completed_env_steps"] = self.action_noise_completed_env_steps
        state["action_noise_std"] = self.sigma
        return state

    def load_state_dict(self, state: dict) -> None:
        """Restore a matching objective and the checkpoint's rollout noise."""
        steps = state.get("action_noise_completed_env_steps")
        sigma = state.get("action_noise_std")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("Invalid codebook checkpoint noise-schedule step")
        if isinstance(sigma, bool) or not isinstance(sigma, (int, float)) or not math.isfinite(sigma):
            raise ValueError("Invalid codebook checkpoint noise standard deviation")
        progress = min(steps / self.sigma_decay_env_steps, 1.0) if self.sigma_decay_env_steps else 0.0
        expected = self.initial_sigma + progress * (self.final_sigma - self.initial_sigma)
        if sigma != expected or float(self.policy.codebook_noise_std) != sigma:
            raise ValueError("Codebook checkpoint noise scale does not match its schedule and policy")
        super().load_state_dict(state)
        self.sigma = float(sigma)
        self.policy.action_perturb_std = self.sigma
        self.action_noise_completed_env_steps = steps
        self.action_noise_schedule_progress = progress
