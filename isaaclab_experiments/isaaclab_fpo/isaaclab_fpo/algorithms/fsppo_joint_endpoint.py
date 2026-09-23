"""Endpoint-action MAE surrogate for one-step pMF policy optimization."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab_fpo.algorithms.fpo import FPO
from isaaclab_fpo.modules import PMFActorCritic
from isaaclab_fpo.storage.joint_rollout_storage import JointRolloutStorage

if TYPE_CHECKING:
    from isaaclab_fpo.rl_cfg import FpoRslRlPpoAlgorithmCfg


class FSPPOJointEndpoint(FPO):
    """Train a one-step pMF actor with a terminal-action MAE score.

    The rollout still samples the same joint latent/action pair as
    :class:`FSPPOJoint`, but the update uses only the terminal transport
    output ``F_theta(obs, xi)``:

    ``mae_theta = mean(abs(F_theta(obs, xi) - action))``

    The importance-ratio surrogate is the engineering endpoint-score ratio
    ``exp(mae_old - mae_new)``.  It is intentionally not described as an
    exact action likelihood ratio.  The direct MAE term uses the same terminal
    output and the sampled rollout action as its target.  No intermediate pMF
    time samples, JVP loss, transport-map samples, or KL budget are used.
    """

    _REPLAY_CHUNK_SIZE = 4096

    def __init__(
        self,
        policy: PMFActorCritic,
        cfg: FpoRslRlPpoAlgorithmCfg,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
    ):
        if not isinstance(policy, PMFActorCritic):
            raise TypeError(
                "FSPPOJointEndpoint requires policy.class_name='PMFActorCritic'"
            )
        if multi_gpu_cfg is not None:
            raise ValueError(
                "FSPPOJointEndpoint currently supports a single learner device only"
            )
        if policy.sampling_steps != 1 or not 0 < policy.pmf_time_eps <= 1:
            raise ValueError(
                "FSPPOJointEndpoint requires sampling_steps=1 and "
                "0 < pmf_time_eps <= 1"
            )
        for name, value in {
            "action_perturb_std": policy.action_perturb_std,
            "actor_scale": policy.actor_scale,
            "learning_rate": cfg.learning_rate,
            "max_grad_norm": cfg.max_grad_norm,
        }.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(policy.mlp_output_scale):
            raise ValueError("actor_mlp_output_scale must be finite")
        if cfg.schedule != "fixed" or cfg.trust_region_mode != "ppo":
            raise ValueError(
                "FSPPOJointEndpoint requires schedule='fixed' and "
                "trust_region_mode='ppo'"
            )
        if cfg.knn_entropy_coef != 0 or cfg.storage_action_noise_std != 0:
            raise ValueError(
                "FSPPOJointEndpoint requires knn_entropy_coef=0 and "
                "storage_action_noise_std=0"
            )
        if cfg.fsppo_joint_enable_budget:
            raise ValueError(
                "FSPPOJointEndpoint has no map budget; set "
                "fsppo_joint_enable_budget=false"
            )
        if not math.isfinite(cfg.clip_param) or not 0 < cfg.clip_param < 1:
            raise ValueError("clip_param must be in (0, 1)")
        if (
            not math.isfinite(cfg.cfm_diff_clamp_max)
            or cfg.cfm_diff_clamp_max <= 0
        ):
            raise ValueError("cfm_diff_clamp_max must be finite and positive")
        if (
            not math.isfinite(cfg.fsppo_joint_endpoint_mae_coef)
            or cfg.fsppo_joint_endpoint_mae_coef < 0
        ):
            raise ValueError(
                "fsppo_joint_endpoint_mae_coef must be finite and non-negative"
            )

        # Keep the endpoint comparison numerically comparable to FSPPOJoint.
        torch.set_float32_matmul_precision("highest")
        super().__init__(policy, cfg, device=device)
        self.transition = JointRolloutStorage.Transition()
        self.sigma = float(policy.action_perturb_std)
        self.endpoint_mae_coef = cfg.fsppo_joint_endpoint_mae_coef
        self._sampler_contract = self._current_sampler_contract()

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
    ):
        """Allocate storage for replayable latents and terminal actions."""
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
        """Sample one rollout action and retain its terminal flow output."""
        self._check_sampler_contract()
        actions, latent, endpoint, log_prob = self.policy.sample_transport(obs)
        # Keep the raw Gaussian sample: it is the target for the endpoint MAE
        # and is also the action that generated the environment transition.
        self.transition.actions = actions.clone()
        self.transition.action_latent = latent
        self.transition.action_mean = endpoint
        self.transition.action_log_prob = log_prob
        self.transition.values = self.policy.evaluate(critic_obs)
        self.transition.observations = obs.clone()
        self.transition.privileged_observations = critic_obs.clone()
        return actions

    def state_dict(self) -> dict:
        """Save the endpoint objective and fixed rollout sampling contract."""
        return {
            "version": 1,
            "objective": "terminal_endpoint_mae",
            "update_counter": self.update_counter,
            "tot_timesteps": self.tot_timesteps,
            "sampler": dict(self._sampler_contract),
        }

    def load_state_dict(self, state: dict) -> None:
        """Reject checkpoints with a different endpoint objective or sampler."""
        if (
            state.get("version") != 1
            or state.get("objective") != "terminal_endpoint_mae"
            or state.get("sampler") != self._sampler_contract
        ):
            raise ValueError(
                "FSPPOJointEndpoint checkpoint objective or sampling contract mismatch"
            )
        for name in ("update_counter", "tot_timesteps"):
            value = state.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"Invalid FSPPOJointEndpoint checkpoint {name}"
                )
        self.update_counter = state["update_counter"]
        self.tot_timesteps = state["tot_timesteps"]

    def update(self, obs_normalizer=None, privileged_obs_normalizer=None) -> dict:
        """Run PPO updates using only terminal endpoint MAE quantities."""
        del obs_normalizer, privileged_obs_normalizer
        self._check_sampler_contract()
        if self.storage.step != self.storage.num_transitions_per_env:
            raise RuntimeError(
                "FSPPOJointEndpoint.update requires a complete rollout"
            )
        self._validate_replay()

        totals: dict[str, torch.Tensor] = {}
        num_updates = 0
        grad_norms_before: list[torch.Tensor] = []
        grad_norms_after: list[torch.Tensor] = []
        for batch in self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        ):
            loss, statistics = self._batch_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite FSPPOJointEndpoint loss; inspect observations/actions"
                )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    "Non-finite FSPPOJointEndpoint gradient before optimizer step"
                )
            clip_coefficient = (
                self.max_grad_norm / (grad_norm.detach() + 1e-6)
            ).clamp(max=1.0)
            self.optimizer.step()

            grad_norms_before.append(grad_norm.detach())
            grad_norms_after.append((grad_norm.detach() * clip_coefficient).detach())
            for name, value in statistics.items():
                totals[name] = totals.get(name, 0.0) + value.detach()
            num_updates += 1

        if num_updates == 0:
            raise RuntimeError("FSPPOJointEndpoint produced no optimizer updates")

        means = {key: value / num_updates for key, value in totals.items()}
        with torch.no_grad():
            variance = self.storage.returns.var(unbiased=False)
            explained = torch.where(
                variance > 1e-8,
                1
                - (self.storage.returns - self.storage.values).var(unbiased=False)
                / variance,
                variance.new_zeros(()),
            )
            action_std = self.storage.actions.std(dim=(0, 1), unbiased=False).mean()

        metrics = {
            "clip_param": self.clip_param,
            "approx_kl": means["endpoint_approx_kl"],
            "clip_fraction": means["clip_fraction"],
            "explained_variance": explained,
            "action_std": action_std,
            "endpoint/mae": means["endpoint_mae"],
            "endpoint/old_mae": means["endpoint_old_mae"],
            "endpoint/mae_ratio": means["ratio_mean"],
            "endpoint/mae_coefficient": self.endpoint_mae_coef,
            "endpoint/accepted_updates": num_updates,
            "mean_grad_norm_before_clip": torch.stack(grad_norms_before).mean(),
            "mean_grad_norm_after_clip": torch.stack(grad_norms_after).mean(),
        }
        self.storage.clear()
        self.update_counter += 1
        self.tot_timesteps += 1
        loss_dict = {
            "surrogate_loss": means["surrogate_loss"],
            "value_loss": means["value_loss"],
            "mae_loss": means["mae_loss"],
            "total_loss": means["total_loss"],
        }
        self._metrics_to_cpu(loss_dict, metrics)
        loss_dict["metrics"] = metrics
        return loss_dict

    def _current_sampler_contract(self) -> dict:
        return {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "num_actor_obs": self.policy.num_actor_obs,
            "num_actions": self.policy.num_actions,
            "timestep_embed_dim": self.policy.timestep_embed_dim,
            "action_perturb_std": float(self.policy.action_perturb_std),
            "actor_scale": float(self.policy.actor_scale),
            "actor_mlp_output_scale": float(self.policy.mlp_output_scale),
            "sampling_steps": self.policy.sampling_steps,
            "pmf_time_eps": float(self.policy.pmf_time_eps),
        }

    def _check_sampler_contract(self) -> None:
        if self._current_sampler_contract() != self._sampler_contract:
            raise ValueError(
                "FSPPOJointEndpoint sampling parameters must remain fixed "
                "throughout training"
            )

    @torch.no_grad()
    def _validate_replay(self) -> None:
        """Verify that stored terminal outputs still replay exactly."""
        observations = self.storage.observations.flatten(0, 1)
        latents = self.storage.action_latents.flatten(0, 1)
        endpoints = self.storage.action_means.flatten(0, 1)
        actions = self.storage.actions.flatten(0, 1)
        log_probs = self.storage.action_log_probs.flatten(0, 1)
        for start in range(0, observations.shape[0], self._REPLAY_CHUNK_SIZE):
            part = slice(start, start + self._REPLAY_CHUNK_SIZE)
            replay = self.policy.transport_actions(observations[part], latents[part])
            log_prob = self.policy.conditional_action_log_prob(actions[part], replay)
            if not torch.allclose(replay, endpoints[part], rtol=1e-4, atol=1e-6):
                error = float((replay - endpoints[part]).abs().max())
                raise RuntimeError(
                    "FSPPOJointEndpoint terminal replay mismatch; "
                    f"max endpoint error={error:.3g}"
                )
            if not torch.allclose(log_prob, log_probs[part], rtol=1e-4, atol=1e-4):
                error = float((log_prob - log_probs[part]).abs().max())
                raise RuntimeError(
                    "FSPPOJointEndpoint action-density replay mismatch; "
                    f"max log-prob error={error:.3g}"
                )

    def _batch_loss(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute endpoint MAE ratio, direct MAE, and value losses."""
        obs, actions = batch["obs"], batch["actions"]
        advantages = batch["advantages"].squeeze(-1)
        if self.normalize_advantage_per_mini_batch:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )
        positive, negative = self.advantage_clamp
        advantages = advantages.clamp(-negative, positive)

        endpoint = self.policy.transport_actions(obs, batch["action_latents"])
        old_endpoint = batch["action_means"]
        old_mae = (old_endpoint - actions).abs().mean(dim=-1)
        mae = (endpoint - actions).abs().mean(dim=-1)
        raw_log_ratio, ratio = self._compute_importance_ratio(old_mae, mae)
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
        mae_loss = mae.mean()
        weighted_mae = self.endpoint_mae_coef * mae_loss
        loss = surrogate + self.value_loss_coef * value_loss + weighted_mae
        statistics = {
            "surrogate_loss": surrogate,
            "value_loss": value_loss,
            "mae_loss": weighted_mae,
            "total_loss": loss,
            "endpoint_mae": mae_loss,
            "endpoint_old_mae": old_mae.mean(),
            "endpoint_approx_kl": 0.5 * raw_log_ratio.square().mean(),
            "clip_fraction": (
                (ratio - 1).abs() > self.clip_param
            ).float().mean(),
            "ratio_mean": ratio.mean(),
        }
        return loss, statistics
