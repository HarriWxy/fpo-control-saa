"""State-wise policy mirror descent followed by independent pMF projection."""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

import torch
from torch import nn

from isaaclab_fpo.algorithms.fsppo_joint import FSPPOJoint
from isaaclab_fpo.modules import PMFActorCritic

if TYPE_CHECKING:
    from isaaclab_fpo.rl_cfg import FpoRslRlPpoAlgorithmCfg


class ActionValue(nn.Module):
    """Estimate on-policy lambda returns conditional on state and raw action."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = observation_dim + action_dim
        for output_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, output_dim), nn.ELU()))
            input_dim = output_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return self.network(torch.cat((observations, actions), dim=-1))


class PMFPMD(FSPPOJoint):
    """Improve old candidate means with Q, then project their weighted distribution.

    This inherits collection and one-NFE sampler contracts only. Its actor never
    uses a PPO likelihood-ratio surrogate. Q is an additional action-conditioned
    critic, fitted to frozen on-policy lambda returns; V supplies their bootstrap.
    Q is evaluated on raw public actions, including independent terminal noise.
    Candidates and state-wise probabilities stay frozen throughout projection.
    Independent flow noise makes projection a distribution fit, not latent replay.

    The optional trust region is a transactional empirical fixed-probe mean of
    squared same-latent displacement / (2 sigma^2). It bounds the coupled joint
    Gaussian KL in expectation, not every state or the population marginal KL.
    """

    def __init__(
        self,
        policy: PMFActorCritic,
        cfg: FpoRslRlPpoAlgorithmCfg,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
    ):
        if type(policy) is not PMFActorCritic:
            raise TypeError(
                "PMFPMD requires PMFActorCritic with a fixed terminal Gaussian"
            )
        if cfg.fsppo_joint_aux_loss_coef != 0 or cfg.fsppo_joint_entropy_coef != 0:
            raise ValueError(
                "PMFPMD replaces the actor objective; disable Joint auxiliary/entropy losses"
            )
        super().__init__(policy, cfg, device=device, multi_gpu_cfg=multi_gpu_cfg)
        for name in (
            "pmd_num_candidates",
            "pmd_q_noise_samples",
            "pmd_q_epochs",
            "pmd_projection_epochs",
            "pmd_batch_size",
        ):
            value = getattr(cfg, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if cfg.pmd_num_candidates < 2:
            raise ValueError("PMFPMD needs at least two candidates per state")
        if (
            isinstance(cfg.pmd_q_warmup_updates, bool)
            or not isinstance(cfg.pmd_q_warmup_updates, int)
            or cfg.pmd_q_warmup_updates < 0
        ):
            raise ValueError("pmd_q_warmup_updates must be a non-negative integer")
        for name in (
            "pmd_temperature",
            "pmd_q_learning_rate",
            "pmd_value_learning_rate",
        ):
            if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < cfg.pmd_min_ess_fraction < 1:
            raise ValueError("pmd_min_ess_fraction must be in (0, 1)")
        if not 0 < cfg.pmd_q_holdout_fraction < 0.5:
            raise ValueError("pmd_q_holdout_fraction must be in (0, 0.5)")
        if not cfg.pmd_q_hidden_dims or any(
            isinstance(x, bool) or not isinstance(x, int) or x < 1
            for x in cfg.pmd_q_hidden_dims
        ):
            raise ValueError("pmd_q_hidden_dims must contain positive integers")
        self.num_candidates = cfg.pmd_num_candidates
        self.q_noise_samples = cfg.pmd_q_noise_samples
        self.q_epochs = cfg.pmd_q_epochs
        self.projection_epochs = cfg.pmd_projection_epochs
        self.batch_size = cfg.pmd_batch_size
        self.q_warmup_updates = cfg.pmd_q_warmup_updates
        self.min_ess_fraction = cfg.pmd_min_ess_fraction
        self.q_holdout_fraction = cfg.pmd_q_holdout_fraction
        self.temperature = cfg.pmd_temperature
        critic_dim = next(
            layer for layer in policy.critic.modules() if isinstance(layer, nn.Linear)
        ).in_features
        self.q = ActionValue(critic_dim, policy.num_actions, cfg.pmd_q_hidden_dims).to(
            device
        )
        # Keep the runner's optimizer API, but make it actor-only. V and Q have
        # independent moments and cannot change during an actor transaction.
        optimizer_class = (
            torch.optim.AdamW if cfg.weight_decay > 0 else torch.optim.Adam
        )
        optimizer_kwargs = {"betas": cfg.adam_betas}
        if cfg.weight_decay > 0:
            optimizer_kwargs["weight_decay"] = cfg.weight_decay
        self.optimizer = optimizer_class(
            policy.actor.parameters(), lr=cfg.learning_rate, **optimizer_kwargs
        )
        self.value_optimizer = optimizer_class(
            policy.critic.parameters(),
            lr=cfg.pmd_value_learning_rate,
            **optimizer_kwargs,
        )
        self.q_optimizer = optimizer_class(
            self.q.parameters(), lr=cfg.pmd_q_learning_rate, **optimizer_kwargs
        )
        self._pmd_contract = {
            "num_candidates": self.num_candidates,
            "q_noise_samples": self.q_noise_samples,
            "q_hidden_dims": list(cfg.pmd_q_hidden_dims),
            "q_observation_dim": critic_dim,
            "q_target": "on_policy_timeout_corrected_lambda_returns",
            "gamma": self.gamma,
            "lambda": self.lam,
            "q_warmup_updates": self.q_warmup_updates,
            "q_epochs": self.q_epochs,
            "projection_epochs": self.projection_epochs,
            "batch_size": self.batch_size,
            "minimum_ess_fraction": self.min_ess_fraction,
            "initial_temperature": cfg.pmd_temperature,
            "q_holdout_fraction": self.q_holdout_fraction,
            "trust_region": self.enable_budget,
            "trust_metric": "squared_public_map_over_2_sigma_squared",
            "kl_target": self.kl_target,
            "value_learning_rate": cfg.pmd_value_learning_rate,
            "q_learning_rate": cfg.pmd_q_learning_rate,
            "actor_learning_rate": cfg.learning_rate,
            "value_epochs": self.num_learning_epochs,
            "actor_minibatches": self.num_mini_batches,
            "value_loss_coef": self.value_loss_coef,
            "value_clipping": self.use_clipped_value_loss,
            "value_clip_param": self.clip_param,
            "max_grad_norm": self.max_grad_norm,
            "weight_decay": cfg.weight_decay,
            "adam_betas": tuple(cfg.adam_betas),
            "flow_samples": self.n_samples_per_action,
            "flow_time_mean": policy.pmf_logit_normal_mean,
            "flow_time_std": policy.pmf_logit_normal_std,
            "flow_anchor_fraction": policy.pmf_fm_proportion,
            "flow_v_coef": policy.pmf_aux_v_loss_coef,
            "flow_gradient_normalization_p": policy.pmf_adaptive_gradient_norm_p,
            "flow_gradient_normalization_eps": policy.pmf_adaptive_gradient_norm_eps,
            "probe_size": self.probe_size,
            "map_samples": self.map_samples,
            "max_backtracks": self.max_backtracks,
            "backtrack_factor": self.backtrack_factor,
        }

    @torch.no_grad()
    def process_env_step(self, rewards, dones, infos):
        """Bootstrap time limits from terminal observations before auto reset.

        The PMD wrapper must supply terminal critic observations in the same
        normalized coordinates as V. Using V of the pre-action or reset state
        would contaminate action-conditioned Q supervision.
        """
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        timeouts = infos.get("time_outs")
        if timeouts is not None and bool(timeouts.any()):
            terminal_obs = infos.get("terminal_critic_observations")
            if terminal_obs is None:
                raise ValueError(
                    "PMFPMD timeout requires normalized terminal_critic_observations before reset"
                )
            if terminal_obs.shape != self.transition.privileged_observations.shape:
                raise ValueError("PMFPMD terminal critic observation shape mismatch")
            terminal_values = self.policy.evaluate(terminal_obs).squeeze(-1)
            self.transition.rewards += (
                self.gamma * terminal_values * timeouts.to(self.device)
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def state_dict(self) -> dict:
        """Save all PMD learners, optimizer moments and adaptive temperature."""
        return {
            "version": 2,
            "sampler": dict(self._sampler_contract),
            "pmd_contract": copy.deepcopy(self._pmd_contract),
            "temperature": self.temperature,
            "update_counter": self.update_counter,
            "tot_timesteps": self.tot_timesteps,
            "action_noise_completed_env_steps": self.action_noise_completed_env_steps,
            "actor": copy.deepcopy(self.policy.actor.state_dict()),
            "value": copy.deepcopy(self.policy.critic.state_dict()),
            "q": copy.deepcopy(self.q.state_dict()),
            "actor_optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "value_optimizer": copy.deepcopy(self.value_optimizer.state_dict()),
            "q_optimizer": copy.deepcopy(self.q_optimizer.state_dict()),
        }

    def load_state_dict(self, state: dict) -> None:
        """Reject incompatible distributions and restore all update state."""
        if (
            state.get("version") != 2
            or state.get("sampler") != self._sampler_contract
            or state.get("pmd_contract") != self._pmd_contract
        ):
            raise ValueError(
                "PMFPMD checkpoint sampling/optimization contract mismatch"
            )
        temperature = state["temperature"]
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Invalid PMFPMD checkpoint temperature")
        for name in (
            "update_counter",
            "tot_timesteps",
            "action_noise_completed_env_steps",
        ):
            value = state[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Invalid PMFPMD checkpoint {name}")
        self.policy.actor.load_state_dict(state["actor"])
        self.policy.critic.load_state_dict(state["value"])
        self.q.load_state_dict(state["q"])
        self.optimizer.load_state_dict(state["actor_optimizer"])
        self.value_optimizer.load_state_dict(state["value_optimizer"])
        self.q_optimizer.load_state_dict(state["q_optimizer"])
        self.temperature = temperature
        self.update_counter = state["update_counter"]
        self.tot_timesteps = state["tot_timesteps"]
        self.begin_rollout(state["action_noise_completed_env_steps"])

    @torch.no_grad()
    def candidate_weights(
        self, q_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize Gibbs probabilities separately per state and enforce ESS.

        Returns probabilities summing to one along the candidate dimension and
        state-wise temperatures. A binary search increases temperature only for
        states whose normalized ESS is below the configured minimum.
        """
        if (
            q_values.ndim != 2
            or q_values.shape[1] != self.num_candidates
            or not bool(torch.isfinite(q_values).all())
        ):
            raise ValueError("PMFPMD requires finite [states, candidates] Q values")
        centered = q_values - q_values.max(dim=-1, keepdim=True).values
        lower = q_values.new_full((q_values.shape[0], 1), self.temperature)
        upper = lower.clone()

        def ess(temperature):
            weights = (centered / temperature).softmax(dim=-1)
            return 1 / (
                self.num_candidates * weights.square().sum(dim=-1, keepdim=True)
            )

        needs_adjustment = ess(lower) < self.min_ess_fraction
        for _ in range(64):
            insufficient = ess(upper) < self.min_ess_fraction
            if not bool(insufficient.any()):
                break
            upper = torch.where(insufficient, upper * 2, upper)
        else:
            raise FloatingPointError("PMFPMD ESS temperature search did not converge")
        for _ in range(20):
            midpoint = (lower + upper) / 2
            insufficient = ess(midpoint) < self.min_ess_fraction
            lower = torch.where(insufficient, midpoint, lower)
            upper = torch.where(insufficient, upper, midpoint)
        temperatures = torch.where(
            needs_adjustment, upper, torch.full_like(upper, self.temperature)
        )
        return (centered / temperatures).softmax(dim=-1).detach(), temperatures.detach()

    @torch.no_grad()
    def _freeze_targets(self, observations, critic_observations):
        """Sample fresh old-policy candidates and freeze noise-smoothed Q scores."""
        candidates, q_values, weights, temperatures = [], [], [], []
        frozen_q = copy.deepcopy(self.q).requires_grad_(False).eval()
        for start in range(0, observations.shape[0], self.batch_size):
            obs = observations[start : start + self.batch_size]
            critic_obs = critic_observations[start : start + self.batch_size]
            latent = torch.randn(
                obs.shape[0],
                self.num_candidates,
                self.policy.num_actions,
                device=self.device,
            )
            means = self.policy.transport_actions(obs, latent, actor=self._old_actor)
            terminal_noise = torch.randn(
                obs.shape[0],
                self.num_candidates,
                self.q_noise_samples,
                self.policy.num_actions,
                device=self.device,
            )
            actions = means[:, :, None, :] + self.sigma * terminal_noise
            expanded_obs = critic_obs[:, None, None, :].expand(
                -1, self.num_candidates, self.q_noise_samples, -1
            )
            scores = (
                frozen_q(
                    expanded_obs.reshape(-1, critic_obs.shape[-1]),
                    actions.reshape(-1, self.policy.num_actions),
                )
                .reshape(obs.shape[0], self.num_candidates, self.q_noise_samples)
                .mean(dim=-1)
            )
            probabilities, eta = self.candidate_weights(scores)
            candidates.append(means)
            q_values.append(scores)
            weights.append(probabilities)
            temperatures.append(eta)
        return tuple(
            torch.cat(parts).detach()
            for parts in (candidates, q_values, weights, temperatures)
        )

    def _projection_loss(self, observations, candidates, weights):
        """Fit all weighted candidate means using fresh independent flow noise."""
        count, candidate_count, action_dim = candidates.shape
        obs = (
            observations[:, None, :]
            .expand(-1, candidate_count, -1)
            .reshape(-1, observations.shape[-1])
        )
        targets = candidates.reshape(-1, action_dim)
        shape = (count * candidate_count, self.n_samples_per_action, 1)
        times = torch.sigmoid(
            torch.randn(2, *shape, device=self.device)
            * self.policy.pmf_logit_normal_std
            + self.policy.pmf_logit_normal_mean
        )
        anchor_count = int(shape[0] * shape[1] * self.policy.pmf_fm_proportion)
        anchors = torch.randperm(shape[0] * shape[1], device=self.device)[:anchor_count]
        times[1].view(-1)[anchors] = times[0].view(-1)[anchors]
        t, r = times.max(dim=0).values, times.min(dim=0).values
        # This draw has no relation to latent used to generate candidates.
        noise = torch.randn(*shape[:2], action_dim, device=self.device)
        scores, _, _, components = self.policy.get_pmf_loss(
            obs, targets, noise, r, t, return_components=True
        )
        per_candidate = scores.mean(dim=-1).reshape(count, candidate_count)
        loss = (per_candidate * weights).sum(dim=-1).mean()
        metrics = {
            f"pmd/{name}": (
                component.mean(-1).reshape(count, candidate_count) * weights
            )
            .sum(-1)
            .mean()
            .detach()
            for name, component in components.items()
            if name in ("u_loss", "v_loss", "jvp_norm")
        }
        return loss, metrics

    @torch.no_grad()
    def _probe_distance(self, probe):
        actions = self.policy.transport_actions(probe["obs"], probe["noise"])
        return (actions - probe["old_actions"]).square().sum(dim=-1) / (
            2 * self.sigma**2
        )

    def _actor_step(self, probe):
        """Accept empirical KL or roll back actor and its full optimizer state."""
        if not self.enable_budget:
            self.optimizer.step()
            if not all(
                bool(torch.isfinite(p).all()) for p in self.policy.actor.parameters()
            ):
                raise FloatingPointError("Non-finite PMFPMD actor after projection")
            return True, 0
        actor_state = copy.deepcopy(self.policy.actor.state_dict())
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        base_lrs = [group["lr"] for group in self.optimizer.param_groups]
        for attempt in range(self.max_backtracks + 1):
            if attempt:
                self.policy.actor.load_state_dict(actor_state)
                self.optimizer.load_state_dict(copy.deepcopy(optimizer_state))
            for group, lr in zip(self.optimizer.param_groups, base_lrs):
                group["lr"] = lr * self.backtrack_factor**attempt
            self.optimizer.step()
            distance = self._probe_distance(probe)
            if (
                bool(torch.isfinite(distance).all())
                and all(
                    bool(torch.isfinite(p).all())
                    for p in self.policy.actor.parameters()
                )
                and float(distance.mean()) <= self.kl_target
            ):
                for group, lr in zip(self.optimizer.param_groups, base_lrs):
                    group["lr"] = lr
                return True, attempt
        self.policy.actor.load_state_dict(actor_state)
        self.optimizer.load_state_dict(optimizer_state)
        return False, self.max_backtracks + 1

    def _train_step(self, loss, optimizer, parameters):
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite PMFPMD critic/projection loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._clip_model_gradients(parameters)

    def _clip_model_gradients(self, parameters):
        """Check and clip one model after all microbatch gradients accumulated."""
        parameters = list(parameters)
        if not all(
            p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters
        ):
            raise FloatingPointError("Non-finite PMFPMD gradient")
        torch.nn.utils.clip_grad_norm_(
            parameters, self.max_grad_norm, error_if_nonfinite=True
        )

    def update(self, obs_normalizer=None, privileged_obs_normalizer=None) -> dict:
        """Fit V/Q, freeze improvement targets, and run pure pMF projection."""
        del obs_normalizer, privileged_obs_normalizer
        self._check_sampler_contract()
        if self.storage.step != self.storage.num_transitions_per_env:
            raise RuntimeError("PMFPMD.update requires a complete rollout")
        self._old_actor.load_state_dict(self.policy.actor.state_dict())
        self._validate_replay()
        obs = self.storage.observations.flatten(0, 1)
        critic_obs = (
            obs
            if self.storage.privileged_observations is None
            else self.storage.privileged_observations.flatten(0, 1)
        )
        actions = self.storage.actions.flatten(0, 1)
        old_values = self.storage.values.flatten(0, 1).detach()
        targets = self.storage.returns.flatten(0, 1).detach().clone()
        if not bool(torch.isfinite(targets).all()):
            raise FloatingPointError("Non-finite PMFPMD frozen lambda return")
        size = obs.shape[0]
        if size < 3:
            raise ValueError(
                "PMFPMD needs at least three rollout transitions for Q holdout"
            )
        if self.num_mini_batches > size:
            raise ValueError("PMFPMD num_mini_batches exceeds rollout transitions")
        permutation = torch.randperm(size, device=self.device)
        holdout_count = max(1, int(size * self.q_holdout_fraction))
        validation_indices, train_indices = (
            permutation[:holdout_count],
            permutation[holdout_count:],
        )
        q_losses, value_losses = [], []
        for _ in range(self.num_learning_epochs):
            for indices in torch.randperm(size, device=self.device).tensor_split(
                self.num_mini_batches
            ):
                self.value_optimizer.zero_grad(set_to_none=True)
                accumulated_loss = targets.new_zeros(())
                for micro_indices in indices.split(self.batch_size):
                    prediction = self.policy.evaluate(critic_obs[micro_indices])
                    squared_error = (prediction - targets[micro_indices]).square()
                    if self.use_clipped_value_loss:
                        clipped = old_values[micro_indices] + (
                            prediction - old_values[micro_indices]
                        ).clamp(-self.clip_param, self.clip_param)
                        squared_error = torch.maximum(
                            squared_error, (clipped - targets[micro_indices]).square()
                        )
                    value_loss = squared_error.mean() * (
                        micro_indices.numel() / indices.numel()
                    )
                    if not bool(torch.isfinite(value_loss)):
                        raise FloatingPointError("Non-finite PMFPMD value loss")
                    (value_loss * self.value_loss_coef).backward()
                    accumulated_loss += value_loss.detach()
                self._clip_model_gradients(self.policy.critic.parameters())
                self.value_optimizer.step()
                value_losses.append(accumulated_loss)
        for _ in range(self.q_epochs):
            shuffled = train_indices[
                torch.randperm(train_indices.numel(), device=self.device)
            ]
            for indices in shuffled.split(self.batch_size):
                q_loss = (
                    (self.q(critic_obs[indices], actions[indices]) - targets[indices])
                    .square()
                    .mean()
                )
                self._train_step(q_loss, self.q_optimizer, self.q.parameters())
                self.q_optimizer.step()
                q_losses.append(q_loss.detach())
        with torch.no_grad():
            validation_targets = targets[validation_indices]
            validation_prediction = self.q(
                critic_obs[validation_indices], actions[validation_indices]
            )
            validation_mse = (
                (validation_prediction - validation_targets).square().mean()
            )
            validation_variance = validation_targets.var(unbiased=False)
            q_explained = torch.where(
                validation_variance > 1e-8,
                1
                - (validation_prediction - validation_targets).var(unbiased=False)
                / validation_variance.clamp_min(1e-8),
                validation_variance.new_zeros(()),
            )
            value_error = self.policy.evaluate(critic_obs) - targets
            return_variance = targets.var(unbiased=False)
            explained = torch.where(
                return_variance > 1e-8,
                1 - value_error.var(unbiased=False) / return_variance.clamp_min(1e-8),
                return_variance.new_zeros(()),
            )
        warmup = self.update_counter < self.q_warmup_updates
        metrics = {
            "pmd/q_warmup": float(warmup),
            "pmd/q_validation_mse": validation_mse,
            "pmd/q_validation_explained_variance": q_explained,
            "pmd/q_holdout_count": holdout_count,
            "explained_variance": explained,
            "joint/action_noise_std": self.sigma,
            "joint/action_noise_completed_env_steps": self.action_noise_completed_env_steps,
            "joint/action_noise_schedule_progress": self.action_noise_schedule_progress,
            "pmd/budget_enabled": float(self.enable_budget),
            "pmd/kl_target": self.kl_target,
        }
        projection_losses, accepted, rejected = [], 0, 0
        attempted = flow_examples = 0
        if not warmup:
            candidates, q_values, weights, temperatures = self._freeze_targets(
                obs, critic_obs
            )
            ess = 1 / (self.num_candidates * weights.square().sum(dim=-1))
            metrics.update(
                {
                    "pmd/candidate_q_mean": q_values.mean(),
                    "pmd/candidate_q_std": q_values.std(dim=-1, unbiased=False).mean(),
                    "pmd/weighted_q_gain": (
                        (weights * q_values).sum(-1) - q_values.mean(-1)
                    ).mean(),
                    "pmd/candidate_mean_std": candidates.std(
                        dim=1, unbiased=False
                    ).mean(),
                    "pmd/ess_fraction_mean": ess.mean(),
                    "pmd/ess_fraction_min": ess.min(),
                    "pmd/weight_max": weights.max(),
                    "pmd/temperature_mean": temperatures.mean(),
                    "pmd/temperature_max": temperatures.max(),
                }
            )
            self.temperature = float(temperatures.median())
            probe = self._make_probe()
            stopped = False
            for _ in range(self.projection_epochs):
                for indices in torch.randperm(size, device=self.device).tensor_split(
                    self.num_mini_batches
                ):
                    attempted += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    accumulated_loss = targets.new_zeros(())
                    diagnostics = {}
                    for micro_indices in indices.split(self.batch_size):
                        loss, micro_diagnostics = self._projection_loss(
                            obs[micro_indices],
                            candidates[micro_indices],
                            weights[micro_indices],
                        )
                        fraction = micro_indices.numel() / indices.numel()
                        flow_examples += (
                            micro_indices.numel()
                            * self.num_candidates
                            * self.n_samples_per_action
                        )
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError(
                                "Non-finite PMFPMD projection loss"
                            )
                        (loss * fraction).backward()
                        accumulated_loss += loss.detach() * fraction
                        for name, value in micro_diagnostics.items():
                            diagnostics[name] = (
                                diagnostics.get(name, 0.0) + value * fraction
                            )
                    self._clip_model_gradients(self.policy.actor.parameters())
                    was_accepted, rejection_count = self._actor_step(probe)
                    rejected += rejection_count
                    if not was_accepted:
                        stopped = True
                        break
                    accepted += 1
                    projection_losses.append(accumulated_loss)
                    for name, value in diagnostics.items():
                        metrics[name] = metrics.get(name, 0.0) + value
                if stopped:
                    break
            for name in ("pmd/u_loss", "pmd/v_loss", "pmd/jvp_norm"):
                if name in metrics:
                    metrics[name] /= max(accepted, 1)
            if self.enable_budget:
                with torch.no_grad():
                    for label, sample in (
                        ("probe", self._probe_distance(probe)),
                        ("audit", self._probe_distance(self._make_probe())),
                    ):
                        if not bool(torch.isfinite(sample).all()):
                            raise FloatingPointError("Non-finite PMFPMD fresh KL audit")
                        metrics[f"pmd/{label}_conditional_joint_kl_mean"] = (
                            sample.mean()
                        )
                        metrics[f"pmd/{label}_conditional_joint_kl_p95"] = (
                            torch.quantile(sample.flatten(), 0.95)
                        )
        metrics["pmd/accepted_projection_steps"] = accepted
        metrics["pmd/attempted_projection_steps"] = attempted
        metrics["pmd/actor_optimizer_trials"] = accepted + rejected
        metrics["pmd/projection_flow_examples"] = flow_examples
        metrics["pmd/q_optimizer_steps"] = len(q_losses)
        metrics["pmd/value_optimizer_steps"] = len(value_losses)
        metrics["pmd/candidate_q_evaluations"] = (
            0 if warmup else size * self.num_candidates * self.q_noise_samples
        )
        metrics["pmd/rejected_candidates"] = rejected
        metrics["pmd/actor_changed"] = float(
            any(
                not torch.equal(p, old)
                for p, old in zip(
                    self.policy.actor.parameters(), self._old_actor.parameters()
                )
            )
        )
        losses = {
            "value_loss": torch.stack(value_losses).mean(),
            "q_loss": torch.stack(q_losses).mean(),
            "pmf_projection_loss": torch.stack(projection_losses).mean()
            if projection_losses
            else targets.new_zeros(()),
        }
        losses["total_loss"] = sum(losses.values())
        self.storage.clear()
        self.update_counter += 1
        self.tot_timesteps += 1
        self._metrics_to_cpu(losses, metrics)
        losses["metrics"] = metrics
        return losses
