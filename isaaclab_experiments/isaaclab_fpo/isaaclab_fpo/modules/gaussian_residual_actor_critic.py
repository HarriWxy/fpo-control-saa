# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gaussian control backbone with a zero-initialized pMF action residual."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

from isaaclab_fpo.utils import resolve_nn_activation

from .actor_critic import PMFActorCritic

if TYPE_CHECKING:
    from isaaclab_fpo.rl_cfg import FpoRslRlPpoActorCriticCfg


class GaussianResidualPMFActorCritic(PMFActorCritic):
    """Start from a Gaussian actor and learn an additive latent-dependent flow.

    A zero residual gives ``a = mean(s) + std * epsilon`` exactly. Both pMF
    endpoint heads contain that same state-only mean; the inherited pMF
    transport and auxiliary losses therefore share one coordinate convention.
    The bounded per-action standard deviation lives inside ``actor`` so an
    old-actor deepcopy retains the complete conditional Gaussian distribution.

    Only the one-step transport contract is supported. The evaluation action
    uses the inherited latent sampling mode and omits conditional action noise.
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        cfg: FpoRslRlPpoActorCriticCfg,
    ) -> None:
        for name in ("actor_scale", "actor_mlp_output_scale"):
            value = getattr(cfg, name)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be a finite, positive scalar")
        residual_scale = cfg.gaussian_residual_scale
        if (
            not isinstance(residual_scale, (int, float))
            or not math.isfinite(residual_scale)
            or residual_scale < 0.0
        ):
            raise ValueError("gaussian_residual_scale must be finite and non-negative")
        public_output_scale = cfg.actor_scale * cfg.actor_mlp_output_scale
        if not math.isfinite(public_output_scale) or public_output_scale <= 0.0:
            raise ValueError(
                "actor_scale * actor_mlp_output_scale must be finite and positive"
            )
        training_steps = cfg.training_sampling_steps
        if cfg.sampling_steps != 1 or training_steps not in (None, 1):
            raise ValueError(
                "GaussianResidualPMFActorCritic requires one sampling step"
            )
        if cfg.pmf_time_eps > 1.0:
            raise ValueError("pmf_time_eps must be at most 1 for one-step transport")

        std_min = float(cfg.action_perturb_std_min)
        std_max = float(cfg.action_perturb_std_max)
        initial_std = float(cfg.action_perturb_std)
        if (
            not math.isfinite(std_min)
            or not math.isfinite(std_max)
            or not 0.0 < std_min < std_max
        ):
            raise ValueError(
                "action noise bounds must satisfy 0 < min < max < infinity"
            )
        if not math.isfinite(initial_std) or initial_std <= 0.0:
            raise ValueError("action_perturb_std must be finite and positive")
        if not isinstance(cfg.learn_action_perturb_std, bool):
            raise ValueError("learn_action_perturb_std must be a bool")
        self.learn_action_perturb_std = cfg.learn_action_perturb_std
        self.action_perturb_std_min = std_min
        self.action_perturb_std_max = std_max
        self.gaussian_residual_scale = float(residual_scale)
        if self.learn_action_perturb_std and not std_min < initial_std < std_max:
            raise ValueError(
                "Learned action_perturb_std must start strictly inside its bounds"
            )

        super().__init__(num_actor_obs, num_critic_obs, num_actions, cfg)
        residual = self.actor
        final_layer = residual[-1]
        if not isinstance(final_layer, nn.Linear):
            raise TypeError("The pMF residual must end with a linear layer")
        nn.init.zeros_(final_layer.weight)
        if final_layer.bias is not None:
            nn.init.zeros_(final_layer.bias)

        mean_layers: list[nn.Module] = []
        input_dim = num_actor_obs
        for hidden_dim in cfg.actor_hidden_dims:
            mean_layers.extend(
                (
                    nn.Linear(input_dim, hidden_dim),
                    resolve_nn_activation(cfg.activation),
                )
            )
            input_dim = hidden_dim
        mean_layers.append(nn.Linear(input_dim, num_actions))
        mean = nn.Sequential(*mean_layers)
        if cfg.actor_final_layer_weight_scale is not None:
            with torch.no_grad():
                mean[-1].weight.mul_(cfg.actor_final_layer_weight_scale)
                mean[-1].bias.mul_(cfg.actor_final_layer_weight_scale)

        self.actor = _GaussianResidualActor(
            mean=mean,
            residual=residual,
            num_actor_obs=num_actor_obs,
            num_actions=num_actions,
            public_output_scale=public_output_scale,
            residual_scale=float(cfg.gaussian_residual_scale),
            initial_std=initial_std,
            std_min=std_min,
            std_max=std_max,
            learn_std=self.learn_action_perturb_std,
        )

    def conditional_action_std(self, actor: nn.Module | None = None) -> torch.Tensor:
        """Return the conditional Gaussian scale in public action coordinates.

        With fixed noise, the current actor follows ``action_perturb_std`` so
        the existing rollout sigma schedule remains usable. An explicitly
        supplied frozen actor retains its saved rollout scale.

        Args:
            actor: Optional frozen actor snapshot, including its noise scale.

        Returns:
            Standard deviations with shape ``[action_dim]``.
        """
        if actor is None:
            actor = self.actor
            if not self.learn_action_perturb_std:
                action_std = float(self.action_perturb_std)
                if not math.isfinite(action_std) or action_std <= 0.0:
                    raise ValueError("action_perturb_std must be finite and positive")
                actor.fixed_action_std.fill_(action_std)
        if not isinstance(actor, _GaussianResidualActor):
            raise TypeError("actor must be a Gaussian residual actor snapshot")
        # Fixed-noise queries update the snapshot buffer for rollout schedules.
        # A density graph must retain its own scale: a later KL/entropy query
        # must not change the version of a tensor saved for backward.
        return actor.conditional_std().clone()

    def _conditional_action_std(self) -> torch.Tensor:
        """Provide the per-action scale to the inherited rollout sampler."""
        return self.conditional_action_std()

    def conditional_action_log_prob(
        self,
        actions: torch.Tensor,
        means: torch.Tensor,
        std: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate a complete diagonal Gaussian density, including log scale.

        Args:
            actions: Public actions with shape ``[..., action_dim]``.
            means: Conditional means with the same shape and units as actions.
            std: Optional saved standard deviations broadcastable to actions.
                By default, use the current actor's scale.

        Returns:
            Log densities summed over action dimensions, shape ``[..., 1]``.
        """
        if actions.shape != means.shape:
            raise ValueError("actions and means must have identical shapes")
        if actions.ndim < 1 or actions.shape[-1] != self.num_actions:
            raise ValueError(f"actions must end with action_dim={self.num_actions}")
        if std is None:
            std = self.conditional_action_std()
        else:
            if not torch.isfinite(std).all() or not (std > 0.0).all():
                raise ValueError("std must contain finite, positive values")
        std = std.to(device=actions.device, dtype=actions.dtype)
        try:
            std = torch.broadcast_to(std, actions.shape)
        except RuntimeError as error:
            raise ValueError("std must be broadcastable to the action shape") from error
        standardized_residual = (actions - means) / std
        return -0.5 * (
            standardized_residual.square() + 2.0 * std.log() + math.log(2.0 * math.pi)
        ).sum(dim=-1, keepdim=True)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """Use the same conditional Gaussian for training rollouts and replay."""
        if self.training:
            return self.sample_transport(observations)[0]
        return super().act(observations, **kwargs)


class _GaussianResidualActor(nn.Module):
    """Own every conditional-policy parameter for frozen actor snapshots."""

    def __init__(
        self,
        mean: nn.Module,
        residual: nn.Module,
        num_actor_obs: int,
        num_actions: int,
        public_output_scale: float,
        residual_scale: float,
        initial_std: float,
        std_min: float,
        std_max: float,
        learn_std: bool,
    ) -> None:
        super().__init__()
        self.mean = mean
        self.residual = residual
        self.num_actor_obs = num_actor_obs
        self.public_output_scale = public_output_scale
        self.residual_scale = residual_scale
        self.learn_std = learn_std
        self.register_buffer(
            "fixed_action_std", torch.full((num_actions,), initial_std)
        )
        self.register_buffer("action_std_min", torch.tensor(std_min))
        self.register_buffer("action_std_range", torch.tensor(std_max - std_min))
        if learn_std:
            fraction = (initial_std - std_min) / (std_max - std_min)
            logit = math.log(fraction) - math.log1p(-fraction)
            self.action_std_logit = nn.Parameter(torch.full((num_actions,), logit))
        else:
            self.register_parameter("action_std_logit", None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # The backbone predicts public actions, while PMFActorCritic expects
        # endpoints before its actor_scale * mlp_output_scale multiplication.
        mean = self.mean(inputs[..., : self.num_actor_obs]) / self.public_output_scale
        mean_heads = torch.cat((mean, mean), dim=-1)
        return mean_heads + self.residual_scale * self.residual(inputs)

    def conditional_std(self) -> torch.Tensor:
        """Return a positive per-action standard deviation in public units."""
        if self.learn_std:
            return self.action_std_min + self.action_std_range * torch.sigmoid(
                self.action_std_logit
            )
        return self.fixed_action_std
