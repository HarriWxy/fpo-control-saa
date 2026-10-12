# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Original pMF transport with an optional learned terminal Gaussian scale."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

from .actor_critic import PMFActorCritic

if TYPE_CHECKING:
    from isaaclab_fpo.rl_cfg import FpoRslRlPpoActorCriticCfg


class LearnedStdPMFActorCritic(PMFActorCritic):
    """Retain the pMF flow heads while learning terminal exploration noise.

    The transport remains ``F(obs, latent)`` from :class:`PMFActorCritic`.
    Terminal noise is diagonal and independent of the transport latent. Its
    parameters live inside ``actor`` so frozen behavior snapshots and EMA
    include both the map and its conditional Gaussian scale. With
    ``learn_action_perturb_std=False``, the same policy supports fixed or
    scheduled noise for comparisons with pMF projection algorithms.
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        cfg: FpoRslRlPpoActorCriticCfg,
    ) -> None:
        std_min = float(cfg.action_perturb_std_min)
        std_max = float(cfg.action_perturb_std_max)
        initial_std = float(cfg.action_perturb_std)
        if not isinstance(cfg.learn_action_perturb_std, bool):
            raise TypeError("learn_action_perturb_std must be a bool")
        if not math.isfinite(initial_std) or initial_std <= 0:
            raise ValueError("action_perturb_std must be finite and positive")
        if (
            not math.isfinite(std_min)
            or not math.isfinite(std_max)
            or not 0 < std_min < std_max
        ):
            raise ValueError(
                "action noise bounds must satisfy 0 < min < max < infinity"
            )
        if cfg.learn_action_perturb_std and not std_min < initial_std < std_max:
            raise ValueError(
                "Learned action_perturb_std must start strictly inside its bounds"
            )
        self.learn_action_perturb_std = cfg.learn_action_perturb_std
        self.action_perturb_std_min = std_min
        self.action_perturb_std_max = std_max
        super().__init__(num_actor_obs, num_critic_obs, num_actions, cfg)
        self.actor = _FlowActorWithScale(
            self.actor,
            num_actions,
            initial_std,
            std_min,
            std_max,
            self.learn_action_perturb_std,
        )

    def conditional_action_std(self, actor: nn.Module | None = None) -> torch.Tensor:
        """Return per-action terminal noise in public action coordinates.

        An explicit frozen actor retains its behavior scale. For fixed noise,
        querying the online actor synchronizes its buffer with the rollout
        schedule; a clone prevents later queries from mutating density graphs.
        """
        if actor is None:
            actor = self.actor
            if not self.learn_action_perturb_std:
                action_std = float(self.action_perturb_std)
                if not math.isfinite(action_std) or action_std <= 0:
                    raise ValueError("action_perturb_std must be finite and positive")
                actor.fixed_action_std.fill_(action_std)
        if not isinstance(actor, _FlowActorWithScale):
            raise TypeError("actor must be a pMF flow actor with a terminal scale")
        return actor.conditional_std().clone()

    def _conditional_action_std(self) -> torch.Tensor:
        return self.conditional_action_std()

    def conditional_action_log_prob(
        self,
        actions: torch.Tensor,
        means: torch.Tensor,
        std: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate the diagonal Gaussian conditional density, including scale.

        Args:
            actions: Public actions with shape ``[..., action_dim]``.
            means: Transport means with the same shape and coordinates.
            std: Optional saved scale broadcastable to the action shape.

        Returns:
            Summed log densities with shape ``[..., 1]``.
        """
        if actions.shape != means.shape:
            raise ValueError("actions and means must have identical shapes")
        if actions.ndim < 1 or actions.shape[-1] != self.num_actions:
            raise ValueError(f"actions must end with action_dim={self.num_actions}")
        if std is None:
            std = self.conditional_action_std()
        elif not torch.isfinite(std).all() or not (std > 0).all():
            raise ValueError("std must contain finite, positive values")
        std = std.to(device=actions.device, dtype=actions.dtype)
        try:
            std = torch.broadcast_to(std, actions.shape)
        except RuntimeError as error:
            raise ValueError("std must be broadcastable to the action shape") from error
        residual = (actions - means) / std
        return -0.5 * (residual.square() + 2 * std.log() + math.log(2 * math.pi)).sum(
            dim=-1, keepdim=True
        )

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """Use terminal Gaussian noise for training and inherited flow inference for evaluation."""
        if self.training:
            return self.sample_transport(observations)[0]
        return super().act(observations, **kwargs)


class _FlowActorWithScale(nn.Module):
    """Keep the original flow network and conditional scale in one snapshot."""

    def __init__(
        self,
        flow: nn.Module,
        num_actions: int,
        initial_std: float,
        std_min: float,
        std_max: float,
        learn_std: bool,
    ) -> None:
        super().__init__()
        self.flow = flow
        self.learn_std = learn_std
        self.register_buffer(
            "fixed_action_std", torch.full((num_actions,), initial_std)
        )
        self.register_buffer("action_std_min", torch.tensor(std_min))
        self.register_buffer("action_std_range", torch.tensor(std_max - std_min))
        if learn_std:
            fraction = (initial_std - std_min) / (std_max - std_min)
            initial_logit = math.log(fraction) - math.log1p(-fraction)
            self.action_std_logit = nn.Parameter(
                torch.full((num_actions,), initial_logit)
            )
        else:
            self.register_parameter("action_std_logit", None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.flow(inputs)

    def conditional_std(self) -> torch.Tensor:
        if self.learn_std:
            return self.action_std_min + self.action_std_range * torch.sigmoid(
                self.action_std_logit
            )
        return self.fixed_action_std
