# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Residual flow actors with time-conditioned feature modulation."""

from __future__ import annotations

import torch
from torch import nn

from isaaclab_fpo.utils import resolve_nn_activation


class _TimeConditionedResidualBlock(nn.Module):
    """Apply time-dependent scale and shift before a residual MLP."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        condition_dim: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.modulation = nn.Linear(condition_dim, 2 * input_dim)
        self.residual = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            resolve_nn_activation(activation),
            nn.Linear(output_dim, output_dim),
        )
        self.shortcut = (
            nn.Identity()
            if input_dim == output_dim
            else nn.Linear(input_dim, output_dim)
        )
        # Start near unmodulated features while allowing time/JVP gradients
        # to reach the time encoder from the first optimization step.
        nn.init.normal_(self.modulation.weight, std=0.01)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(condition).chunk(2, dim=-1)
        modulated = (1.0 + scale) * self.norm(features) + shift
        return self.shortcut(features) + self.residual(modulated)


class TimeConditionedMLP(nn.Module):
    """Encode flow time and modulate every residual block independently.

    The packed input is ``[observations, embedded_times, x_t]`` with any
    leading batch dimensions. All learned conditioning parameters live in
    this actor so frozen actor copies and EMA include the complete policy.
    Samples are processed independently, including Monte Carlo score samples.
    """

    def __init__(
        self,
        num_observations: int,
        num_actions: int,
        time_embedding_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        condition_dim: int,
        activation: str,
    ) -> None:
        """Build a time encoder and one residual block per hidden width.

        Args:
            num_observations: Width of the observation segment.
            num_actions: Width of the noisy-action segment.
            time_embedding_dim: Total width of all Fourier time embeddings.
            output_dim: Width of the velocity or denoised-action heads.
            hidden_dims: Positive block widths; different widths use projections.
            condition_dim: Positive width of the learned time representation.
            activation: Activation used inside residual blocks.
        """
        super().__init__()
        if time_embedding_dim <= 0:
            raise ValueError("time_conditioned_mlp requires timestep_embed_dim > 0")
        if condition_dim <= 0:
            raise ValueError("actor_time_condition_dim must be positive")
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("actor_hidden_dims must contain positive widths")

        self.input_dims = (num_observations, time_embedding_dim, num_actions)
        self.time_encoder = nn.Sequential(
            nn.Linear(time_embedding_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.input_layer = nn.Linear(num_observations + num_actions, hidden_dims[0])
        self.blocks = nn.ModuleList()
        input_dim = hidden_dims[0]
        for output_width in hidden_dims:
            self.blocks.append(
                _TimeConditionedResidualBlock(
                    input_dim, output_width, condition_dim, activation
                )
            )
            input_dim = output_width
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict actor heads from ``[..., obs_dim + time_dim + action_dim]``."""
        observations, embedded_times, noisy_actions = inputs.split(
            self.input_dims, dim=-1
        )
        condition = self.time_encoder(embedded_times)
        features = self.input_layer(torch.cat((observations, noisy_actions), dim=-1))
        for block in self.blocks:
            features = block(features, condition)
        return self.output_layer(features)
