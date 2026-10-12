# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint PPO with pMF transport and Gaussian-PPO optimization controls."""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

import torch

from isaaclab_fpo.modules import LearnedStdPMFActorCritic

from .fsppo_joint import FSPPOJoint

if TYPE_CHECKING:
    from isaaclab_fpo.rl_cfg import FpoRslRlPpoAlgorithmCfg


class FSPPOJointAligned(FSPPOJoint):
    """Use joint PPO without the legacy map budget and optionally adapt LR.

    Adaptive learning rates use ``KL(old || current)`` of the conditional
    diagonal Gaussians, averaged over saved rollout states and generating
    latents. This is an empirical joint KL, rather than marginal mixture KL.
    PPO clipping, value clipping and conditional entropy remain independent
    controls. Set the auxiliary coefficient to zero for experiment A, or keep
    the original pMF regression for experiment D.
    """

    def __init__(
        self,
        policy: LearnedStdPMFActorCritic,
        cfg: FpoRslRlPpoAlgorithmCfg,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        if not isinstance(policy, LearnedStdPMFActorCritic):
            raise TypeError("FSPPOJointAligned requires LearnedStdPMFActorCritic")
        if cfg.fsppo_joint_enable_budget:
            raise ValueError(
                "FSPPOJointAligned requires fsppo_joint_enable_budget=False"
            )
        if cfg.schedule not in ("fixed", "adaptive"):
            raise ValueError("FSPPOJointAligned schedule must be 'fixed' or 'adaptive'")
        if cfg.schedule == "adaptive" and (
            cfg.desired_kl is None
            or not math.isfinite(cfg.desired_kl)
            or cfg.desired_kl <= 0
        ):
            raise ValueError("Adaptive joint PPO requires a finite positive desired_kl")
        # The legacy implementation deliberately rejects adaptive scheduling.
        # This independent class extends it without changing that old contract.
        fixed_cfg = copy.copy(cfg)
        fixed_cfg.schedule = "fixed"
        super().__init__(policy, fixed_cfg, device=device, multi_gpu_cfg=multi_gpu_cfg)
        self.schedule = cfg.schedule

    def _batch_loss(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss, statistics = super()._batch_loss(batch)
        conditional_kl = statistics["map_kl"].detach()
        if not torch.isfinite(conditional_kl):
            raise FloatingPointError("Non-finite empirical conditional Gaussian KL")
        if self.schedule == "adaptive":
            kl = float(conditional_kl)
            if kl > self.desired_kl * 2:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif 0 < kl < self.desired_kl / 2:
                self.learning_rate = min(1e-2, self.learning_rate * 1.5)
            for group in self.optimizer.param_groups:
                group["lr"] = self.learning_rate
        statistics["aligned_conditional_joint_kl"] = conditional_kl
        return loss, statistics

    def _extra_update_metrics(self, means: dict[str, torch.Tensor]) -> dict:
        return {
            "joint/aligned_conditional_joint_kl": means.get(
                "aligned_conditional_joint_kl", 0.0
            ),
            "joint/adaptive_lr_enabled": float(self.schedule == "adaptive"),
            "joint/adaptive_learning_rate": self.learning_rate,
        }

    def state_dict(self) -> dict:
        state = super().state_dict()
        state.update(
            aligned_schedule=self.schedule,
            aligned_desired_kl=self.desired_kl,
            aligned_learning_rate=self.learning_rate,
        )
        return state

    def load_state_dict(self, state: dict) -> None:
        if (
            state.get("aligned_schedule") != self.schedule
            or state.get("aligned_desired_kl") != self.desired_kl
        ):
            raise ValueError("Aligned joint checkpoint learning-rate contract mismatch")
        rate = state.get("aligned_learning_rate")
        if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("Invalid aligned joint checkpoint learning rate")
        super().load_state_dict(state)
        self.learning_rate = float(rate)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate
