# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rollout storage for explicit latent-action joint policy updates."""

from __future__ import annotations

from collections.abc import Generator

import torch

from .rollout_storage import RolloutStorage


class JointRolloutStorage(RolloutStorage):
    """Store rollouts with replayable transport latents and action densities.

    The base storage continues to own the common observation, reward, value,
    return, advantage, and GAE implementation.  Its legacy flow-score arrays
    are allocated with zero samples because the joint policy path replays the
    stored latent, mean, and conditional action log-probability instead.
    """

    class Transition(RolloutStorage.Transition):
        """One joint-policy transition before it is copied into storage."""

        def __init__(self):
            super().__init__()
            self.action_latent = None
            self.action_mean = None
            self.action_log_prob = None

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs_shape: tuple[int, ...] | list[int],
        privileged_obs_shape: tuple[int, ...] | list[int] | None,
        actions_shape: tuple[int, ...] | list[int],
        device: str | torch.device = "cpu",
    ):
        """Initialize storage without allocating legacy flow-score tensors.

        Args:
            num_envs: Number of vectorized environments.
            num_transitions_per_env: Rollout horizon for each environment.
            obs_shape: Shape of one actor observation.
            privileged_obs_shape: Shape of one critic observation, if distinct.
            actions_shape: Shape of one action.
            device: Device on which to allocate storage tensors.
        """
        super().__init__(
            num_envs,
            num_transitions_per_env,
            obs_shape,
            privileged_obs_shape,
            actions_shape,
            device=device,
            n_samples_per_action=0,
        )
        self.action_latents = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.action_means = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.action_log_probs = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )

    def add_transitions(self, transition: Transition):
        """Copy a complete joint-policy transition into the next rollout slot.

        ``copy_`` keeps storage independent from the reusable transition
        object that an algorithm clears after every environment step.

        Args:
            transition: Transition containing core RL and joint-policy fields.

        Raises:
            OverflowError: The rollout buffer is already full.
            ValueError: A required transition field is missing.
        """
        if self.step >= self.num_transitions_per_env:
            raise OverflowError(
                "Rollout buffer overflow! You should call clear() before adding new transitions."
            )

        required_fields = (
            "observations",
            "actions",
            "rewards",
            "dones",
            "values",
            "action_latent",
            "action_mean",
            "action_log_prob",
        )
        missing_fields = [
            field for field in required_fields if getattr(transition, field) is None
        ]
        if (
            self.privileged_observations is not None
            and transition.privileged_observations is None
        ):
            missing_fields.append("privileged_observations")
        if missing_fields:
            raise ValueError(
                "Joint transition is missing required fields: "
                f"{', '.join(missing_fields)}"
            )

        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(
                transition.privileged_observations
            )
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)

        self.action_latents[self.step].copy_(transition.action_latent)
        self.action_means[self.step].copy_(transition.action_mean)
        self.action_log_probs[self.step].copy_(transition.action_log_prob)

        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8
    ) -> Generator[dict[str, torch.Tensor], None, None]:
        """Yield shuffled, complete joint-policy mini-batches.

        A fresh permutation is sampled for every epoch.  When the rollout
        batch size is not divisible by ``num_mini_batches``, the first batches
        receive one extra sample so every stored transition is yielded once.

        Args:
            num_mini_batches: Number of non-empty mini-batches per epoch.
            num_epochs: Number of independently shuffled passes over rollout.

        Yields:
            Dictionaries containing the core RL tensors and the replayable
            joint-policy fields.

        Raises:
            RuntimeError: The rollout is not full.
            ValueError: The mini-batch or epoch count is invalid.
        """
        if self.step != self.num_transitions_per_env:
            raise RuntimeError(
                "JointRolloutStorage requires a full rollout before generating "
                f"mini-batches; stored {self.step} of {self.num_transitions_per_env} steps."
            )
        if isinstance(num_mini_batches, bool) or not isinstance(num_mini_batches, int):
            raise TypeError("num_mini_batches must be a positive integer")
        if isinstance(num_epochs, bool) or not isinstance(num_epochs, int):
            raise TypeError("num_epochs must be a positive integer")
        if num_mini_batches <= 0 or num_epochs <= 0:
            raise ValueError("num_mini_batches and num_epochs must be positive")

        batch_size = self.num_envs * self.num_transitions_per_env
        if num_mini_batches > batch_size:
            raise ValueError(
                "num_mini_batches cannot exceed the complete rollout batch size; "
                f"got {num_mini_batches} for {batch_size} samples"
            )

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is None:
            critic_observations = observations
        else:
            critic_observations = self.privileged_observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        action_latents = self.action_latents.flatten(0, 1)
        action_means = self.action_means.flatten(0, 1)
        action_log_probs = self.action_log_probs.flatten(0, 1)

        base_mini_batch_size, num_larger_batches = divmod(batch_size, num_mini_batches)
        for _ in range(num_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            start = 0
            for mini_batch_index in range(num_mini_batches):
                mini_batch_size = base_mini_batch_size + (
                    mini_batch_index < num_larger_batches
                )
                stop = start + mini_batch_size
                batch_indices = indices[start:stop]
                start = stop

                yield {
                    "obs": observations[batch_indices],
                    "critic_obs": critic_observations[batch_indices],
                    "actions": actions[batch_indices],
                    "values": values[batch_indices],
                    "advantages": advantages[batch_indices],
                    "returns": returns[batch_indices],
                    "action_latents": action_latents[batch_indices],
                    "action_means": action_means[batch_indices],
                    "action_log_probs": action_log_probs[batch_indices],
                }

    def recurrent_mini_batch_generator(self, *args, **kwargs):
        """Reject recurrent use because the joint generator has no sequence API."""
        del args, kwargs
        raise NotImplementedError(
            "JointRolloutStorage currently supports feedforward policies only."
        )
