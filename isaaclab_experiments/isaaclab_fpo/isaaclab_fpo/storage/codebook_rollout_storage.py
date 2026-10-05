"""Finite-latent rollout storage with cached frozen behavior densities."""

from __future__ import annotations

import torch

from .joint_rollout_storage import JointRolloutStorage


class CodebookRolloutStorage(JointRolloutStorage):
    """Keep selected conditional densities separate from full mixture densities."""

    class Transition(JointRolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            self.component_index = None

    def __init__(self, *args, codebook_size: int, **kwargs):
        super().__init__(*args, **kwargs)
        shape = (self.num_transitions_per_env, self.num_envs)
        self.component_indices = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.old_component_log_probs = torch.zeros(*shape, codebook_size, device=self.device)
        self.component_log_probs_ready = False

    def add_transitions(self, transition: Transition) -> None:
        if transition.component_index is None:
            raise ValueError("Codebook transition is missing its component index")
        step = self.step
        super().add_transitions(transition)
        self.component_indices[step].copy_(transition.component_index)

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        if not self.component_log_probs_ready:
            raise RuntimeError("Behavior component densities must be cached before updating")
        indices = self.component_indices.flatten(0, 1)
        log_probs = self.old_component_log_probs.flatten(0, 1)
        for batch in super().mini_batch_generator(num_mini_batches, num_epochs):
            selection = batch["transition_indices"]
            batch["component_indices"] = indices[selection]
            batch["old_component_log_probs"] = log_probs[selection]
            yield batch

    def clear(self) -> None:
        super().clear()
        self.component_log_probs_ready = False
