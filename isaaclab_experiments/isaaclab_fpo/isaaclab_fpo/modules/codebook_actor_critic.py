"""One-step pMF policy with a fixed, uniform finite latent prior."""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING

import torch

from .actor_critic import PMFActorCritic

if TYPE_CHECKING:
    from isaaclab_fpo.rl_cfg import FpoRslRlPpoActorCriticCfg


class CodebookPMFActorCritic(PMFActorCritic):
    """Sample a finite Gaussian mixture and evaluate its exact action density.

    Each component is the existing one-NFE transport evaluated at one fixed
    code vector. Codebook initialization uses an independent CPU generator,
    so it does not consume the actor/environment random stream.
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        cfg: FpoRslRlPpoActorCriticCfg,
    ):
        size, seed = cfg.codebook_size, cfg.codebook_seed
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("codebook_size must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("codebook_seed must be an integer in [0, 2**63)")
        super().__init__(num_actor_obs, num_critic_obs, num_actions, cfg)
        self.codebook_size = size
        self.codebook_seed = seed
        generator = torch.Generator(device="cpu").manual_seed(seed)
        codebook = torch.randn(size, num_actions, generator=generator)
        self.codebook_sha256 = hashlib.sha256(codebook.numpy().tobytes()).hexdigest()
        self.register_buffer("latent_codebook", codebook)
        self.register_buffer("_expected_codebook", codebook.clone(), persistent=False)
        # The legacy scalar is not in the base policy state_dict. Persist it
        # here so evaluation uses the checkpoint's actual annealed noise.
        self.register_buffer("codebook_noise_std", torch.tensor(self.action_perturb_std, dtype=torch.float64))

    def validate_codebook(self) -> None:
        """Reject mutation of the prior used to collect and replay actions."""
        if not torch.equal(self.latent_codebook, self._expected_codebook):
            raise ValueError("The fixed latent codebook has changed")

    def component_means(self, observations: torch.Tensor, actor: torch.nn.Module | None = None) -> torch.Tensor:
        """Return all public-action component means, with shape [batch, K, D]."""
        codes = self.latent_codebook.unsqueeze(0).expand(observations.shape[0], -1, -1)
        return self.transport_actions(observations, codes, actor=actor)

    def component_action_log_probs(self, actions: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
        """Return D-dimensional conditional densities with shape [batch, K]."""
        if means.ndim != 3 or means.shape[1:] != (self.codebook_size, self.num_actions):
            raise ValueError("means must have shape [batch, codebook_size, num_actions]")
        if actions.shape != (means.shape[0], self.num_actions):
            raise ValueError("actions must have shape [batch, num_actions]")
        expanded_actions = actions.unsqueeze(1).expand_as(means)
        return self.conditional_action_log_prob(expanded_actions, means).squeeze(-1)

    def marginal_action_log_prob(self, actions: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
        """Return the exact uniform mixture log density, with shape [batch, 1]."""
        components = self.component_action_log_probs(actions, means)
        return torch.logsumexp(components, dim=-1, keepdim=True) - math.log(self.codebook_size)

    def sample_codebook_transport(
        self, observations: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions, component indices, codes, means, and conditional log densities."""
        indices = torch.randint(
            self.codebook_size, (observations.shape[0],), device=observations.device, generator=generator
        )
        latent = self.latent_codebook[indices]
        mean = self.transport_actions(observations, latent)
        noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        actions = mean + self._conditional_action_std() * noise
        return actions, indices, latent, mean, self.conditional_action_log_prob(actions, mean)

    def sample_transport(self, observations: torch.Tensor):
        """Use the finite prior in the base policy's transport-sampling interface."""
        actions, _, latent, mean, log_prob = self.sample_codebook_transport(observations)
        return actions, latent, mean, log_prob

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample from the finite prior in the public stochastic-action interface."""
        return self.sample_codebook_transport(observations)[0]

    def act_inference(
        self, observations: torch.Tensor, eval_mode: str = "random", eval_fixed_seed: int | None = 12345
    ) -> torch.Tensor:
        """Evaluate the actual mixture, or its explicitly deterministic mean.

        ``random`` includes both discrete latent and conditional Gaussian
        noise. ``fixed_seed`` reinitializes a local generator per call.
        ``mean`` (and the compatibility alias ``zero``) averages all component
        means without noise; it is a deterministic reference, not a sample.
        """
        if eval_mode in ("mean", "zero"):
            return self.component_means(observations).mean(dim=1)
        if eval_mode == "random":
            return self.sample_codebook_transport(observations)[0]
        if eval_mode == "fixed_seed":
            seed = 12345 if eval_fixed_seed is None else eval_fixed_seed
            generator = torch.Generator(device=observations.device).manual_seed(seed)
            return self.sample_codebook_transport(observations, generator=generator)[0]
        raise ValueError(f"Unknown codebook evaluation mode: {eval_mode!r}")

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Validate and restore the saved codebook and conditional noise scale."""
        codebook = state_dict.get("latent_codebook")
        if codebook is None or not torch.equal(codebook.detach().cpu(), self._expected_codebook.detach().cpu()):
            raise ValueError("Codebook checkpoint latent prior mismatch")
        noise = state_dict.get("codebook_noise_std")
        if noise is None or noise.ndim != 0 or not math.isfinite(float(noise)) or float(noise) <= 0:
            raise ValueError("Codebook checkpoint has an invalid noise standard deviation")
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.action_perturb_std = float(self.codebook_noise_std)
        return result
