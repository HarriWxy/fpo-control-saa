"""Posterior and clipping diagnostics for uniform Gaussian codebook policies."""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def codebook_diagnostics(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_param: float,
) -> dict[str, torch.Tensor]:
    """Compare marginal clipping with posterior-averaged joint clipping.

    The inputs contain component action log densities evaluated at the same
    actions, sampled from the old uniform mixture. Component labels must refer
    to the same fixed codebook in both policies. The component log densities
    already sum over action dimensions.

    Posterior KL and the clipping Jensen gap are non-negative for each action
    in exact arithmetic. Small floating-point negatives are retained. The
    marginal log-ratio KL estimate can be negative on a finite batch; the k3
    estimate is non-negative in exact arithmetic. Both estimate old-to-new
    marginal KL in expectation over old-policy actions.

    Args:
        old_log_probs: Finite old component log densities, shape [batch, components].
        new_log_probs: Finite new component log densities, with the same shape,
            dtype, and device as ``old_log_probs``.
        advantages: Advantages used by the policy loss, shape [batch, 1].
        clip_param: PPO clipping parameter in (0, 1).

    Returns:
        Scalar diagnostic tensors, detached from the policy gradient graph.

    Raises:
        ValueError: Tensor shapes are incompatible or the clipping parameter is invalid.
    """
    if (
        old_log_probs.ndim != 2
        or old_log_probs.shape != new_log_probs.shape
        or old_log_probs.shape[0] == 0
        or old_log_probs.shape[1] == 0
    ):
        raise ValueError("Component log densities must have matching non-empty [batch, components] shapes")
    if advantages.shape != (old_log_probs.shape[0], 1):
        raise ValueError("Advantages must have shape [batch, 1]")
    if not math.isfinite(clip_param) or not 0.0 < clip_param < 1.0:
        raise ValueError("clip_param must be finite and in (0, 1)")

    old_log_total = torch.logsumexp(old_log_probs, dim=-1, keepdim=True)
    new_log_total = torch.logsumexp(new_log_probs, dim=-1, keepdim=True)
    old_log_posterior = torch.log_softmax(old_log_probs, dim=-1)
    new_log_posterior = torch.log_softmax(new_log_probs, dim=-1)
    old_posterior = old_log_posterior.exp()
    marginal_log_ratio = new_log_total - old_log_total
    marginal_ratio = marginal_log_ratio.exp()

    # w_old * r_joint = w_new * r_marginal. This avoids overflowing component
    # ratios in old tails and cancellation of large absolute log densities.
    weighted_joint_ratios = new_log_posterior.exp() * marginal_ratio
    joint_positive = torch.minimum(weighted_joint_ratios, (1.0 + clip_param) * old_posterior).sum(dim=-1, keepdim=True)
    joint_negative = torch.maximum(weighted_joint_ratios, (1.0 - clip_param) * old_posterior).sum(dim=-1, keepdim=True)
    joint_clipped = advantages * torch.where(advantages >= 0.0, joint_positive, joint_negative)
    marginal_clipped = advantages * torch.where(
        advantages >= 0.0,
        marginal_ratio.clamp(max=1.0 + clip_param),
        marginal_ratio.clamp(min=1.0 - clip_param),
    )

    return {
        "codebook/posterior_entropy": -(old_posterior * old_log_posterior).sum(dim=-1).mean(),
        "codebook/posterior_max_probability": old_posterior.max(dim=-1).values.mean(),
        "codebook/effective_components": old_posterior.square().sum(dim=-1).reciprocal().mean(),
        "codebook/posterior_kl": (old_posterior * (old_log_posterior - new_log_posterior)).sum(dim=-1).mean(),
        "codebook/clip_jensen_gap": (marginal_clipped - joint_clipped).mean(),
        "codebook/marginal_kl_k3": (torch.expm1(marginal_log_ratio) - marginal_log_ratio).mean(),
        "codebook/marginal_kl_mc": -marginal_log_ratio.mean(),
    }
