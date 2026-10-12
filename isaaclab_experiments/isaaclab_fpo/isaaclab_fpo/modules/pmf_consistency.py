# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Paired, offline pMF endpoint diagnostics in public action coordinates.

The primary reference solves ``dz/dt = u(z,t,t,s)`` backwards from the same
noise used by the one-step map. It uses the *mean* x head with ``h=t-r=0``;
ordinary multi-step pMF inference instead uses finite-interval mean transport.
The auxiliary x head is a separate reference because it drives the training
JVP, and agreement of that head alone does not establish diagonal consistency.

No field is evaluated at zero time. Integration stops at ``time_eps`` and one
last Euler extrapolation, ``z_0 = z_eps - eps*v(z_eps,eps)``, completes the
endpoint. This is a documented numerical approximation, not a MeanFlow jump.
Refinement measures discretization error at a fixed cutoff; it does not remove
the cutoff/boundary approximation or establish distribution matching.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .actor_critic import PMFActorCritic


_FIELDS = ("diagonal_mean", "auxiliary")


@contextmanager
def _evaluation_mode(policy):
    """Preserve each module's mode, including independently frozen modules."""
    modes = [(module, module.training) for module in policy.modules()]
    policy.eval()
    try:
        with torch.inference_mode():
            yield
    finally:
        for module, training in modes:
            module.training = training


def _validate_inputs(policy, observations, noise, batch_size, time_eps):
    if not getattr(policy, "is_pixel_mean_flow", False):
        raise TypeError("Consistency diagnostics require a pMF actor.")
    if observations.ndim != 2 or observations.shape[1] != policy.num_actor_obs:
        raise ValueError(
            "normalized_observations must have shape [states, actor_obs_dim]."
        )
    if noise.ndim not in (2, 3) or noise.shape[0] != observations.shape[0]:
        raise ValueError(
            "initial_noise must have shape [states, actions] or [states, samples, actions]."
        )
    if noise.shape[-1] != policy.num_actions or noise.numel() == 0:
        raise ValueError("initial_noise must contain nonempty policy action samples.")
    if noise.device != observations.device or noise.dtype != observations.dtype:
        raise ValueError(
            "Observations and noise must have the same device and floating dtype."
        )
    if not observations.is_floating_point():
        raise ValueError("Observations and noise must use a floating dtype.")
    if not torch.isfinite(observations).all() or not torch.isfinite(noise).all():
        raise ValueError("Input observations and prescribed noise must be finite.")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")
    policy_eps = float(policy.pmf_time_eps)
    if not math.isfinite(policy_eps) or not 0.0 < policy_eps < 1.0:
        raise ValueError("policy.pmf_time_eps must be finite and in (0, 1).")
    cutoff = policy_eps if time_eps is None else float(time_eps)
    if not math.isfinite(cutoff) or not 0.0 < cutoff < 1.0 or cutoff < policy_eps:
        raise ValueError("time_eps must be in (0, 1) and at least policy.pmf_time_eps.")
    if not math.isfinite(float(policy.actor_scale)) or policy.actor_scale <= 0:
        raise ValueError("actor_scale must be finite and positive.")
    return cutoff


def _validate_solver(ode_steps, solver, field):
    if isinstance(ode_steps, bool) or not isinstance(ode_steps, int) or ode_steps <= 0:
        raise ValueError("ode_steps must contain positive integers.")
    if solver not in ("euler", "heun"):
        raise ValueError("solver must be 'euler' or 'heun'.")
    if field not in _FIELDS:
        raise ValueError(
            f"Unknown reference field {field!r}; expected one of {_FIELDS}."
        )


def _paired_chunks(observations, noise, batch_size):
    samples = noise.shape[1] if noise.ndim == 3 else 1
    flat_noise = noise.reshape(-1, noise.shape[-1])
    for start in range(0, flat_noise.shape[0], batch_size):
        stop = min(start + batch_size, flat_noise.shape[0])
        indices = torch.arange(start, stop, device=noise.device) // samples
        yield indices, observations[indices], flat_noise[start:stop]


def _integrate_chunk(
    policy, observations, initial_noise, ode_steps, solver, cutoff, field
):
    # Geometric times bound the relative dt/t near the singular x-to-v conversion.
    times = torch.exp(
        torch.linspace(
            0.0,
            math.log(cutoff),
            ode_steps + 1,
            device=initial_noise.device,
            dtype=initial_noise.dtype,
        )
    )
    times[0], times[-1] = 1.0, cutoff
    zero_interval = initial_noise.new_zeros((initial_noise.shape[0], 1))
    head = 0 if field == "diagonal_mean" else 1

    def velocity(state, time):
        x_head = policy._predict_x_heads(observations, state, zero_interval)[head]
        return (state - x_head) / time

    state = initial_noise.clone()
    for index in range(ode_steps):
        time, next_time = times[index], times[index + 1]
        dt = next_time - time  # Negative: integrate from noise at t=1 towards actions.
        first_velocity = velocity(state, time)
        if solver == "euler":
            state = state + dt * first_velocity
        else:
            predicted = state + dt * first_velocity
            second_velocity = velocity(predicted, next_time)
            state = state + 0.5 * dt * (first_velocity + second_velocity)
    state = state - times[-1] * velocity(state, times[-1])
    return policy.actor_scale * state


def integrate_pmf_diagonal(
    policy: PMFActorCritic,
    normalized_observations: torch.Tensor,
    initial_noise: torch.Tensor,
    *,
    ode_steps: int = 128,
    solver: str = "heun",
    time_eps: float | None = None,
    batch_size: int = 1024,
    field: str = "diagonal_mean",
) -> torch.Tensor:
    """Return prescribed-noise ODE endpoints, shaped like noise and actor-scaled.

    Inputs are already-normalized policy observations; this function neither
    updates normalizers nor samples noise. ``field='auxiliary'`` explicitly
    selects the second x head instead of the primary diagonal mean field.
    No action clipping occurs during integration or on the returned endpoints.
    NFE is ``steps+1`` for Euler or ``2*steps+1`` for Heun, including the tail.
    """
    cutoff = _validate_inputs(
        policy, normalized_observations, initial_noise, batch_size, time_eps
    )
    _validate_solver(ode_steps, solver, field)
    endpoints = []
    with _evaluation_mode(policy):
        for _, observations, noise in _paired_chunks(
            normalized_observations, initial_noise, batch_size
        ):
            endpoints.append(
                _integrate_chunk(
                    policy, observations, noise, ode_steps, solver, cutoff, field
                )
            )
        return torch.cat(endpoints).reshape(initial_noise.shape)


def _finite_number(value):
    number = float(value)
    return number if math.isfinite(number) else None


class _ErrorAccumulator:
    def __init__(self):
        self.pairs = self.valid = self.elements = 0
        self.squared_l2 = self.l2 = self.max_l2 = 0.0

    def update(self, first, second, clip=None):
        self.pairs += first.shape[0]
        valid = torch.isfinite(first).all(dim=-1) & torch.isfinite(second).all(dim=-1)
        first, second = first[valid].double(), second[valid].double()
        if clip is not None:
            first, second = first.clamp(-clip, clip), second.clamp(-clip, clip)
        squared = (first - second).square().sum(dim=-1)
        if squared.numel():
            self.valid += squared.numel()
            self.elements += first.numel()
            self.squared_l2 += squared.sum().item()
            self.l2 += squared.sqrt().sum().item()
            self.max_l2 = max(self.max_l2, squared.sqrt().max().item())

    def result(self):
        mse = self.squared_l2 / self.elements if self.elements else float("nan")
        return {
            "valid_pairs": self.valid,
            "finite_pair_fraction": self.valid / self.pairs,
            "mse": _finite_number(mse),
            "mean_squared_l2": _finite_number(self.squared_l2 / self.valid)
            if self.valid
            else None,
            "rmse": _finite_number(math.sqrt(mse)),
            "mean_l2": _finite_number(self.l2 / self.valid) if self.valid else None,
            "max_l2": _finite_number(self.max_l2) if self.valid else None,
        }


class _EndpointAccumulator:
    def __init__(self, clip):
        self.clip = clip
        self.pairs = self.valid_pairs = self.elements = self.valid_elements = (
            self.clipped_elements
        ) = 0
        self.max_abs = 0.0

    def update(self, endpoints):
        finite = torch.isfinite(endpoints)
        values = endpoints[finite].double()
        self.pairs += endpoints.shape[0]
        self.elements += endpoints.numel()
        self.valid_pairs += finite.all(dim=-1).sum().item()
        self.valid_elements += values.numel()
        if values.numel():
            self.max_abs = max(self.max_abs, values.abs().max().item())
            if self.clip is not None:
                self.clipped_elements += (values.abs() > self.clip).sum().item()

    def result(self):
        return {
            "finite_pair_fraction": self.valid_pairs / self.pairs,
            "finite_element_fraction": self.valid_elements / self.elements,
            "max_abs": _finite_number(self.max_abs) if self.valid_elements else None,
            "clipped_element_fraction": (self.clipped_elements / self.valid_elements)
            if self.clip is not None and self.valid_elements
            else None,
        }


def evaluate_pmf_consistency(
    policy: PMFActorCritic,
    normalized_observations: torch.Tensor,
    initial_noise: torch.Tensor,
    *,
    ode_steps: Sequence[int] = (8, 32, 128),
    solver: str = "heun",
    time_eps: float | None = None,
    batch_size: int = 1024,
    action_clip: float | None = None,
    fields: Sequence[str] = _FIELDS,
) -> dict:
    """Compare a paired one-step map with independent, refined ODE references.

    Accept ``observations[S,O]`` and explicit ``noise[S,K,A]`` (or ``[S,A]``).
    Actor calls are bounded by ``batch_size`` and statistics are accumulated
    without retaining the whole endpoint bank. NaN/Inf endpoints remain visible
    through finite fractions; errors use only pairs with both endpoints finite.
    Clipping is applied only to final, already-scaled diagnostic endpoints.
    Conditional variance across prescribed noise is action diversity, not proof
    of multimodality. The global RNG and all policy training modes are preserved.
    """
    cutoff = _validate_inputs(
        policy, normalized_observations, initial_noise, batch_size, time_eps
    )
    requested_steps = tuple(ode_steps)
    if not requested_steps or not fields or len(set(fields)) != len(fields):
        raise ValueError(
            "Supply nonempty unique reference fields and at least one ODE resolution."
        )
    for step in requested_steps:
        for field in fields:
            _validate_solver(step, solver, field)
    steps = sorted(set(requested_steps))
    if action_clip is not None and (not math.isfinite(action_clip) or action_clip <= 0):
        raise ValueError("action_clip must be finite and positive.")

    one_stats = _EndpointAccumulator(action_clip)
    accumulators = {
        (field, step): (
            _EndpointAccumulator(action_clip),
            _ErrorAccumulator(),
            _ErrorAccumulator(),
            _ErrorAccumulator(),
            _ErrorAccumulator(),
        )
        for field in fields
        for step in steps
    }
    states = normalized_observations.shape[0]
    samples = initial_noise.shape[1] if initial_noise.ndim == 3 else 1
    # State-indexed sufficient statistics avoid storing S*K endpoint tensors.
    noise_sum = torch.zeros((states, policy.num_actions), dtype=torch.float64)
    noise_squared_sum = torch.zeros_like(noise_sum)
    clipped_noise_sum = torch.zeros_like(noise_sum) if action_clip is not None else None
    clipped_noise_squared_sum = (
        torch.zeros_like(noise_sum) if action_clip is not None else None
    )
    noise_counts = torch.zeros(states, dtype=torch.float64)
    with _evaluation_mode(policy):
        for indices, observations, noise in _paired_chunks(
            normalized_observations, initial_noise, batch_size
        ):
            one = policy.transport_actions(observations, noise)
            one_stats.update(one)
            finite = torch.isfinite(one).all(dim=-1)
            valid_indices, valid_one = indices[finite].cpu(), one[finite].double().cpu()
            noise_sum.index_add_(0, valid_indices, valid_one)
            noise_squared_sum.index_add_(0, valid_indices, valid_one.square())
            if action_clip is not None:
                clipped_one = valid_one.clamp(-action_clip, action_clip)
                clipped_noise_sum.index_add_(0, valid_indices, clipped_one)
                clipped_noise_squared_sum.index_add_(
                    0, valid_indices, clipped_one.square()
                )
            noise_counts.index_add_(
                0, valid_indices, torch.ones(valid_indices.numel(), dtype=torch.float64)
            )
            for field in fields:
                endpoints = {
                    step: _integrate_chunk(
                        policy, observations, noise, step, solver, cutoff, field
                    )
                    for step in steps
                }
                reference = endpoints[steps[-1]]
                for step, endpoint in endpoints.items():
                    stats, error, clipped_error, convergence, clipped_convergence = (
                        accumulators[field, step]
                    )
                    stats.update(endpoint)
                    error.update(one, endpoint)
                    convergence.update(endpoint, reference)
                    if action_clip is not None:
                        clipped_error.update(one, endpoint, action_clip)
                        clipped_convergence.update(endpoint, reference, action_clip)

    valid_states = noise_counts >= 2
    variance = None
    clipped_variance = None
    if valid_states.any():
        counts = noise_counts[valid_states, None]
        variances = (
            noise_squared_sum[valid_states] / counts
            - (noise_sum[valid_states] / counts).square()
        ).clamp_min(0.0)
        variance = _finite_number(variances.mean().item())
        if action_clip is not None:
            clipped_variances = (
                clipped_noise_squared_sum[valid_states] / counts
                - (clipped_noise_sum[valid_states] / counts).square()
            ).clamp_min(0.0)
            clipped_variance = _finite_number(clipped_variances.mean().item())
    one_result = one_stats.result()
    one_result.update(
        noise_conditional_variance=variance,
        clipped_noise_conditional_variance=clipped_variance,
        noise_conditional_variance_valid_states=int(valid_states.sum().item()),
    )
    result = {
        "diagnostic": "pmf_one_step_ode_consistency",
        "states": states,
        "noise_per_state": samples,
        "pairs": states * samples,
        "actor_scale": float(policy.actor_scale),
        "action_clip": action_clip,
        "solver": solver,
        "time_grid": "geometric",
        "time_eps": cutoff,
        "boundary_correction": "Euler: z_0 = z_eps - eps*v(z_eps,eps)",
        "primary_field": "diagonal_mean" if "diagonal_mean" in fields else fields[0],
        "one_step_nfe": 1,
        "reference_ode_steps": steps[-1],
        "one_step": one_result,
        "fields": {},
    }
    for field in fields:
        resolutions = []
        for step in steps:
            stats, error, clipped_error, convergence, clipped_convergence = (
                accumulators[field, step]
            )
            resolutions.append(
                {
                    "ode_steps": step,
                    "nfe": step * (2 if solver == "heun" else 1) + 1,
                    "endpoint": stats.result(),
                    "one_step_error": error.result(),
                    "clipped_one_step_error": clipped_error.result()
                    if action_clip is not None
                    else None,
                    "reference_error": convergence.result(),
                    "clipped_reference_error": clipped_convergence.result()
                    if action_clip is not None
                    else None,
                }
            )
        result["fields"][field] = {
            "definition": "(z - x_mean(s,z,h=0))/t"
            if field == "diagonal_mean"
            else "(z - x_instantaneous(s,z,h=0))/t",
            "resolutions": resolutions,
        }
    return result
