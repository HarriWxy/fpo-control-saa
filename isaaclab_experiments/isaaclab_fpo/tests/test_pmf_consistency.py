"""Independent analytic oracles for paired pMF numerical diagnostics."""

from __future__ import annotations

import json
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab")

from isaaclab_fpo.modules import PMFActorCritic
from isaaclab_fpo.modules.pmf_consistency import (
    evaluate_pmf_consistency,
    integrate_pmf_diagonal,
)
from isaaclab_fpo.rl_cfg import FpoRslRlPpoActorCriticCfg


@pytest.fixture(autouse=True)
def eager_cpu(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, *args, **kwargs: fn)
    with torch.random.fork_rng(devices=[]):
        yield


class AnalyticXActor(torch.nn.Module):
    """Known field dz/dt=beta*(z-mu)/t; the auxiliary field is zero.

    Its exact trajectory is mu+(noise-mu)*t**beta. The exact endpoint after
    the specified Euler boundary correction is therefore
    mu+(1-beta)*(noise-mu)*cutoff**beta. These formulas give an oracle
    independent of the numerical integrator, including its integration sign.
    """

    def __init__(self, beta, one_step_coefficient, output_scale, fail_negative=False):
        super().__init__()
        self.beta = beta
        self.one_step_coefficient = one_step_coefficient
        self.output_scale = output_scale
        self.fail_negative = fail_negative

    def forward(self, inputs):
        mu, state = inputs[:, :2], inputs[:, -2:]
        sin_h = inputs[:, 3:4]  # actual actor's [cos(h), sin(h)] embedding
        coefficient = (
            1.0
            - self.beta
            + (self.one_step_coefficient - (1.0 - self.beta)) * sin_h / math.sin(1.0)
        )
        mean_x = mu + coefficient * (state - mu)
        output = torch.cat((mean_x, state), dim=-1) / self.output_scale
        if self.fail_negative:
            output = torch.where(
                mu[:, :1] < 0, torch.full_like(output, float("nan")), output
            )
        return output


def make_policy(beta=1.0, coefficient=0.0, fail_negative=False):
    cfg = FpoRslRlPpoActorCriticCfg(
        actor_hidden_dims=[4],
        critic_hidden_dims=[4],
        activation="elu",
        timestep_embed_dim=2,
        actor_scale=0.7,
        actor_mlp_output_scale=1.8,
        action_perturb_std=0.0,
        sampling_steps=1,
        pmf_time_eps=0.04,
    )
    policy = PMFActorCritic(2, 2, 2, cfg).double()
    policy.actor = AnalyticXActor(
        beta, coefficient, cfg.actor_mlp_output_scale, fail_negative
    )
    return policy


def test_diagonal_reference_uses_mean_head_and_preserves_scale_rng_and_modes():
    """The mean diagonal contracts to mu; the auxiliary zero field stays noise."""
    policy = make_policy()
    policy.train()
    policy.critic.eval()  # restoring only the root mode would change this mode
    observations = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float64)
    noise = torch.tensor(
        [[[0.0, 4.0], [2.0, -2.0]], [[1.0, 3.0], [-3.0, 1.5]]], dtype=torch.float64
    )
    previous_rng = torch.get_rng_state().clone()
    report = evaluate_pmf_consistency(
        policy, observations, noise, ode_steps=(4, 16), batch_size=3
    )

    assert torch.equal(previous_rng, torch.get_rng_state())
    assert policy.training and policy.actor.training and not policy.critic.training
    assert report["one_step"]["noise_conditional_variance"] == pytest.approx(
        0.0, abs=1e-14
    )
    primary = report["fields"]["diagonal_mean"]["resolutions"][-1]
    auxiliary = report["fields"]["auxiliary"]["resolutions"][-1]
    assert primary["one_step_error"]["mse"] < 1e-28
    # Independent closed-form predictions: primary=scale*mu, auxiliary=scale*noise.
    expected_l2 = (
        (0.7**2 * (noise - observations[:, None]).square().sum(dim=-1)).mean().item()
    )
    assert auxiliary["one_step_error"]["mean_squared_l2"] == pytest.approx(expected_l2)
    assert primary["nfe"] == 33  # sixteen Heun intervals plus final field evaluation
    endpoints = integrate_pmf_diagonal(
        policy, observations, noise, ode_steps=16, batch_size=3
    )
    torch.testing.assert_close(
        endpoints, 0.7 * observations[:, None].expand_as(noise), atol=1e-14, rtol=0
    )


@pytest.mark.parametrize("solver,tolerance", [("euler", 0.003), ("heun", 1e-5)])
def test_refinement_matches_analytic_tail_and_keeps_state_noise_pairs(
    solver, tolerance
):
    """Both numerical solvers converge to the specified cutoff/tail oracle."""
    policy = make_policy(beta=0.5, coefficient=2.0)
    observations = torch.tensor([[1.0, 3.0], [-2.0, 0.5]], dtype=torch.float64)
    noise = torch.tensor(
        [[[0.0, 2.0], [2.0, 4.0], [4.0, -1.0]], [[-2.0, 1.0], [1.0, -2.0], [6.0, 5.0]]],
        dtype=torch.float64,
    )
    # Analytic trajectory and the independently prescribed linear tail.
    expected = 0.7 * (observations[:, None] + 0.1 * (noise - observations[:, None]))
    one_step = 0.7 * (observations[:, None] + 2.0 * (noise - observations[:, None]))
    report = evaluate_pmf_consistency(
        policy,
        observations,
        noise,
        ode_steps=(8, 64, 512),
        solver=solver,
        batch_size=2,
        action_clip=1.0,
        fields=("diagonal_mean",),
    )
    resolutions = report["fields"]["diagonal_mean"]["resolutions"]
    assert (
        resolutions[0]["reference_error"]["mse"]
        > resolutions[1]["reference_error"]["mse"]
        > 0
    )
    assert resolutions[-1]["reference_error"]["mse"] == 0.0
    endpoint = integrate_pmf_diagonal(
        policy, observations, noise, ode_steps=512, solver=solver, batch_size=2
    )
    torch.testing.assert_close(endpoint, expected, atol=tolerance, rtol=0)
    expected_mse = (one_step - expected).square().mean().item()
    expected_clipped_mse = (
        (one_step.clamp(-1, 1) - expected.clamp(-1, 1)).square().mean().item()
    )
    assert resolutions[-1]["one_step_error"]["mse"] == pytest.approx(
        expected_mse, rel=0.002
    )
    assert resolutions[-1]["clipped_one_step_error"]["mse"] == pytest.approx(
        expected_clipped_mse, rel=0.002
    )
    assert (
        resolutions[-1]["clipped_one_step_error"]["mse"]
        < resolutions[-1]["one_step_error"]["mse"]
    )
    expected_variance = one_step.var(dim=1, correction=0).mean().item()
    assert report["one_step"]["noise_conditional_variance"] == pytest.approx(
        expected_variance
    )
    expected_clipped_variance = (
        one_step.clamp(-1, 1).var(dim=1, correction=0).mean().item()
    )
    assert report["one_step"]["clipped_noise_conditional_variance"] == pytest.approx(
        expected_clipped_variance
    )
    assert report["one_step"]["noise_conditional_variance_valid_states"] == 2
    # A chunk crossing the state boundary must preserve exactly the same coupling.
    larger_chunks = integrate_pmf_diagonal(
        policy, observations, noise, ode_steps=512, solver=solver, batch_size=5
    )
    torch.testing.assert_close(endpoint, larger_chunks, atol=1e-14, rtol=0)


def test_invalid_endpoints_are_visible_and_report_remains_strict_json():
    """Invalid policy samples cannot disappear into clipped or aggregate errors."""
    policy = make_policy(fail_negative=True)
    observations = torch.tensor([[1.0, 2.0], [-1.0, 3.0]], dtype=torch.float64)
    noise = torch.zeros((2, 2), dtype=torch.float64)
    report = evaluate_pmf_consistency(
        policy, observations, noise, ode_steps=(2,), action_clip=1.0
    )
    assert report["one_step"]["finite_pair_fraction"] == 0.5
    assert report["one_step"]["noise_conditional_variance"] is None
    errors = report["fields"]["diagonal_mean"]["resolutions"][0]
    assert errors["one_step_error"]["valid_pairs"] == 1
    assert errors["clipped_one_step_error"]["finite_pair_fraction"] == 0.5
    json.dumps(report, allow_nan=False)
    all_invalid = evaluate_pmf_consistency(
        policy, observations[1:], noise[1:], ode_steps=(2,)
    )
    assert (
        all_invalid["fields"]["diagonal_mean"]["resolutions"][0]["one_step_error"][
            "mse"
        ]
        is None
    )
    json.dumps(all_invalid, allow_nan=False)

    with pytest.raises(ValueError, match="at least policy.pmf_time_eps"):
        evaluate_pmf_consistency(policy, observations, noise, time_eps=0.001)
    with pytest.raises(ValueError, match="must be finite"):
        evaluate_pmf_consistency(policy, observations * float("nan"), noise)
