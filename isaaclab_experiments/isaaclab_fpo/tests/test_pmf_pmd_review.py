"""Independent integration audit of frozen PMD targets and projection noise."""

from __future__ import annotations

import copy
from itertools import pairwise

import torch
from test_pmf_pmd import (
    collect_bandit,
    cpu_execution,  # noqa: F401 -- shared autouse CPU fixture
    make_algorithm,
)


def test_projection_keeps_improvement_targets_and_q_frozen_but_draws_new_flow_noise(
    monkeypatch,
):
    """Actor projection cannot refit Q, regenerate targets or replay candidate latents.

    Trace real production forwards during a complete update. No replacement
    scoring function supplies the expected behavior: Q fitting and every pMF
    JVP/backward still execute. Multiple actor steps make a moving target or
    accidentally retained candidate noise observable.
    """
    algorithm = make_algorithm(
        pmd_projection_epochs=3, pmd_batch_size=7, n_samples_per_action=2
    )
    collect_bandit(algorithm, count=64)
    frozen = {}
    projected = []
    candidate_latents = []
    projection_noise = []
    real_transport = algorithm.policy.transport_actions
    real_freeze = algorithm._freeze_targets
    real_projection = algorithm._projection_loss
    real_flow_loss = algorithm.policy.get_pmf_loss

    def observe_transport(observations, noise, actor=None):
        if actor is algorithm._old_actor:
            candidate_latents.append(noise.detach().clone())
        return real_transport(observations, noise, actor=actor)

    def observe_freeze(observations, critic_observations):
        result = real_freeze(observations, critic_observations)
        frozen["candidates"] = result[0].clone()
        frozen["weights"] = result[2].clone()
        frozen["q"] = copy.deepcopy(algorithm.q.state_dict())
        frozen["value"] = copy.deepcopy(algorithm.policy.critic.state_dict())
        frozen["q_gradients"] = [
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in algorithm.q.parameters()
        ]
        assert not result[0].requires_grad and not result[2].requires_grad
        return result

    def observe_projection(observations, candidates, weights):
        # Trace microbatches, then compare the complete paired target/weight
        # multiset for each epoch independently of their shuffled partitions.
        pairs = torch.cat((candidates.flatten(1), weights), dim=-1)
        projected.append(pairs.detach().clone())
        return real_projection(observations, candidates, weights)

    def observe_flow_loss(observations, targets, noise, r, t, **kwargs):
        projection_noise.append(noise.detach().clone())
        return real_flow_loss(observations, targets, noise, r, t, **kwargs)

    monkeypatch.setattr(algorithm.policy, "transport_actions", observe_transport)
    monkeypatch.setattr(algorithm, "_freeze_targets", observe_freeze)
    monkeypatch.setattr(algorithm, "_projection_loss", observe_projection)
    monkeypatch.setattr(algorithm.policy, "get_pmf_loss", observe_flow_loss)
    result = algorithm.update()
    assert result["metrics"]["pmd/accepted_projection_steps"] == 6
    assert result["metrics"]["pmd/actor_changed"] == 1
    old_latents = torch.cat(candidate_latents).reshape(-1)
    assert old_latents.numel() == 64 * algorithm.num_candidates
    expected_pairs = torch.cat(
        (frozen["candidates"].flatten(1), frozen["weights"]), dim=-1
    )
    expected_pairs = expected_pairs[expected_pairs[:, 0].argsort()]
    epoch_parts, epoch_size, completed_epochs = [], 0, 0
    for pairs in projected:
        epoch_parts.append(pairs)
        epoch_size += pairs.shape[0]
        if epoch_size == 64:
            complete_epoch = torch.cat(epoch_parts)
            assert torch.equal(
                complete_epoch[complete_epoch[:, 0].argsort()], expected_pairs
            )
            completed_epochs += 1
            epoch_parts, epoch_size = [], 0
    assert completed_epochs == 3 and epoch_size == 0
    assert len(projection_noise) == len(projected)
    for before, after in pairwise(projection_noise):
        assert not torch.equal(before, after)
    # Each target gets independent flow noise; the generating candidate latent
    # is never used as a regression start, including the first projection.
    for pairs, noise in zip(projected, projection_noise):
        assert noise.shape == (pairs.shape[0] * algorithm.num_candidates, 2, 1)
        assert not torch.isin(noise[:, 0].flatten(), old_latents).all()
    for name, value in algorithm.q.state_dict().items():
        assert torch.equal(value, frozen["q"][name])
    for name, value in algorithm.policy.critic.state_dict().items():
        assert torch.equal(value, frozen["value"][name])
    for before, parameter in zip(frozen["q_gradients"], algorithm.q.parameters()):
        assert (
            (parameter.grad is None)
            if before is None
            else torch.equal(before, parameter.grad)
        )
