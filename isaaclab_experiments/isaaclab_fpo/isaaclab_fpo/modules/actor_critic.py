# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

from isaaclab_fpo.utils import resolve_nn_activation

if TYPE_CHECKING:
    from isaaclab_fpo.rl_cfg import FpoRslRlPpoActorCriticCfg


class ActorCritic(nn.Module):
    is_recurrent = False
    # Subclasses can add flow-time conditions or actor heads without changing
    # the default CFM actor checkpoint layout.
    actor_num_time_embeddings = 1
    actor_output_multiplier = 1

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        cfg: FpoRslRlPpoActorCriticCfg,
    ):
        super().__init__()
        activation = resolve_nn_activation(cfg.activation)

        # Policy parameters
        self.num_actor_obs = num_actor_obs
        self.num_actions = num_actions
        self.timestep_embed_dim = cfg.timestep_embed_dim
        self.mlp_output_scale = cfg.actor_mlp_output_scale
        self.cfm_loss_t_inverse_cdf_beta = cfg.cfm_loss_t_inverse_cdf_beta
        self.sampling_steps = cfg.sampling_steps
        self.cfm_loss_reduction = cfg.cfm_loss_reduction

        # Deterministic sampling data is runtime-only, preserving old checkpoint
        # keys. Integer frequencies also survive module dtype conversions exactly.
        self.register_buffer(
            "_timestep_freqs",
            2 ** torch.arange(self.timestep_embed_dim // 2),
            persistent=False,
        )
        for name in (
            "_flow_t_current",
            "_flow_dt",
            "_flow_embeddings",
            "_unit_time_embedding",
        ):
            self.register_buffer(name, torch.empty(0), persistent=False)
        self._flow_cache_key = None
        self._unit_time_cache_key = None

        # Inference parameters
        self.actor_scale = cfg.actor_scale

        # Training parameters
        self.action_perturb_std = cfg.action_perturb_std
        if cfg.training_sampling_steps is not None:
            self.training_sampling_steps = cfg.training_sampling_steps
        else:
            self.training_sampling_steps = cfg.sampling_steps

        # Policy Network: Actor
        actor_hidden_dims = cfg.actor_hidden_dims
        critic_hidden_dims = cfg.critic_hidden_dims
        mlp_input_dim_a = (
            num_actor_obs
            + self.actor_num_time_embeddings * self.timestep_embed_dim
            + num_actions
        )
        mlp_input_dim_c = num_critic_obs
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(
                    nn.Linear(
                        actor_hidden_dims[layer_index],
                        self.actor_output_multiplier * num_actions,
                    )
                )
            else:
                actor_layers.append(
                    nn.Linear(
                        actor_hidden_dims[layer_index],
                        actor_hidden_dims[layer_index + 1],
                    )
                )
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Apply scaling to actor's final layer weights
        if (
            cfg.actor_final_layer_weight_scale is not None
            and cfg.actor_final_layer_weight_scale != 1.0
        ):
            final_layer = self.actor[-1]
            assert isinstance(final_layer, nn.Linear), (
                "Expected final layer to be Linear"
            )
            with torch.no_grad():
                final_layer.weight.data *= cfg.actor_final_layer_weight_scale
                if final_layer.bias is not None:
                    final_layer.bias.data *= cfg.actor_final_layer_weight_scale
            print(
                f"Applied actor_final_layer_weight_scale={cfg.actor_final_layer_weight_scale} to final layer"
            )

        # Policy Network: Critic
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(
                    nn.Linear(
                        critic_hidden_dims[layer_index],
                        critic_hidden_dims[layer_index + 1],
                    )
                )
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Keep parameters on the original modules so compilation does not alter
        # checkpoint keys. CPU callers use the eager path without compilation.
        self._compiled_integrate_flow = torch.compile(
            self._integrate_flow, mode="reduce-overhead"
        )

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        # Recompute constants rather than reusing values rounded by, for example,
        # a float32 -> float16 -> float32 module conversion.
        self._flow_cache_key = None
        self._unit_time_cache_key = None
        return result

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def act(self, observations: torch.Tensor, **kwargs):
        device = observations.device

        assert len(observations.shape) == 2, (
            "observations should be of shape (batch_size, obs_dim)"
        )
        batch_size = observations.shape[0]

        if not self.training:
            x_t = torch.zeros(
                size=(batch_size, self.num_actions),
                device=device,
                dtype=observations.dtype,
            )
        else:
            x_t = torch.randn(
                size=(batch_size, self.num_actions),
                device=device,
                dtype=observations.dtype,
            )

        actions = self._sample_flow(observations, x_t)

        # Perturb action with random noise, this can be interpreted as an entropy regularizer
        if self.training and self.action_perturb_std > 0:
            noise = self.action_perturb_std * torch.randn_like(actions)
            actions = actions + noise

        return actions

    def get_cfm_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
        actor: torch.nn.Module | None = None,
    ):
        """Compute CFM loss for training.

        Returns:
            Tuple of (loss, x1_pred, x0_pred)  x1_pred and x0_pred are the predicted next and current states, respectively.
        """
        # Use provided actor or default to self.actor
        if actor is None:
            actor = self.actor

        (batch_dims, action_dim) = actions.shape
        assert len(observations.shape) == 2, (
            "observations should be of shape (batch_size, obs_dim)"
        )
        assert observations.shape[0] == batch_dims, (
            "actor_obs and actions should have the same batch size"
        )

        # Scale actions to match the scaled action space used during inference
        # During inference, we output self.actor_scale * x_t, so during training
        # we need to learn flow in the same scaled space
        scaled_actions = actions / self.actor_scale

        # Naive velocity MSE loss (hardcoded "u" mode)
        n_samples_per_action = eps.shape[1]
        assert eps.shape == (batch_dims, n_samples_per_action, action_dim)
        assert t.shape == (batch_dims, n_samples_per_action, 1)

        # Compute the embedded timestep
        embedded_t = self._embed_timestep(t)
        x_t = t * eps + (1.0 - t) * scaled_actions[:, None, :]
        # Broadcast actor_obs to match the batch shape
        actor_obs_expanded = observations[:, None, :].expand(
            batch_dims, n_samples_per_action, -1
        )
        # Handle flow network output parameterization (hardcoded to "u" mode)
        mlp_output = actor(torch.cat([actor_obs_expanded, embedded_t, x_t], dim=-1))
        mlp_output = self.mlp_output_scale * mlp_output  # Scale MLP output

        # Direct velocity prediction (u mode)
        velocity_pred = mlp_output
        x0_pred = x_t - t * velocity_pred
        x1_pred = x0_pred + velocity_pred

        # Target velocity is eps - scaled_actions (true flow velocity in scaled space)
        target_velocity = eps - scaled_actions[:, None, :]
        loss = self._compute_squared_error(velocity_pred, target_velocity)
        assert loss.shape == (batch_dims, n_samples_per_action)

        return loss, x1_pred, x0_pred

    def _embed_timestep(self, t: torch.Tensor) -> torch.Tensor:
        """Embed (*, 1) timestep into (*, timestep_embed_dim)."""
        assert t.shape[-1] == 1
        freqs = self._timestep_freqs.to(device=t.device)
        scaled_t = t * freqs
        out = torch.cat([torch.cos(scaled_t), torch.sin(scaled_t)], dim=-1)
        assert out.shape == (*t.shape[:-1], self.timestep_embed_dim)
        return out

    def _embed_flow_schedule(
        self, t_current: torch.Tensor, dt: torch.Tensor
    ) -> torch.Tensor:
        """Embed all fixed inference nodes once, before entering the flow loop."""
        return self._embed_timestep(t_current[:, None])

    def _get_flow_schedule(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cache the sampling grid and embeddings for this step count/device/dtype."""
        key = (self.sampling_steps, observations.device, observations.dtype)
        if self._flow_cache_key != key:
            # A cache first warmed during inference must remain usable by later
            # autograd calls, which cannot save inference tensors for backward.
            with torch.inference_mode(False), torch.no_grad():
                path = torch.linspace(
                    1.0,
                    0.0,
                    self.sampling_steps + 1,
                    device=observations.device,
                    dtype=observations.dtype,
                )
                self._flow_t_current = path[:-1]
                self._flow_dt = path[1:] - path[:-1]
                self._flow_embeddings = self._embed_flow_schedule(
                    self._flow_t_current, self._flow_dt
                )
            self._flow_cache_key = key
        return self._flow_t_current, self._flow_dt, self._flow_embeddings

    def _get_unit_time_embedding(self, reference: torch.Tensor) -> torch.Tensor:
        """Return the single h=1 embedding shared by every transport sample."""
        key = (reference.device, reference.dtype)
        if self._unit_time_cache_key != key:
            with torch.inference_mode(False), torch.no_grad():
                interval = torch.ones(
                    (1, 1), device=reference.device, dtype=reference.dtype
                )
                self._unit_time_embedding = self._embed_timestep(interval)
            self._unit_time_cache_key = key
        return self._unit_time_embedding

    def _sample_flow(
        self, observations: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Generate public actions from prescribed noise and cached flow nodes."""
        t_current, dt, embedded_steps = self._get_flow_schedule(observations)
        integrate = (
            self._compiled_integrate_flow
            if observations.is_cuda
            else self._integrate_flow
        )
        result = integrate(
            observations, noise, t_current, dt, self.sampling_steps, embedded_steps
        )
        return self.actor_scale * result

    def flow_step(
        self,
        observations: torch.Tensor,
        x_t: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Take one reverse flow step in the scaled action space.

        This small public interface is used by diagnostic tools instead of
        assuming a particular actor input layout.  For CFM, it is an Euler
        step of the instantaneous velocity field.
        """
        batch_size = observations.shape[0]
        assert observations.shape == (batch_size, self.num_actor_obs)
        assert x_t.shape == (batch_size, self.num_actions)
        assert r.shape == t.shape == (batch_size, 1)

        embedded_t = self._embed_timestep(t)
        velocity = self.actor(torch.cat([observations, embedded_t, x_t], dim=-1))
        velocity = self.mlp_output_scale * velocity
        return x_t + velocity * (r - t)

    def _integrate_flow(
        self,
        observations: torch.Tensor,
        x_t: torch.Tensor,
        t_current: torch.Tensor,
        dt: torch.Tensor,
        flow_steps: int,
        embedded_steps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inner flow integration loop extracted for torch.compile.

        This method contains only static-shape tensor operations and constant
        control flow (hardcoded to "u" mode velocity prediction),
        making it safe for CUDA graph capture via torch.compile(mode="reduce-overhead").

        Args:
            observations: (batch_size, obs_dim) observation tensor.
            x_t: (batch_size, num_actions) initial noise / sample.
            t_current: (flow_steps,) current timestep values.
            dt: (flow_steps,) timestep deltas.
            flow_steps: Number of integration steps (must be constant across calls).
            embedded_steps: Optional cached embeddings at the sampling nodes.

        Returns:
            x_t: (batch_size, num_actions) integrated sample (denoised actions).
        """
        batch_size = observations.shape[0]
        if embedded_steps is None:
            embedded_steps = self._embed_flow_schedule(t_current, dt)

        for i in range(flow_steps):
            embedded_t = embedded_steps[i].expand(batch_size, -1)

            # Forward through actor network
            mlp_output = self.actor(torch.cat([observations, embedded_t, x_t], dim=-1))
            mlp_output = self.mlp_output_scale * mlp_output

            # Compute velocity from network output (hardcoded to "u" mode)
            u = mlp_output
            x_t = x_t + u * dt[i]

        return x_t

    def _compute_squared_error(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute squared error with configurable reduction."""
        if self.cfm_loss_reduction == "mean":
            return torch.mean((predictions - targets) ** 2, dim=-1)
        elif self.cfm_loss_reduction == "sum":
            return torch.sum((predictions - targets) ** 2, dim=-1)
        else:  # "sqrt"
            squared_errors = (predictions - targets) ** 2
            return torch.sum(squared_errors, dim=-1) / (
                squared_errors.shape[-1] ** 0.5
            )

    def act_inference(self, observations, eval_mode="zero", eval_fixed_seed=12345):
        """Inference with configurable deterministic sampling for flow matching.

        Args:
            observations: Input observations
            eval_mode: Sampling strategy for initial noise
                - "zero": Use zeros for initial noise
                - "fixed_seed": Use fixed seed for reproducible noise
                - "random": Use random noise (different each time)
            eval_fixed_seed: Random seed for fixed_seed mode

        Returns:
            Actions tensor
        """
        device = observations.device
        assert len(observations.shape) == 2, (
            "observations should be of shape (batch_size, obs_dim)"
        )
        batch_size = observations.shape[0]

        # Initialize x_t based on eval_mode
        if eval_mode == "zero":
            x_t = torch.zeros(
                size=(batch_size, self.num_actions),
                device=device,
                dtype=observations.dtype,
            )
        elif eval_mode == "fixed_seed":
            generator = torch.Generator(device=device)
            generator.manual_seed(eval_fixed_seed)
            x_t = torch.randn(
                size=(batch_size, self.num_actions),
                device=device,
                dtype=observations.dtype,
                generator=generator,
            )
        elif eval_mode == "random":
            x_t = torch.randn(
                size=(batch_size, self.num_actions),
                device=device,
                dtype=observations.dtype,
            )
        else:
            raise ValueError(f"Unknown eval_mode: {eval_mode}")

        return self._sample_flow(observations, x_t)

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


class IMFActorCritic(ActorCritic):
    """Conditional improved-MeanFlow actor with the original FPO critic.

    The actor predicts an average velocity ``u(s, z_t, r, t)`` over the
    interval ``[r, t]`` and an auxiliary instantaneous velocity
    ``v(s, z_t, t)``.  The latter is evaluated at the boundary ``r=t`` so it
    does not acquire an accidental dependence on the interval length.

    All flow quantities use the same *scaled action space* as ``ActorCritic``:
    action scaling happens only at the public ``act``/``act_inference``
    boundary.  This preserves the FPO action contract.
    """

    actor_num_time_embeddings = 2
    actor_output_multiplier = 2
    is_improved_mean_flow = True

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        cfg: FpoRslRlPpoActorCriticCfg,
    ):
        super().__init__(num_actor_obs, num_critic_obs, num_actions, cfg)

        self.imf_aux_v_loss_coef = cfg.imf_aux_v_loss_coef
        self.imf_adaptive_gradient_norm_p = cfg.imf_adaptive_gradient_norm_p
        self.imf_adaptive_gradient_norm_eps = cfg.imf_adaptive_gradient_norm_eps
        self.imf_logit_normal_mean = cfg.imf_logit_normal_mean
        self.imf_logit_normal_std = cfg.imf_logit_normal_std
        self.imf_fm_proportion = cfg.imf_fm_proportion

        if self.imf_aux_v_loss_coef < 0:
            raise ValueError("imf_aux_v_loss_coef must be non-negative")
        if self.imf_adaptive_gradient_norm_p < 0:
            raise ValueError("imf_adaptive_gradient_norm_p must be non-negative")
        if self.imf_adaptive_gradient_norm_eps <= 0:
            raise ValueError("imf_adaptive_gradient_norm_eps must be positive")
        if self.imf_logit_normal_std <= 0:
            raise ValueError("imf_logit_normal_std must be positive")
        if not 0.0 <= self.imf_fm_proportion <= 1.0:
            raise ValueError("imf_fm_proportion must be in [0, 1]")

    def _predict_u_and_v(
        self,
        observations: torch.Tensor,
        x_t: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        actor: torch.nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict mean and instantaneous velocities for flattened samples.

        Args:
            observations: ``[batch, obs_dim]`` fixed policy conditions.
            x_t: ``[batch, action_dim]`` point on the noise--action path.
            r, t: ``[batch, 1]`` interval endpoints with ``r <= t``.
            actor: Optional actor module, primarily useful for evaluation.
        """
        if actor is None:
            actor = self.actor

        batch_size = observations.shape[0]
        assert observations.shape == (batch_size, self.num_actor_obs)
        assert x_t.shape == (batch_size, self.num_actions)
        assert r.shape == t.shape == (batch_size, 1)

        embedded_r = self._embed_timestep(r)
        embedded_t = self._embed_timestep(t)
        output = actor(torch.cat([observations, embedded_r, embedded_t, x_t], dim=-1))
        output = self.mlp_output_scale * output
        mean_velocity, instantaneous_velocity = output.split(self.num_actions, dim=-1)
        assert mean_velocity.shape == instantaneous_velocity.shape == x_t.shape
        return mean_velocity, instantaneous_velocity

    def _with_imf_gradient_normalization(self, loss: torch.Tensor) -> torch.Tensor:
        """Keep raw iMF scores while optionally normalizing their gradients.

        The official iMF adaptive loss has a nearly constant forward value for
        exponent one.  That cannot be inserted directly into FPO's
        ``exp(old_score - new_score)`` surrogate.  This straight-through form
        retains the unmodified residual as the score while applying the
        requested iMF-style normalization only to its gradient.
        """
        if self.imf_adaptive_gradient_norm_p == 0.0:
            return loss

        denominator = (
            loss.detach() + self.imf_adaptive_gradient_norm_eps
        ).pow(self.imf_adaptive_gradient_norm_p)
        normalized_loss = loss / denominator
        return loss.detach() + normalized_loss - normalized_loss.detach()

    def get_imf_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        eps: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        actor: torch.nn.Module | None = None,
        return_components: bool = False,
    ):
        """Compute the improved-MeanFlow score used by experimental IMF-FPO.

        For ``z_t = t * eps + (1-t) * action``, iMF trains the compound
        instantaneous field

        ``V = u_theta(z_t, r, t) + (t-r) * stopgrad(JVP(u; (v_theta, 0, 1)))``

        against ``eps - action``.  ``v_theta`` is the auxiliary head evaluated
        at ``r=t``.  Both the iMF residual and the auxiliary velocity residual
        are returned as a per-Monte-Carlo-sample score.  The caller must store
        and replay the same ``eps``, ``r``, and ``t`` when forming FPO ratios.

        This is an experimental FPO score surrogate: unlike the original CFM
        score, iMF's JVP residual has no established likelihood-ratio/ELBO
        derivation.
        """
        if actor is None:
            actor = self.actor

        batch_size, action_dim = actions.shape
        assert observations.shape == (batch_size, self.num_actor_obs)
        assert action_dim == self.num_actions

        n_samples_per_action = eps.shape[1]
        expected_noise_shape = (batch_size, n_samples_per_action, action_dim)
        expected_time_shape = (batch_size, n_samples_per_action, 1)
        assert eps.shape == expected_noise_shape
        assert r.shape == t.shape == expected_time_shape

        scaled_actions = actions / self.actor_scale
        x_t = t * eps + (1.0 - t) * scaled_actions[:, None, :]
        target_velocity = eps - scaled_actions[:, None, :]

        # Work on a flat batch and treat observations as a fixed condition, so
        # the JVP is only along (z_t, r, t).
        flat_size = batch_size * n_samples_per_action
        flat_observations = (
            observations[:, None, :]
            .expand(batch_size, n_samples_per_action, -1)
            .reshape(flat_size, self.num_actor_obs)
        )
        flat_x_t = x_t.reshape(flat_size, action_dim)
        flat_r = r.reshape(flat_size, 1)
        flat_t = t.reshape(flat_size, 1)
        flat_target = target_velocity.reshape(flat_size, action_dim)

        # v_theta(z_t, t) is evaluated on the h=t-r=0 boundary.  It drives
        # the JVP tangent and gets its own direct regression loss.
        _, instantaneous_velocity = self._predict_u_and_v(
            flat_observations, flat_x_t, flat_t, flat_t, actor=actor
        )
        mean_velocity, _ = self._predict_u_and_v(
            flat_observations, flat_x_t, flat_r, flat_t, actor=actor
        )

        def mean_velocity_fn(
            z_input: torch.Tensor, r_input: torch.Tensor, t_input: torch.Tensor
        ) -> torch.Tensor:
            mean_velocity, _ = self._predict_u_and_v(
                flat_observations, z_input, r_input, t_input, actor=actor
            )
            return mean_velocity

        # iMF explicitly stop-grads this derivative.  The autograd.functional
        # implementation is deliberately used with create_graph=False: it
        # supports the project's standard MLP activations on current PyTorch
        # builds while avoiding a second-order parameter graph.
        _, mean_velocity_time_derivative = torch.autograd.functional.jvp(
            mean_velocity_fn,
            (flat_x_t, flat_r, flat_t),
            (
                instantaneous_velocity.detach(),
                torch.zeros_like(flat_r),
                torch.ones_like(flat_t),
            ),
            create_graph=False,
        )
        compound_velocity = mean_velocity + (flat_t - flat_r) * (
            mean_velocity_time_derivative.detach()
        )

        imf_u_loss = self._compute_squared_error(compound_velocity, flat_target)
        imf_v_loss = self._compute_squared_error(
            instantaneous_velocity, flat_target
        )
        score = self._with_imf_gradient_normalization(imf_u_loss)
        score = score + self.imf_aux_v_loss_coef * self._with_imf_gradient_normalization(
            imf_v_loss
        )

        # Reuse the old FPO diagnostics interface with the local auxiliary-v
        # endpoint estimates.  They are proxies, not a mean-flow KL.
        x0_pred = flat_x_t - flat_t * instantaneous_velocity
        x1_pred = x0_pred + instantaneous_velocity

        score = score.reshape(batch_size, n_samples_per_action)
        x0_pred = x0_pred.reshape(batch_size, n_samples_per_action, action_dim)
        x1_pred = x1_pred.reshape(batch_size, n_samples_per_action, action_dim)
        assert score.shape == (batch_size, n_samples_per_action)

        if return_components:
            components = {
                "u_loss": imf_u_loss.reshape(batch_size, n_samples_per_action),
                "v_loss": imf_v_loss.reshape(batch_size, n_samples_per_action),
                "interval": (flat_t - flat_r).reshape(
                    batch_size, n_samples_per_action
                ),
                "jvp_norm": mean_velocity_time_derivative.detach()
                .norm(dim=-1)
                .reshape(batch_size, n_samples_per_action),
            }
            return score, x1_pred, x0_pred, components
        return score, x1_pred, x0_pred

    def flow_step(
        self,
        observations: torch.Tensor,
        x_t: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Take one iMF average-velocity jump in the scaled action space."""
        mean_velocity, _ = self._predict_u_and_v(observations, x_t, r, t)
        return x_t - (t - r) * mean_velocity

    def _embed_flow_schedule(
        self, t_current: torch.Tensor, dt: torch.Tensor
    ) -> torch.Tensor:
        """Cache iMF's ordered pair of endpoint embeddings."""
        return torch.cat(
            [
                self._embed_timestep((t_current + dt)[:, None]),
                self._embed_timestep(t_current[:, None]),
            ],
            dim=-1,
        )

    def _integrate_flow(
        self,
        observations: torch.Tensor,
        x_t: torch.Tensor,
        t_current: torch.Tensor,
        dt: torch.Tensor,
        flow_steps: int,
        embedded_steps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate mean-flow jumps from noise time one to action time zero."""
        batch_size = observations.shape[0]
        if embedded_steps is None:
            embedded_steps = self._embed_flow_schedule(t_current, dt)

        for i in range(flow_steps):
            embedded_rt = embedded_steps[i].expand(batch_size, -1)
            output = self.actor(torch.cat([observations, embedded_rt, x_t], dim=-1))
            mean_velocity = self.mlp_output_scale * output[:, : self.num_actions]

            # dt = r - t is negative during reverse generation, hence this is
            # exactly x_r = x_t - (t-r) * u_theta(x_t, r, t).
            x_t = x_t + mean_velocity * dt[i]

        return x_t


class PMFActorCritic(ActorCritic):
    """Pixel Mean Flow actor with the original FPO value critic.

    pMF changes the prediction space compared with iMF: the actor directly
    predicts two denoised-action-like endpoints (one for ``u`` and one for the
    auxiliary boundary ``v``).  They are converted to velocity space through
    ``u=(z_t-x_u)/max(t, eps)`` and ``v=(z_t-x_v)/max(t, eps)`` before the iMF
    compound JVP residual is formed.  The actor is conditioned on ``h=t-r``
    only, matching the pMF implementation; the explicit ``t`` dependence is
    retained in the x-to-velocity conversion.
    """

    actor_num_time_embeddings = 1
    actor_output_multiplier = 2
    is_pixel_mean_flow = True

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        cfg: FpoRslRlPpoActorCriticCfg,
    ):
        super().__init__(num_actor_obs, num_critic_obs, num_actions, cfg)

        self.pmf_logit_normal_mean = cfg.pmf_logit_normal_mean
        self.pmf_logit_normal_std = cfg.pmf_logit_normal_std
        self.pmf_fm_proportion = cfg.pmf_fm_proportion
        self.pmf_aux_v_loss_coef = cfg.pmf_aux_v_loss_coef
        self.pmf_time_eps = cfg.pmf_time_eps
        self.pmf_adaptive_gradient_norm_p = cfg.pmf_adaptive_gradient_norm_p
        self.pmf_adaptive_gradient_norm_eps = cfg.pmf_adaptive_gradient_norm_eps

        if self.pmf_logit_normal_std <= 0:
            raise ValueError("pmf_logit_normal_std must be positive")
        if not 0.0 <= self.pmf_fm_proportion <= 1.0:
            raise ValueError("pmf_fm_proportion must be in [0, 1]")
        if self.pmf_aux_v_loss_coef < 0:
            raise ValueError("pmf_aux_v_loss_coef must be non-negative")
        if self.pmf_time_eps <= 0:
            raise ValueError("pmf_time_eps must be positive")
        if self.pmf_adaptive_gradient_norm_p < 0:
            raise ValueError("pmf_adaptive_gradient_norm_p must be non-negative")
        if self.pmf_adaptive_gradient_norm_eps <= 0:
            raise ValueError("pmf_adaptive_gradient_norm_eps must be positive")

        # Joint training retains several map outputs until a shared backward,
        # and rollout retains means for replay. Disable CUDA-graph buffer reuse
        # on this path so later forwards cannot overwrite those live tensors.
        self._compiled_transport = torch.compile(
            self._transport_core,
            dynamic=True,
            options={"triton.cudagraphs": False},
        )

    def _predict_x_heads(
        self,
        observations: torch.Tensor,
        x_t: torch.Tensor,
        h: torch.Tensor,
        actor: torch.nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict pMF's mean-flow and instantaneous denoised-action fields."""
        if actor is None:
            actor = self.actor

        batch_size = observations.shape[0]
        assert observations.shape == (batch_size, self.num_actor_obs)
        assert x_t.shape == (batch_size, self.num_actions)
        assert h.shape == (batch_size, 1)

        embedded_h = self._embed_timestep(h)
        output = actor(torch.cat([observations, embedded_h, x_t], dim=-1))
        output = self.mlp_output_scale * output
        x_mean, x_instantaneous = output.split(self.num_actions, dim=-1)
        assert x_mean.shape == x_instantaneous.shape == x_t.shape
        return x_mean, x_instantaneous

    def transport_actions(
        self,
        observations: torch.Tensor,
        noise: torch.Tensor,
        actor: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        """Evaluate the one-NFE pMF transport map with prescribed noise.

        The returned tensor is in the public (scaled) action coordinates used
        by the environment.  Supplying the same ``noise`` to two actors thus
        gives the explicit coupling used by FSPPO's map-space trust region.

        Args:
            observations: Policy observations with shape ``[batch, obs_dim]``.
            noise: Base samples with shape ``[batch, action_dim]`` or
                ``[batch, samples, action_dim]``.
            actor: Optional actor module, used to evaluate a frozen old policy.

        Returns:
            Transported actions with the same leading dimensions as ``noise``.
        """
        batch_size = observations.shape[0]
        assert observations.shape == (batch_size, self.num_actor_obs)
        if noise.ndim not in (2, 3):
            raise ValueError(
                "noise must have shape [batch, action_dim] or "
                f"[batch, samples, action_dim], got {tuple(noise.shape)}"
            )
        if noise.shape[0] != batch_size or noise.shape[-1] != self.num_actions:
            raise ValueError(
                "noise batch/action dimensions must match observations and policy; "
                f"got observations={tuple(observations.shape)}, noise={tuple(noise.shape)}"
            )

        # A pMF jump from t=1 to r=0 has h=t-r=1 and returns the x-mean head
        # exactly.  Apply actor_scale here so D_map is measured in the same
        # coordinates as the action delivered by ``act``/``act_inference``.
        embedding = self._get_unit_time_embedding(noise)
        samples = noise[:, None, :] if noise.ndim == 2 else noise
        transport = (
            self._compiled_transport if observations.is_cuda else self._transport_core
        )
        transported = transport(
            observations,
            samples,
            embedding,
            self.actor if actor is None else actor,
            self.mlp_output_scale,
            self.actor_scale,
        )
        return transported.reshape(*noise.shape[:-1], self.num_actions)

    @staticmethod
    def _transport_core(
        observations: torch.Tensor,
        noise: torch.Tensor,
        embedding: torch.Tensor,
        actor: nn.Module,
        mlp_output_scale: float,
        actor_scale: float,
    ) -> torch.Tensor:
        """Evaluate a batch of transport samples without copying expanded obs. actor(obs,embedding,noise)"""
        batch_size, num_samples, num_actions = noise.shape
        inputs = torch.cat(
            [
                observations[:, None, :].expand(batch_size, num_samples, -1),
                embedding[None, :, :].expand(batch_size, num_samples, -1),
                noise,
            ],
            dim=-1,
        )
        # Flatten only after cat materializes the inputs, keeping the actor's
        # original 2-D GEMM layout and avoiding a separate repeated-obs tensor.
        output = actor(inputs.reshape(batch_size * num_samples, inputs.shape[-1]))
        x_mean = mlp_output_scale * output[:, :num_actions]
        return actor_scale * x_mean

    def _sample_flow(
        self, observations: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Use the exact transport map for a single unclamped pMF jump."""
        if self.sampling_steps == 1 and self.pmf_time_eps <= 1.0:
            return self.transport_actions(observations, noise)
        return super()._sample_flow(observations, noise)

    def sample_transport(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action from the explicit pMF transport distribution.

        This interface represents the rollout distribution as a joint sample
        of a standard-normal transport latent and a fixed-variance conditional
        Gaussian action for the current rollout. The latent prior is
        intentionally not part of the returned log-probability: it cancels
        when old and new policies are evaluated at the stored latent.

        Args:
            observations: Policy observations with shape ``[batch, obs_dim]``.

        Returns:
            A tuple containing the sampled actions ``[batch, action_dim]``,
            transport latents ``[batch, action_dim]``, conditional means
            ``[batch, action_dim]``, and conditional action log-probabilities
            ``[batch, 1]``.
        """
        if observations.ndim != 2 or observations.shape[1] != self.num_actor_obs:
            raise ValueError(
                "observations must have shape [batch, "
                f"{self.num_actor_obs}], got {tuple(observations.shape)}"
            )

        batch_size = observations.shape[0]
        latent = torch.randn(
            (batch_size, self.num_actions),
            device=observations.device,
            dtype=observations.dtype,
        )
        mean = self.transport_actions(observations, latent)
        action_std = self._conditional_action_std()
        actions = mean + action_std * torch.randn_like(mean)
        log_prob = self.conditional_action_log_prob(actions, mean)
        return actions, latent, mean, log_prob

    def conditional_action_log_prob(
        self, actions: torch.Tensor, means: torch.Tensor
    ) -> torch.Tensor:
        """Compute the fixed-variance Gaussian conditional action density.

        The returned density is ``log p(actions | means)`` only.  It does not
        include the standard-normal transport-latent prior, allowing callers
        to form an old/new joint ratio with the same stored latent.

        Args:
            actions: Actions with shape ``[..., action_dim]``.
            means: Conditional transport means with shape
                ``[..., action_dim]``.

        Returns:
            Conditional log-probabilities summed over action dimensions with
            shape ``[..., 1]``.
        """
        if actions.shape != means.shape:
            raise ValueError(
                "actions and means must have identical shapes; got "
                f"{tuple(actions.shape)} and {tuple(means.shape)}"
            )
        if actions.ndim < 1 or actions.shape[-1] != self.num_actions:
            raise ValueError(
                "actions and means must end with action_dim="
                f"{self.num_actions}; got {tuple(actions.shape)}"
            )

        action_std = self._conditional_action_std()
        standardized_residual = (actions - means) / action_std
        log_normalizer = math.log(2.0 * math.pi * action_std**2)
        return -0.5 * (
            standardized_residual.square() + log_normalizer
        ).sum(dim=-1, keepdim=True)

    def _conditional_action_std(self) -> float:
        """Return the finite, positive action-noise standard deviation."""
        action_std = float(self.action_perturb_std)
        if not math.isfinite(action_std) or action_std <= 0.0:
            raise ValueError(
                "The joint transport interface requires a finite, positive "
                "action_perturb_std."
            )
        return action_std

    def _predict_u_and_v(
        self,
        observations: torch.Tensor,
        x_t: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        actor: torch.nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert pMF x predictions into mean and instantaneous velocities."""
        x_mean, x_instantaneous = self._predict_x_heads(
            observations, x_t, t - r, actor=actor
        )
        denominator = torch.clamp(t, min=self.pmf_time_eps)
        mean_velocity = (x_t - x_mean) / denominator
        instantaneous_velocity = (x_t - x_instantaneous) / denominator
        return mean_velocity, instantaneous_velocity

    def _with_pmf_gradient_normalization(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply pMF-style gradient normalization without changing FPO scores."""
        if self.pmf_adaptive_gradient_norm_p == 0.0:
            return loss

        denominator = (
            loss.detach() + self.pmf_adaptive_gradient_norm_eps
        ).pow(self.pmf_adaptive_gradient_norm_p)
        normalized_loss = loss / denominator
        # Straight-through estimator: raw loss in the forward ratio, normalized
        # gradient during the optimizer update.
        return loss.detach() + normalized_loss - normalized_loss.detach()

    def get_pmf_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        eps: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        actor: torch.nn.Module | None = None,
        return_components: bool = False,
    ):
        """Compute pMF's x-prediction-to-v-space FPO score.

        The network output is ``x`` rather than velocity.  The pMF conversion
        is applied before constructing

        ``V = u + (t-r) * stopgrad(JVP(u; (v, 0, 1)))``.

        As in the official implementation, the instantaneous auxiliary head is
        trained at the boundary ``r=t``.  The low-time clamp is also used when
        forming the conditional target, avoiding an unstable division near
        ``t=0``.  The combined u/v residual is an experimental FPO score.
        """
        if actor is None:
            actor = self.actor

        batch_size, action_dim = actions.shape
        assert observations.shape == (batch_size, self.num_actor_obs)
        assert action_dim == self.num_actions

        n_samples_per_action = eps.shape[1]
        expected_noise_shape = (batch_size, n_samples_per_action, action_dim)
        expected_time_shape = (batch_size, n_samples_per_action, 1)
        assert eps.shape == expected_noise_shape
        assert r.shape == t.shape == expected_time_shape

        scaled_actions = actions / self.actor_scale
        x_t = t * eps + (1.0 - t) * scaled_actions[:, None, :]
        # pMF's x-to-v conversion uses the same lower clamp as its network;
        # above the clamp this is exactly the CFM target eps - scaled_action.
        target_velocity = (x_t - scaled_actions[:, None, :]) / torch.clamp(
            t, min=self.pmf_time_eps
        )

        flat_size = batch_size * n_samples_per_action
        flat_observations = (
            observations[:, None, :]
            .expand(batch_size, n_samples_per_action, -1)
            .reshape(flat_size, self.num_actor_obs)
        )
        flat_x_t = x_t.reshape(flat_size, action_dim)
        flat_r = r.reshape(flat_size, 1)
        flat_t = t.reshape(flat_size, 1)
        flat_target = target_velocity.reshape(flat_size, action_dim)

        # v_theta is the auxiliary instantaneous field evaluated at h=0.
        _, instantaneous_velocity = self._predict_u_and_v(
            flat_observations, flat_x_t, flat_t, flat_t, actor=actor
        )
        mean_velocity, _ = self._predict_u_and_v(
            flat_observations, flat_x_t, flat_r, flat_t, actor=actor
        )

        def mean_velocity_fn(
            z_input: torch.Tensor, r_input: torch.Tensor, t_input: torch.Tensor
        ) -> torch.Tensor:
            mean_velocity, _ = self._predict_u_and_v(
                flat_observations, z_input, r_input, t_input, actor=actor
            )
            return mean_velocity

        # The pMF/iMF identity differentiates along the trajectory tangent v,
        # with r held fixed and t increasing.  Stop-gradient is applied to the
        # JVP output, so this update does not create a second-order parameter
        # graph.
        _, mean_velocity_time_derivative = torch.autograd.functional.jvp(
            mean_velocity_fn,
            (flat_x_t, flat_r, flat_t),
            (
                instantaneous_velocity.detach(),
                torch.zeros_like(flat_r),
                torch.ones_like(flat_t),
            ),
            create_graph=False,
        )
        compound_velocity = mean_velocity + (flat_t - flat_r) * (
            mean_velocity_time_derivative.detach()
        )

        pmf_u_loss = self._compute_squared_error(compound_velocity, flat_target)
        pmf_v_loss = self._compute_squared_error(
            instantaneous_velocity, flat_target
        )
        score = self._with_pmf_gradient_normalization(pmf_u_loss)
        score = score + self.pmf_aux_v_loss_coef * self._with_pmf_gradient_normalization(
            pmf_v_loss
        )

        # Keep the legacy diagnostics/storage contract.  IMFFPO/PMFFPO disable
        # the old KL/kNN proxies, so these endpoint values are informational.
        x0_pred = flat_x_t - flat_t * instantaneous_velocity
        x1_pred = x0_pred + instantaneous_velocity

        score = score.reshape(batch_size, n_samples_per_action)
        x0_pred = x0_pred.reshape(batch_size, n_samples_per_action, action_dim)
        x1_pred = x1_pred.reshape(batch_size, n_samples_per_action, action_dim)
        assert score.shape == (batch_size, n_samples_per_action)

        if return_components:
            components = {
                "u_loss": pmf_u_loss.reshape(batch_size, n_samples_per_action),
                "v_loss": pmf_v_loss.reshape(batch_size, n_samples_per_action),
                "interval": (flat_t - flat_r).reshape(
                    batch_size, n_samples_per_action
                ),
                "jvp_norm": mean_velocity_time_derivative.detach()
                .norm(dim=-1)
                .reshape(batch_size, n_samples_per_action),
            }
            return score, x1_pred, x0_pred, components
        return score, x1_pred, x0_pred

    def get_pmf_v_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
        actor: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        """Compute per-sample instantaneous-velocity regression losses.

        This avoids the mean-flow JVP when only the pMF v-head loss is needed.

        Args:
            observations: Policy observations with shape [batch, obs_dim].
            actions: Public actions with shape [batch, action_dim].
            eps: Noise samples with shape [batch, samples, action_dim].
            t: Upper time endpoints with shape [batch, samples, 1].
            actor: Optional actor module, primarily useful for evaluation.

        Returns:
            Per-action, per-noise-sample squared errors with shape [batch, samples].
        """
        if actor is None:
            actor = self.actor

        batch_size, action_dim = actions.shape
        assert observations.shape == (batch_size, self.num_actor_obs)
        assert action_dim == self.num_actions
        n_samples_per_action = eps.shape[1]
        expected_noise_shape = (batch_size, n_samples_per_action, action_dim)
        expected_time_shape = (batch_size, n_samples_per_action, 1)
        assert eps.shape == expected_noise_shape
        assert t.shape == expected_time_shape

        scaled_actions = actions / self.actor_scale
        x_t = t * eps + (1.0 - t) * scaled_actions[:, None, :]
        target_velocity = (x_t - scaled_actions[:, None, :]) / torch.clamp(
            t, min=self.pmf_time_eps
        )

        flat_size = batch_size * n_samples_per_action
        flat_observations = (
            observations[:, None, :]
            .expand(batch_size, n_samples_per_action, -1)
            .reshape(flat_size, self.num_actor_obs)
        )
        flat_x_t = x_t.reshape(flat_size, action_dim)
        flat_t = t.reshape(flat_size, 1)
        flat_target = target_velocity.reshape(flat_size, action_dim)
        _, instantaneous_velocity = self._predict_u_and_v(
            flat_observations, flat_x_t, flat_t, flat_t, actor=actor
        )
        loss = self._compute_squared_error(instantaneous_velocity, flat_target)
        return loss.reshape(batch_size, n_samples_per_action)

    def flow_step(
        self,
        observations: torch.Tensor,
        x_t: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Take one pMF mean-flow jump in the scaled action space."""
        mean_velocity, _ = self._predict_u_and_v(observations, x_t, r, t)
        return x_t - (t - r) * mean_velocity

    def _integrate_flow(
        self,
        observations: torch.Tensor,
        x_t: torch.Tensor,
        t_current: torch.Tensor,
        dt: torch.Tensor,
        flow_steps: int,
        embedded_steps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate pMF x-derived mean velocities from t=1 to t=0."""
        batch_size = observations.shape[0]
        if embedded_steps is None:
            embedded_steps = self._embed_flow_schedule(t_current, dt)

        for i in range(flow_steps):
            t_value = t_current[i].reshape(1, 1)
            embedded_h = embedded_steps[i].expand(batch_size, -1)
            output = self.actor(torch.cat([observations, embedded_h, x_t], dim=-1))
            output = self.mlp_output_scale * output
            x_mean = output[:, : self.num_actions]
            denominator = torch.clamp(t_value, min=self.pmf_time_eps)
            mean_velocity = (x_t - x_mean) / denominator
            x_t = x_t + mean_velocity * dt[i]

        return x_t

    def _embed_flow_schedule(
        self, t_current: torch.Tensor, dt: torch.Tensor
    ) -> torch.Tensor:
        """Cache pMF's interval embeddings, where h=t-r=-dt."""
        return self._embed_timestep(-dt[:, None])
