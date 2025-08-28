from typing import NamedTuple, Tuple
import pickle
import numpy as np

import jax, jax.numpy as jnp
import optax
import haiku as hk

from relax.algorithm.base import Algorithm
from relax.network.sac_v import SACNet_V, SACParams
from relax.utils.experience import Experience
from relax.utils.typing import Metric


class SACOptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState
    encoder: optax.OptState


class SACTrainState(NamedTuple):
    params: SACParams
    opt_state: SACOptStates


class SAC_V(Algorithm):
    def __init__(self, agent: SACNet_V, params: SACParams, *, gamma: float = 0.99, lr: float = 1e-4,
                 alpha_lr: float = 3e-4, tau: float = 0.005, reward_scale: float = 0.2,):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.optim = optax.adam(lr)
        self.log_alpha_optim = optax.adam(alpha_lr)
        self.encoder_optim = optax.adam(lr)
        self.reward_scale = reward_scale

        self.state = SACTrainState(
            params=params,
            opt_state=SACOptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                policy=self.optim.init(params.policy),
                log_alpha=self.log_alpha_optim.init(params.log_alpha),
                encoder=self.encoder_optim.init(params.encoder)
            ),
        )

        @jax.jit
        def stateless_update(
            key: jax.Array, state: SACTrainState, data: Experience
        ) -> Tuple[SACTrainState, Metric]:
            obs, action, reward, next_obs, discount = data
            action = np.squeeze(action)
            reward = np.squeeze(reward)
            discount = np.squeeze(discount)
            q1_params, q2_params, target_q1_params, target_q2_params, policy_params, log_alpha, encoder_params = state.params
            q1_opt_state, q2_opt_state, policy_opt_state, log_alpha_opt_state, encoder_opt_state = state.opt_state
            next_eval_key, new_eval_key = jax.random.split(key)

            reward *= self.reward_scale
            next_obs = jax.lax.stop_gradient(self.agent.encoder(encoder_params, next_obs))

            # compute target q
            next_action, next_logp = self.agent.evaluate(next_eval_key, policy_params, next_obs)
            q1_target = self.agent.q(target_q1_params, next_obs, next_action)
            q2_target = self.agent.q(target_q2_params, next_obs, next_action)
            q_target = jnp.minimum(q1_target, q2_target) - jnp.exp(log_alpha) * next_logp
            q_backup = reward + discount * q_target

            def q_loss_fn(q1_params: hk.Params, q2_params: hk.Params, encoder_params: hk.Params) -> jax.Array:
                obs_latent = self.agent.encoder(encoder_params, obs)
                q1 = self.agent.q(q1_params, obs_latent, action)
                q1_loss = jnp.mean((q1 - q_backup) ** 2)
                q2 = self.agent.q(q2_params, obs_latent, action)
                q2_loss = jnp.mean((q2 - q_backup) ** 2)
                q_loss = q1_loss + q2_loss
                return q_loss, (q1_loss, q2_loss, q1, q2, obs_latent)
            
            (q_loss, (q1_loss, q2_loss, q1, q2, obs_latent)), (q1_grads, q2_grads, encoder_grads) = jax.value_and_grad(q_loss_fn, argnums=(0, 1, 2), has_aux=True)(q1_params, q2_params, encoder_params)
            obs_latent = jax.lax.stop_gradient(obs_latent)
            q1_update, q1_opt_state = self.optim.update(q1_grads, q1_opt_state)
            q2_update, q2_opt_state = self.optim.update(q2_grads, q2_opt_state)
            encoder_update, encoder_opt_state = self.optim.update(encoder_grads, encoder_opt_state)
            q1_params = optax.apply_updates(q1_params, q1_update)
            q2_params = optax.apply_updates(q2_params, q2_update)
            encoder_params = optax.apply_updates(encoder_params, encoder_update)

            # update policy
            def policy_loss_fn(policy_params: hk.Params) -> jax.Array:
                new_action, new_logp = self.agent.evaluate(new_eval_key, policy_params, obs_latent)
                q1 = self.agent.q(q1_params, obs_latent, new_action)
                q2 = self.agent.q(q2_params, obs_latent, new_action)
                q = jnp.minimum(q1, q2)
                policy_loss = jnp.mean(jnp.exp(log_alpha) * new_logp - q)
                return policy_loss, (q1, q2, new_logp)

            (policy_loss, aux), policy_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_params)
            q1, q2, new_logp = aux
            policy_update, policy_opt_state = self.optim.update(policy_grads, policy_opt_state)
            policy_params = optax.apply_updates(policy_params, policy_update)

            # update alpha
            def log_alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
                log_alpha_loss = -jnp.mean(log_alpha * (new_logp + self.agent.target_entropy))
                return log_alpha_loss

            log_alpha_grads = jax.grad(log_alpha_loss_fn)(log_alpha)
            log_alpha_update, log_alpha_opt_state = self.log_alpha_optim.update(log_alpha_grads, log_alpha_opt_state)
            log_alpha = optax.apply_updates(log_alpha, log_alpha_update)

            # update target q
            target_q1_params = optax.incremental_update(q1_params, target_q1_params, self.tau)
            target_q2_params = optax.incremental_update(q2_params, target_q2_params, self.tau)

            state = SACTrainState(
                params=SACParams(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, log_alpha, encoder_params),
                opt_state=SACOptStates(q1_opt_state, q2_opt_state, policy_opt_state, log_alpha_opt_state, encoder_opt_state),
            )
            info = {
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
                "q1": jnp.mean(q1),
                "q2": jnp.mean(q2),
                "policy_loss": policy_loss,
                "entropy": -jnp.mean(new_logp),
                "alpha": jnp.exp(log_alpha),
            }
            return state, info

        self._implement_common_behavior(stateless_update, self.agent.get_action, self.agent.get_deterministic_action)

    def get_policy_params(self):
        return (self.state.params.policy, self.state.params.encoder)

    def get_policy_params_to_save(self):
        return (self.state.params.policy, self.state.params.encoder)

    def save_policy(self, path: str) -> None:
        policy = jax.device_get(self.get_policy_params_to_save())
        with open(path, "wb") as f:
            pickle.dump(policy, f)

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        action = self._get_action(key, self.get_policy_params_to_save(), obs)
        return np.asarray(action)
    
    def warmup(self, data: tuple) -> None:
        key = jax.random.key(0)
        obs, _, _, _, _ = data
        obs = obs[0]
        policy_params = self.get_policy_params()
        self._update(key, self.state, data)
        self._get_action(key, policy_params, obs)
        self._get_deterministic_action(policy_params, obs)