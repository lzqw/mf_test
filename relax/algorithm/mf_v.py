from typing import NamedTuple, Tuple

import jax, jax.numpy as jnp
import numpy as np
import optax
import haiku as hk
import pickle

from relax.algorithm.base import Algorithm
from relax.network.mf_v import MFNet_V, Diffv2Params
from relax.utils.experience import Experience
from relax.utils.typing import Metric


class Diffv2OptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState
    encoder: optax.OptState


class Diffv2TrainState(NamedTuple):
    params: Diffv2Params
    opt_state: Diffv2OptStates
    step: int
    entropy: float
    running_mean: float
    running_std: float

@jax.jit
def augment_batch(obs: jnp.ndarray,
                  next_obs: jnp.ndarray,
                  obs_key: jax.Array,
                  next_obs_key: jax.Array,
                  padding: int = 4
                 ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    
    def random_crop(key, img, padding):
        crop_from = jax.random.randint(key, (2, ), 0, 2 * padding + 1)
        crop_from = jnp.concatenate([crop_from, jnp.zeros((1, ), dtype=jnp.int32)])
        padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)),
                            mode='edge')
        return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)
    
    obs_keys = jax.random.split(obs_key, obs.shape[0])
    obs = jnp.reshape(obs, (obs.shape[0], -1, 84, 84))
    obs = obs.transpose((0, 2, 3, 1))
    obs = jax.vmap(random_crop, (0, 0, None))(obs_keys, obs, padding)
    obs = obs.transpose((0, 3, 1, 2))

    next_obs_keys = jax.random.split(next_obs_key, next_obs.shape[0])
    next_obs = jnp.reshape(next_obs, (next_obs.shape[0], -1, 84, 84))
    next_obs = next_obs.transpose((0, 2, 3, 1))
    next_obs = jax.vmap(random_crop, (0, 0, None))(next_obs_keys, next_obs, padding)
    next_obs = next_obs.transpose((0, 3, 1, 2))

    return jnp.squeeze(jnp.reshape(obs, (obs.shape[0], -1))), jnp.squeeze(jnp.reshape(next_obs, (next_obs.shape[0], -1)))

class MF_V(Algorithm):

    def __init__(
        self,
        agent: MFNet_V,
        params: Diffv2Params,
        *,
        gamma: float = 0.99,
        lr: float = 1e-4,
        alpha_lr: float = 3e-2,
        lr_schedule_end: float = 5e-5,
        tau: float = 0.005,
        delay_alpha_update: int = 250,
        delay_update: int = 2,
        reward_scale: float = 1.0,
        num_samples: int = 200,
        use_ema: bool = True,
        temperature: float = 1.0,
    ):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.delay_alpha_update = delay_alpha_update
        self.delay_update = delay_update
        self.reward_scale = reward_scale
        self.num_samples = num_samples
        self.optim = optax.adam(lr)
        lr_schedule = optax.schedules.linear_schedule(
            init_value=lr,
            end_value=lr_schedule_end,
            transition_steps=int(5e4),
            transition_begin=int(2.5e4),
        )
        self.policy_optim = optax.adam(learning_rate=lr_schedule)
        self.alpha_optim = optax.adam(alpha_lr)
        self.encoder_optim = optax.adam(lr)
        self.entropy = 0.0

        self.state = Diffv2TrainState(
            params=params,
            opt_state=Diffv2OptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                # policy=self.optim.init(params.policy),
                policy=self.policy_optim.init(params.policy),
                log_alpha=self.alpha_optim.init(params.log_alpha),
                encoder=self.encoder_optim.init(params.encoder),
            ),
            step=jnp.int32(0),
            entropy=jnp.float32(0.0),
            running_mean=jnp.float32(0.0),
            running_std=jnp.float32(1.0)
        )
        self.use_ema = use_ema
        self.temperature = temperature

        @jax.jit
        def stateless_update(
            key: jax.Array, state: Diffv2TrainState, data: Experience
        ) -> Tuple[Diffv2OptStates, Metric]:
            obs, action, reward, next_obs, discount = data
            if len(action.shape) == 2 and action.shape[0] == 1:
                action = np.squeeze(action, axis=0)
            reward = np.squeeze(reward)
            discount = np.squeeze(discount)
            q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, log_alpha, encoder_params = state.params
            q1_opt_state, q2_opt_state, policy_opt_state, log_alpha_opt_state, encoder_opt_state = state.opt_state
            step = state.step
            running_mean = state.running_mean
            running_std = state.running_std
            next_eval_key, acts_key, flow_noise_key, r_key, mask_key, t_key, obs_aug_key, next_obs_aug_key = jax.random.split(
                key, 8)

            # data augmentation
            obs, next_obs = augment_batch(obs, next_obs, obs_aug_key, next_obs_aug_key)

            reward *= self.reward_scale
            next_obs = jax.lax.stop_gradient(self.agent.encoder(encoder_params, next_obs))

            def get_min_q(s, a):
                q1 = self.agent.q(q1_params, s, a)
                q2 = self.agent.q(q2_params, s, a)
                q = jnp.minimum(q1, q2)
                return q

            next_action = self.agent.get_action(next_eval_key, (policy_params, log_alpha, q1_params, q2_params, encoder_params), next_obs)
            q1_target = self.agent.q(target_q1_params, next_obs, next_action)
            q2_target = self.agent.q(target_q2_params, next_obs, next_action)
            q_target = jnp.minimum(q1_target, q2_target)  # - jnp.exp(log_alpha) * next_logp
            q_backup = reward + discount * q_target
            
            def q_loss_fn(q1_params: hk.Params, q2_params: hk.Params, encoder_params: hk.Params) -> jax.Array:
                obs_latent = self.agent.encoder(encoder_params, obs)
                q1 = self.agent.q(q1_params, obs_latent, action)
                q1_loss = jnp.mean((q1 - q_backup) ** 2)
                q2 = self.agent.q(q2_params, obs_latent, action)
                q2_loss = jnp.mean((q2 - q_backup) ** 2)
                q_loss = q1_loss + q2_loss
                return q_loss, (q1_loss, q2_loss, q1, q2)
            
            (q_loss, (q1_loss, q2_loss, q1, q2)), (q1_grads, q2_grads, encoder_grads) = jax.value_and_grad(q_loss_fn, argnums=(0, 1, 2), has_aux=True)(q1_params, q2_params, encoder_params)
            q1_update, q1_opt_state = self.optim.update(q1_grads, q1_opt_state)
            q2_update, q2_opt_state = self.optim.update(q2_grads, q2_opt_state)
            encoder_update, encoder_opt_state = self.optim.update(encoder_grads, encoder_opt_state)
            q1_params = optax.apply_updates(q1_params, q1_update)
            q2_params = optax.apply_updates(q2_params, q2_update)
            encoder_params = optax.apply_updates(encoder_params, encoder_update)

            def policy_loss_fn(policy_params) -> jax.Array:
                q_min = get_min_q(next_obs, next_action)
                q_mean, q_std = q_min.mean(), q_min.std()
                # norm_q = q_min - running_mean / running_std
                # scaled_q = norm_q.clip(-3., 3.) / jnp.exp(log_alpha)
                norm_q = q_min / running_std
                scaled_q = norm_q / self.temperature
                q_weights = jnp.exp(scaled_q)
                def denoiser(x, r, t):
                    return self.agent.policy(policy_params, next_obs, x, r, t)
                
                r0 = jax.random.uniform(r_key, shape=(next_obs.shape[0],), minval=0.0, maxval=1.0)
                #0.75
                mask = jax.random.bernoulli(mask_key, p=0.0, shape=(next_obs.shape[0],))  
                t0 = jax.random.uniform(t_key, shape=(next_obs.shape[0],), minval=0.0, maxval=1.0)
                is_t_gt_r = t0 > r0
                t_swap = jnp.where(is_t_gt_r, t0, r0)
                r_swap = jnp.where(is_t_gt_r, r0, t0)
                r_final = jnp.where(mask, r0, r_swap)
                t_final = jnp.where(mask, r0, t_swap)

                loss = self.agent.flow.weighted_p_loss(flow_noise_key, q_weights, denoiser, r_final, t_final,
                                                            jax.lax.stop_gradient(next_action))
                
                # acts = self.agent.get_vanilla_action(acts_key, (policy_params, log_alpha, q1_params, q2_params, encoder_params), next_obs)
                # q1_target = self.agent.q(target_q1_params, next_obs, acts)
                # q2_target = self.agent.q(target_q2_params, next_obs, acts)
                # q_target = jnp.minimum(q1_target, q2_target)
                # loss = jnp.mean(-q_target)

                return loss, (q_weights, scaled_q, q_mean, q_std)
                # return loss, (0, 0, 0, 0)

            (total_loss, (q_weights, scaled_q, q_mean, q_std)), policy_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_params)

            # update alpha
            def log_alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
                approx_entropy = 0.5 * self.agent.act_dim * jnp.log( 2 * jnp.pi * jnp.exp(1) * (0.1 * jnp.exp(log_alpha)) ** 2)
                log_alpha_loss = -1 * log_alpha * (-1 * jax.lax.stop_gradient(approx_entropy) + self.agent.target_entropy)
                return log_alpha_loss

            # update networks
            def param_update(optim, params, grads, opt_state):
                update, new_opt_state = optim.update(grads, opt_state)
                new_params = optax.apply_updates(params, update)
                return new_params, new_opt_state

            def delay_param_update(optim, params, grads, opt_state):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda params, opt_state: param_update(optim, params, grads, opt_state),
                    lambda params, opt_state: (params, opt_state),
                    params, opt_state
                )

            def delay_alpha_param_update(optim, params, opt_state):
                return jax.lax.cond(
                    step % self.delay_alpha_update == 0,
                    lambda params, opt_state: param_update(optim, params, jax.grad(log_alpha_loss_fn)(params), opt_state),
                    lambda params, opt_state: (params, opt_state),
                    params, opt_state
                )

            def delay_target_update(params, target_params, tau):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda target_params: optax.incremental_update(params, target_params, tau),
                    lambda target_params: target_params,
                    target_params
                )

            q1_params, q1_opt_state = param_update(self.optim, q1_params, q1_grads, q1_opt_state)
            q2_params, q2_opt_state = param_update(self.optim, q2_params, q2_grads, q2_opt_state)
            policy_params, policy_opt_state = delay_param_update(self.policy_optim, policy_params, policy_grads, policy_opt_state)
            log_alpha, log_alpha_opt_state = delay_alpha_param_update(self.alpha_optim, log_alpha, log_alpha_opt_state)

            target_q1_params = delay_target_update(q1_params, target_q1_params, self.tau)
            target_q2_params = delay_target_update(q2_params, target_q2_params, self.tau)
            target_policy_params = delay_target_update(policy_params, target_policy_params, self.tau)

            new_running_mean = running_mean + 0.001 * (q_mean - running_mean)
            new_running_std = running_std + 0.001 * (q_std - running_std)

            state = Diffv2TrainState(
                params=Diffv2Params(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, log_alpha, encoder_params),
                opt_state=Diffv2OptStates(q1=q1_opt_state, q2=q2_opt_state, policy=policy_opt_state, log_alpha=log_alpha_opt_state, encoder=encoder_opt_state),
                step=step + 1,
                entropy=jnp.float32(0.0),
                running_mean=new_running_mean,
                running_std=new_running_std
            )
            info = {
                "q1_loss": q1_loss,
                "q1_mean": jnp.mean(q1),
                "q1_max": jnp.max(q1),
                "q1_min": jnp.min(q1),
                "q2_loss": q2_loss,
                "policy_loss": total_loss,
                "alpha": jnp.exp(log_alpha),
                "q_weights_std": jnp.std(q_weights),
                "q_weights_mean": jnp.mean(q_weights),
                "q_weights_min": jnp.min(q_weights),
                "q_weights_max": jnp.max(q_weights),
                "scale_q_mean": jnp.mean(scaled_q),
                "scale_q_std": jnp.std(scaled_q),
                "running_q_mean": new_running_mean,
                "running_q_std": new_running_std,
                "entropy_approx": 0.5 * self.agent.act_dim * jnp.log( 2 * jnp.pi * jnp.exp(1) * (0.1 * jnp.exp(log_alpha)) ** 2),
            }
            return state, info
        
        self._implement_common_behavior(stateless_update, self.agent.get_action, self.agent.get_deterministic_action, 
                                        stateless_get_action_full=self.agent.get_action_full,
                                        stateless_get_vanilla_action_step=self.agent.get_vanilla_action_step)

    def get_policy_params(self):
        return (self.state.params.policy, self.state.params.log_alpha, self.state.params.q1, self.state.params.q2, self.state.params.encoder)

    def get_policy_params_to_save(self):
        return (self.state.params.target_poicy, self.state.params.log_alpha, self.state.params.q1, self.state.params.q2, self.state.params.encoder)

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
