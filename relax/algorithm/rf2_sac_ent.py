from functools import partial
from typing import NamedTuple, Tuple

import jax, jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

import numpy as np
import optax
import haiku as hk
import pickle

from relax.algorithm.base import Algorithm
from relax.network.rf2_sac_ent import RF2SACENTNet, Diffv2Params
from scripts.experience import Experience
from relax.utils.typing import Metric



class Diffv2OptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState


class Diffv2TrainState(NamedTuple):
    params: Diffv2Params
    opt_state: Diffv2OptStates
    step: int
    entropy: float
    running_mean: float
    running_std: float

class RF2SACENT(Algorithm):

    def __init__(
        self,
        agent: RF2SACENTNet,
        params: Diffv2Params,
        *,
        gamma: float = 0.99,
        lr: float = 1e-4,
        alpha_lr: float = 3e-2,
        lr_schedule_end: float = 5e-5,
        tau: float = 0.005,
        delay_alpha_update: int = 2000,
        delay_update: int = 2,
        reward_scale: float = 0.2,
        num_samples: int = 200,
        use_ema: bool = True,
        sample_k:int = 500,
        alpha_value: float = 0.01,
        fixed_alpha: bool = True,
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
        self.entropy = 0.0
        self.fixed_alpha=fixed_alpha
        self.state = Diffv2TrainState(
            params=params,
            opt_state=Diffv2OptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                # policy=self.optim.init(params.policy),
                policy=self.policy_optim.init(params.policy),
                log_alpha=self.alpha_optim.init(params.log_alpha),
            ),
            step=jnp.int32(0),
            entropy=jnp.float32(0.0),
            running_mean=jnp.float32(0.0),
            running_std=jnp.float32(1.0)
        )
        self.use_ema = use_ema
        self.fixed_alpha = fixed_alpha
        self.alpha_value = alpha_value
        self.K=sample_k

        @jax.jit
        def stateless_update(
            key: jax.Array, state: Diffv2TrainState, data: Experience
        ) -> Tuple[Diffv2OptStates, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, log_alpha = state.params
            q1_opt_state, q2_opt_state, policy_opt_state, log_alpha_opt_state = state.opt_state
            step = state.step
            running_mean = state.running_mean
            running_std = state.running_std
            next_eval_key, new_eval_key, new_q1_eval_key, new_q2_eval_key, log_alpha_key, flow_time_key, flow_noise_key = jax.random.split(
                key, 7)

            reward *= self.reward_scale

            def get_min_q(s, a):
                q1 = self.agent.q(q1_params, s, a)
                q2 = self.agent.q(q2_params, s, a)
                q = jnp.minimum(q1, q2)
                return q

            # Get the next action and its corresponding entropy from the policy
            next_action, entropy = self.agent.get_action_ent(next_eval_key,
                                                             (policy_params, log_alpha, q1_params, q2_params),
                                                             next_obs)

            # Calculate target Q-values
            q1_target = self.agent.q(target_q1_params, next_obs, next_action)
            q2_target = self.agent.q(target_q2_params, next_obs, next_action)

            if self.fixed_alpha:
                q_target = jnp.minimum(q1_target, q2_target)#  - jnp.float32(self.alpha_value) * entropy
            else:
                q_target = jnp.minimum(q1_target, q2_target)# - jnp.exp(log_alpha) * entropy

            q_backup = reward + (1 - done) * self.gamma * q_target

            def q_loss_fn(q_params: hk.Params) -> jax.Array:
                q = self.agent.q(q_params, obs, action)
                q_loss = jnp.mean((q - q_backup) ** 2)
                return q_loss, q

            (q1_loss, q1), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params)
            (q2_loss, q2), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params)


            # prev_entropy = state.entropy if hasattr(state, 'entropy') else jnp.float32(0.0)

            # get the noise action
            flow_time_key,time_rng=jax.random.split(flow_time_key,2)
            flow_noise_key,noise_rng=jax.random.split(flow_noise_key,2)
            t = jax.random.uniform(flow_time_key, shape=(next_obs.shape[0],), minval=1e-3, maxval=0.9946)
            t = jnp.expand_dims(t, axis=1)

            # uniform_key, key = jax.random.split(key)
            # x_start_uniform = jax.random.uniform(uniform_key, shape=action.shape, minval=-1.0, maxval=1.0)
            # action=x_start_uniform
            noise_sample = jax.random.normal(flow_noise_key, action.shape)



            def q_sample(t: int, x_start: jax.Array, noise: jax.Array):
                return t * x_start + (1 - t) * noise

            noisy_actions = q_sample(t, action, noise_sample)
            noisy_actions=noisy_actions.clip(-1,1)

            #TODO: is a1=(1/t)at+(1-t)/t*et or a1=(1/t)at-(1-t)/t*et
            noisy_actions_repeat = jnp.repeat(jnp.expand_dims(noisy_actions, axis=1), axis=1, repeats=self.K)
            std = jnp.expand_dims((1-t) / t, axis=-1)
            lower_bound = 1 / (1-t)[:, :, None] * noisy_actions_repeat - (1 / std)
            upper_bound = 1 / (1-t)[:, :, None] * noisy_actions_repeat + (1 / std)
            #action: batch_size, action_dim
            tnormal_noise = jax.random.truncated_normal(
                key, lower=lower_bound, upper=upper_bound, shape=(action.shape[0], self.K, action.shape[1]))
            flow_noise_key,noise_rng=jax.random.split(flow_noise_key,2)
            normal_noise = jax.random.normal(flow_noise_key, shape=((action.shape[0], self.K, action.shape[1])))
            normal_noise_clip = jnp.clip(normal_noise, min=lower_bound, max=upper_bound)
            noise = jnp.where(jnp.isnan(tnormal_noise), normal_noise_clip, tnormal_noise)
            clean_samples = 1 / t[:, :, None] * noisy_actions_repeat - std * noise

            observations_repeat = jnp.repeat(jnp.expand_dims(obs, axis=1), axis=1, repeats=self.K)

            devices = jax.devices()
            compute_Q_DDP = partial(shard_map, mesh=Mesh(devices, ('i',)), in_specs=(P('i'), P('i')), out_specs=(P('i')))(get_min_q)
            critic = compute_Q_DDP( observations_repeat, clean_samples)  # batch_size, K
            # critic=critic*self.init_alpha / jnp.exp(log_alpha)
            # critic=critic/ jnp.float32(0.1)
            if self.fixed_alpha:
                critic=critic/jnp.float32(agent.alpha_value)
            else:
                safe_alpha = jnp.maximum(jnp.exp(log_alpha), 0.001)
                critic=critic/safe_alpha
            q_mean, q_std = critic.mean(), critic.std()
            Z = jax.nn.logsumexp(critic, axis=1, keepdims=True)  # [batch_size, 1]
            q_weights = jnp.exp(critic - Z) # [batch_size, mc_num]

            clean_samples_reshape=clean_samples.reshape(-1, clean_samples.shape[-1])
            obs_reshape = observations_repeat.reshape(-1, observations_repeat.shape[-1])
            noise_reshape=noise.reshape(-1, noise.shape[-1])
            u=clean_samples_reshape-noise_reshape
            t_reshape = jnp.repeat(t.squeeze(), repeats=self.K)
            noisy_actions_reshape=noisy_actions_repeat.reshape(-1, noisy_actions_repeat.shape[-1])
            weight_reshape=q_weights.reshape(-1,1)
            d = action.shape[-1]  # 获取动作维度 d
            t_factor = jnp.power(t_reshape, -d).reshape(-1, 1)  # shape: (batch_size * K, 1)

            # 【实验版本 1: Exact t^(-d)】严格相乘，高维极大概率直接 NaN
            exact_weight_reshape = weight_reshape * t_factor

            # 【实验版本 2: Clipped t^(-d)】对权重进行硬截断，防止程序崩溃但破坏了分布
            # exact_weight_reshape = weight_reshape * jnp.clip(t_factor, a_max=1e4)

            # 【默认版本 (Ours)】不乘 t_factor，即下方的传参保持 weight_reshape
            # ===============================================================


            #clean_samples: batch_size, K, action_dim,
            def policy_loss_fn(policy_params) -> jax.Array:

                def denoiser(t, x):
                    return self.agent.policy(policy_params, obs_reshape, x, t)
                #loss = self.agent.flow.reverse_weighted_p_loss2(denoiser, t_reshape, noisy_actions_reshape,
                #                                                jax.lax.stop_gradient(weight_reshape),
                #                                            jax.lax.stop_gradient(u))


                loss = self.agent.flow.reverse_weighted_p_loss2(denoiser, t_reshape, noisy_actions_reshape,
                                                                jax.lax.stop_gradient(exact_weight_reshape), # <--- 修改这里
                                                                jax.lax.stop_gradient(u))
                """
                    def reverse_weighted_p_loss(self,  model: FlowModel, t: jax.Array,
                        x_t: jax.Array, u_estimation:jax.Array):
                """
                return loss, (weight_reshape, critic, q_mean, q_std)

            (total_loss, (weight, u_estimation, critic_mean, critic_std)), policy_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_params)

            # update alpha
            def log_alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
                approx_entropy = jnp.mean(entropy)
                #log_alpha_loss = jnp.mean(log_alpha * (approx_entropy-self.agent.target_entropy))
                log_alpha_loss = jnp.mean(log_alpha * (approx_entropy-self.agent.target_entropy))
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

            new_running_mean = running_mean + 0.001 * (critic_mean - running_mean)
            new_running_std = running_std + 0.001 * (critic_std - running_std)

            state = Diffv2TrainState(
                params=Diffv2Params(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, log_alpha),
                opt_state=Diffv2OptStates(q1=q1_opt_state, q2=q2_opt_state, policy=policy_opt_state, log_alpha=log_alpha_opt_state),
                step=step + 1,
                entropy=entropy,
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
                "weights_std": jnp.std(weight),
                "weights_mean": jnp.mean(weight),
                "weights_min": jnp.min(weight),
                "weights_max": jnp.max(weight),
                "u_estimation_mean": jnp.mean(u_estimation),
                "u_estimation_std": jnp.std(u_estimation),
                "critic_mean": critic_mean,
                "critic_std": critic_std,
                "running_q_mean": new_running_mean,
                "running_q_std": new_running_std,
                "entropy_approx": jnp.mean(entropy),
            }
            return state, info

        self._implement_common_behavior(stateless_update, self.agent.get_action, self.agent.get_deterministic_action,
                                        stateless_get_vanilla_action=self.agent.get_vanilla_action,
                                        stateless_get_vanilla_action_step=self.agent.get_vanilla_action_step)

    def get_policy_params(self):
        return (self.state.params.policy, self.state.params.log_alpha, self.state.params.q1, self.state.params.q2 )

    def get_policy_params_to_save(self):
        return (self.state.params.target_poicy, self.state.params.log_alpha, self.state.params.q1, self.state.params.q2)

    def save_policy(self, path: str) -> None:
        policy = jax.device_get(self.get_policy_params_to_save())
        with open(path, "wb") as f:
            pickle.dump(policy, f)

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        action = self._get_action(key, self.get_policy_params_to_save(), obs)
        return np.asarray(action)


