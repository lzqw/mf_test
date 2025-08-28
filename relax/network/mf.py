from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple, Union
import functools

import jax, jax.numpy as jnp
import haiku as hk
import math

from relax.network.blocks import Activation, DACERPolicyNet2, QNet
from relax.utils.flow import MeanFlow
from relax.utils.jax_utils import random_key_from_data

class Diffv2Params(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    policy: hk.Params
    target_poicy: hk.Params
    log_alpha: jax.Array


@dataclass
class MFNet:
    q: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    policy: Callable[[hk.Params, jax.Array, jax.Array, jax.Array], jax.Array]
    num_timesteps: int
    num_timesteps_test: int
    act_dim: int
    num_particles: int
    target_entropy: float
    noise_scale: float
    noise_schedule: str

    @property
    def flow(self) -> MeanFlow:
        return MeanFlow(self.num_timesteps)
    
    @property
    def flow_test(self) -> MeanFlow:
        return MeanFlow(self.num_timesteps_test)

    def get_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        policy_params, log_alpha, q1_params, q2_params = policy_params

        def model_fn(x, r, t):
            return self.policy(policy_params, obs, x, r, t)

        def sample(key: jax.Array) -> Union[jax.Array, jax.Array]:
            act = self.flow.p_sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
            q1 = self.q(q1_params, obs, act)
            q2 = self.q(q2_params, obs, act)
            q = jnp.minimum(q1, q2)
            return act.clip(-1, 1), q

        key, noise_key = jax.random.split(key)
        if self.num_particles == 1:
            act = sample(key)[0]
        else:
            keys = jax.random.split(key, self.num_particles)
            acts, qs = jax.vmap(sample)(keys)
            q_best_ind = jnp.argmax(qs, axis=0, keepdims=True)
            act = jnp.take_along_axis(acts, q_best_ind[..., None], axis=0).squeeze(axis=0)
        act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(log_alpha) * self.noise_scale
        return act
    
    def get_vanilla_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        policy_params, _, _, _ = policy_params

        def model_fn(x, r, t):
            return self.policy(policy_params, obs, x, r, t)

        def sample(key: jax.Array) -> Union[jax.Array, jax.Array]:
            act = self.flow.p_sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
            return act.clip(-1, 1)

        act = sample(key)
        return act
    
    def get_vanilla_action_step(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        policy_params, _, _, _ = policy_params

        def model_fn(x, r, t):
            return self.policy(policy_params, obs, x, r, t)

        def sample(key: jax.Array) -> Union[jax.Array, jax.Array]:
            act = self.flow_test.p_sample_traj(key, model_fn, (*obs.shape[:-1], self.act_dim))
            return act

        act = sample(key)
        return act

    def get_deterministic_action(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        key = random_key_from_data(obs)
        policy_params, log_alpha, q1_params, q2_params = policy_params
        log_alpha = -jnp.inf
        policy_params = (policy_params, log_alpha, q1_params, q2_params)
        return self.get_action(key, policy_params, obs)
    

def create_mf_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    diffusion_hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
    num_timesteps: int = 20,
    num_timesteps_test: int = 20,
    num_particles: int = 32,
    noise_scale: float = 0.05,
    target_entropy_scale = 0.9,
    ) -> Tuple[MFNet, Diffv2Params]:
    q = hk.without_apply_rng(hk.transform(lambda obs, act: QNet(hidden_sizes, activation)(obs, act)))
    policy = hk.without_apply_rng(hk.transform(lambda obs, act, r, t: DACERPolicyNet2(diffusion_hidden_sizes, activation)(obs, act, r, t)))

    @jax.jit
    def init(key, obs, act):
        q1_key, q2_key, policy_key = jax.random.split(key, 3)
        q1_params = q.init(q1_key, obs, act)
        q2_params = q.init(q2_key, obs, act)
        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params = policy.init(policy_key, obs, act, 0, 0)
        target_policy_params = policy_params
        log_alpha = jnp.array(math.log(5), dtype=jnp.float32) # math.log(3) or math.log(5) choose one
        return Diffv2Params(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, log_alpha)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)

    net = MFNet(q=q.apply, policy=policy.apply, num_timesteps=num_timesteps, num_timesteps_test=num_timesteps_test, act_dim=act_dim, 
                    target_entropy=-act_dim*target_entropy_scale, num_particles=num_particles, noise_scale=noise_scale,
                    noise_schedule='linear')
    return net, params
