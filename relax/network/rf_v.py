from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple, Union
import functools

import jax, jax.numpy as jnp
import haiku as hk
import math

from relax.network.blocks import Activation, DACERPolicyNet_V, QNet_V, EncoderNet
from relax.utils.flow import OTFlow
from relax.utils.jax_utils import random_key_from_data

class Diffv2Params(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    policy: hk.Params
    target_poicy: hk.Params
    log_alpha: jax.Array
    encoder: hk.Params


@dataclass
class RFNet_V:
    q: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    policy: Callable[[hk.Params, jax.Array, jax.Array, jax.Array], jax.Array]
    encoder: Callable[[hk.Params, jax.Array], jax.Array]
    num_timesteps: int
    num_timesteps_test: int
    act_dim: int
    num_particles: int
    target_entropy: float
    noise_scale: float
    noise_schedule: str

    @property
    def flow(self) -> OTFlow:
        return OTFlow(self.num_timesteps)
    
    @property
    def flow_test(self) -> OTFlow:
        return OTFlow(self.num_timesteps_test)

    def get_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        policy_params, log_alpha, q1_params, q2_params, _ = policy_params

        def model_fn(t, x):
            return self.policy(policy_params, obs, x, t)

        def sample(key: jax.Array) -> Union[jax.Array, jax.Array]:
            sample_key, gnoise_key = jax.random.split(key, 2)
            act = self.flow.p_sample(sample_key, model_fn, (*obs.shape[:-1], self.act_dim))
            act = act + jax.random.normal(gnoise_key, act.shape) * jnp.exp(log_alpha) * self.noise_scale
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
        # act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(log_alpha) * self.noise_scale / 2 
        return act
    
    def get_action_full(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        policy_params, log_alpha, q1_params, q2_params, encoder_params = policy_params

        def model_fn(t, x):
            return self.policy(policy_params, obs, x, t)

        def sample(key: jax.Array) -> Union[jax.Array, jax.Array]:
            act = self.flow.p_sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
            q1 = self.q(q1_params, obs, act)
            q2 = self.q(q2_params, obs, act)
            q = jnp.minimum(q1, q2)
            return act.clip(-1, 1), q
        
        obs = self.encoder(encoder_params, obs)

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
        policy_params, _, _, _, encoder_params = policy_params

        obs = self.encoder(encoder_params, obs)

        def model_fn(t, x):
            return self.policy(policy_params, obs, x, t)

        def sample(key: jax.Array) -> Union[jax.Array, jax.Array]:
            act = self.flow.p_sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
            return act.clip(-1, 1)

        act = sample(key)
        return act
    
    def get_vanilla_action_step(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        policy_params, _, _, _, encoder_params = policy_params
        obs = self.encoder(encoder_params, obs)

        def model_fn(t, x):
            return self.policy(policy_params, obs, x, t)

        def sample(key: jax.Array) -> Union[jax.Array, jax.Array]:
            act = self.flow_test.p_sample_traj(key, model_fn, (*obs.shape[:-1], self.act_dim))
            return act

        act = sample(key)
        return act

    def get_vanilla_action_fast(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        policy_params, _, _, _, encoder_params = policy_params
        obs = self.encoder(encoder_params, obs)

        def model_fn(t, x):
            return self.policy(policy_params, obs, x, t)

        def sample() -> Union[jax.Array, jax.Array]:
            act = self.flow_test.p_sample_fast(model_fn, (*obs.shape[:-1], self.act_dim))
            return act

        act = sample()
        return act

    def get_deterministic_action(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        key = random_key_from_data(obs)
        policy_params, log_alpha, q1_params, q2_params, encoder_params = policy_params
        log_alpha = -jnp.inf
        policy_params = (policy_params, log_alpha, q1_params, q2_params, encoder_params)
        return self.get_action_full(key, policy_params, obs)
    

def create_rf_net_visual(
    key: jax.Array,
    obs_dim: int,
    latent_obs_dim: int, 
    act_dim: int,
    hidden_sizes: Sequence[int],
    diffusion_hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
    num_timesteps: int = 20,
    num_timesteps_test: int = 20,
    num_particles: int = 32,
    noise_scale: float = 0.05,
    target_entropy_scale = 0.9,
    ) -> Tuple[RFNet_V, Diffv2Params]:
    # q = hk.without_apply_rng(hk.transform(lambda obs, act: DistributionalQNet2(hidden_sizes, activation)(obs, act)))
    q = hk.without_apply_rng(hk.transform(lambda obs, act: QNet_V(hidden_sizes, activation)(obs, act)))
    policy = hk.without_apply_rng(hk.transform(lambda obs, act, t: DACERPolicyNet_V(diffusion_hidden_sizes, activation)(obs, act, t)))
    encoder = hk.without_apply_rng(hk.transform(lambda obs: EncoderNet()(obs)))

    @jax.jit
    def init(key, obs, latent_obs, act):
        q1_key, q2_key, policy_key, encoder_key = jax.random.split(key, 4)
        q1_params = q.init(q1_key, latent_obs, act)
        q2_params = q.init(q2_key, latent_obs, act)
        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params = policy.init(policy_key, latent_obs, act, 0)
        target_policy_params = policy_params
        log_alpha = jnp.array(math.log(5), dtype=jnp.float32) # math.log(3) or math.log(5) choose one
        encoder_params = encoder.init(encoder_key, obs)
        return Diffv2Params(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, log_alpha, encoder_params)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_latent_obs = jnp.zeros((1, latent_obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_latent_obs, sample_act)

    net = RFNet_V(q=q.apply, policy=policy.apply, encoder=encoder.apply, num_timesteps=num_timesteps, num_timesteps_test=num_timesteps_test, act_dim=act_dim, 
                    target_entropy=-act_dim*target_entropy_scale, num_particles=num_particles, noise_scale=noise_scale,
                    noise_schedule='linear')
    return net, params
