from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple

import jax, jax.numpy as jnp
import haiku as hk

from relax.network.blocks import Activation, QNet_V, PolicyNet_V, EncoderNet
from relax.network.common import WithSquashedGaussianPolicy


class SACParams(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    policy: hk.Params
    log_alpha: jax.Array
    encoder: jax.Array


@dataclass
class SACNet_V(WithSquashedGaussianPolicy):
    q: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    encoder: Callable[[hk.Params, jax.Array], jax.Array]
    target_entropy: float

    def get_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        policy_params, encoder_params = policy_params
        obs = self.encoder(encoder_params, obs)
        mean, std = self.policy(policy_params, obs)
        z = jax.random.normal(key, mean.shape)
        act = mean + std * z
        return jnp.tanh(act)

    def get_deterministic_action(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        policy_params, encoder_params = policy_params
        obs = self.encoder(encoder_params, obs)
        mean, _ = self.policy(policy_params, obs)
        return jnp.tanh(mean)


def create_sac_net_visual(
    key: jax.Array,
    obs_dim: int,
    latent_obs_dim: int, 
    act_dim: int,
    hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
) -> Tuple[SACNet_V, SACParams]:
    q = hk.without_apply_rng(hk.transform(lambda obs, act: QNet_V(hidden_sizes, activation)(obs, act)))
    policy = hk.without_apply_rng(hk.transform(lambda obs: PolicyNet_V(act_dim, hidden_sizes, activation)(obs)))
    encoder = hk.without_apply_rng(hk.transform(lambda obs: EncoderNet()(obs)))

    @jax.jit
    def init(key, obs, latent_obs, act):
        q1_key, q2_key, policy_key, encoder_key = jax.random.split(key, 4)
        q1_params = q.init(q1_key, latent_obs, act)
        q2_params = q.init(q2_key, latent_obs, act)
        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params = policy.init(policy_key, latent_obs)
        log_alpha = jnp.array(1.0, dtype=jnp.float32)
        encoder_params = encoder.init(encoder_key, obs)
        return SACParams(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, log_alpha, encoder_params)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_latent_obs = jnp.zeros((1, latent_obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_latent_obs, sample_act)

    net = SACNet_V(policy=policy.apply, q=q.apply, encoder=encoder.apply, target_entropy=-act_dim)
    return net, params
