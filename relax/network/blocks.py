from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union

import jax, jax.numpy as jnp
import haiku as hk
from haiku.initializers import Constant

from relax.utils.jax_utils import fix_repr, is_broadcastable

Activation = Callable[[jax.Array], jax.Array]
Identity: Activation = lambda x: x
Tanh: Activation = lambda x: jnp.tanh(x)


@dataclass
@fix_repr
class ValueNet(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    def __call__(self, obs: jax.Array) -> jax.Array:
        return mlp(self.hidden_sizes, 1, self.activation, self.output_activation, squeeze_output=True)(obs)


@dataclass
@fix_repr
class QNet(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array) -> jax.Array:
        input = jnp.concatenate((obs, act), axis=-1)
        return mlp(self.hidden_sizes, 1, self.activation, self.output_activation, squeeze_output=True)(input)
    
@dataclass
@fix_repr
class QNet_V(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    feature_dim: int = 50
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array) -> jax.Array:
        x = hk.Linear(self.feature_dim)(obs)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.tanh(x)
        input = jnp.concatenate((x, act), axis=-1)
        return mlp(self.hidden_sizes, 1, self.activation, self.output_activation, squeeze_output=True)(input)


@dataclass
@fix_repr
class DistributionalQNet(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    min_log_std: float = -0.1
    max_log_std: float = 4.0
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array) -> Tuple[jax.Array, jax.Array]:
        input = jnp.concatenate((obs, act), axis=-1)
        value_mean = mlp(self.hidden_sizes, 1, self.activation, self.output_activation, squeeze_output=True)(input)
        value_log_std = mlp(self.hidden_sizes, 1, self.activation, self.output_activation, squeeze_output=True)(input)
        denominator = max(abs(self.min_log_std), abs(self.max_log_std))
        value_log_std = (
            jnp.maximum( self.max_log_std * jnp.tanh(value_log_std / denominator), 0.0) +
            jnp.minimum(-self.min_log_std * jnp.tanh(value_log_std / denominator), 0.0)
        )
        return value_mean, value_log_std

@dataclass
@fix_repr
class DistributionalQNet2(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array) -> Tuple[jax.Array, jax.Array]:
        input = jnp.concatenate((obs, act), axis=-1)
        output = mlp(self.hidden_sizes, 2, self.activation, self.output_activation)(input)
        value_mean = output[..., 0]
        value_std = jax.nn.softplus(output[..., 1])
        return value_mean, value_std


@dataclass
@fix_repr
class PolicyNet(hk.Module):
    act_dim: int
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    min_log_std: float = -20.0
    max_log_std: float = 0.5
    log_std_mode: Union[str, float] = 'shared'  # shared, separate, global (provide initial value)
    name: str = None

    def __call__(self, obs: jax.Array, *, return_log_std: bool = False) -> jax.Array:
        if self.log_std_mode == 'shared':
            output = mlp(self.hidden_sizes, self.act_dim * 2, self.activation, self.output_activation)(obs)
            mean, log_std = jnp.split(output, 2, axis=-1)
        elif self.log_std_mode == 'separate':
            mean = mlp(self.hidden_sizes, self.act_dim, self.activation, self.output_activation)(obs)
            log_std = mlp(self.hidden_sizes, self.act_dim, self.activation, self.output_activation)(obs)
        else:
            initial_log_std = float(self.log_std_mode)
            mean = mlp(self.hidden_sizes, self.act_dim, self.activation, self.output_activation)(obs)
            log_std = hk.get_parameter('log_std', shape=(self.act_dim,), init=Constant(initial_log_std))
            log_std = jnp.broadcast_to(log_std, mean.shape)
        if not (self.min_log_std is None and self.max_log_std is None):
            log_std = jnp.clip(log_std, self.min_log_std, self.max_log_std)
        if return_log_std:
            return mean, log_std
        else:
            return mean, jnp.exp(log_std)
        
@dataclass
@fix_repr
class PolicyNet_V(hk.Module):
    act_dim: int
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    min_log_std: float = -20.0
    max_log_std: float = 0.5
    log_std_mode: Union[str, float] = 'shared'  # shared, separate, global (provide initial value)
    feature_dim: int = 50
    name: str = None

    def __call__(self, obs: jax.Array, *, return_log_std: bool = False) -> jax.Array:
        obs = hk.Linear(self.feature_dim)(obs)
        obs = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(obs)
        obs = jax.nn.tanh(obs)
        if self.log_std_mode == 'shared':
            output = mlp(self.hidden_sizes, self.act_dim * 2, self.activation, self.output_activation)(obs)
            mean, log_std = jnp.split(output, 2, axis=-1)
        elif self.log_std_mode == 'separate':
            mean = mlp(self.hidden_sizes, self.act_dim, self.activation, self.output_activation)(obs)
            log_std = mlp(self.hidden_sizes, self.act_dim, self.activation, self.output_activation)(obs)
        else:
            initial_log_std = float(self.log_std_mode)
            mean = mlp(self.hidden_sizes, self.act_dim, self.activation, self.output_activation)(obs)
            log_std = hk.get_parameter('log_std', shape=(self.act_dim,), init=Constant(initial_log_std))
            log_std = jnp.broadcast_to(log_std, mean.shape)
        if not (self.min_log_std is None and self.max_log_std is None):
            log_std = jnp.clip(log_std, self.min_log_std, self.max_log_std)
        if return_log_std:
            return mean, log_std
        else:
            return mean, jnp.exp(log_std)
        

@dataclass
@fix_repr
class PolicyStdNet(hk.Module):
    act_dim: int
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Tanh
    min_log_std: float = -5.0
    max_log_std: float = 2.0
    name: str = None

    def __call__(self, obs: jax.Array) -> jax.Array:
        log_std = mlp(self.hidden_sizes, self.act_dim, self.activation, self.output_activation)(obs)
        return self.min_log_std + (log_std + 1) / 2 * (self.max_log_std - self.min_log_std)


@dataclass
@fix_repr
class DeterministicPolicyNet(hk.Module):
    act_dim: int
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    def __call__(self, obs: jax.Array) -> jax.Array:
        return mlp(self.hidden_sizes, self.act_dim, self.activation, self.output_activation)(obs)


@dataclass
@fix_repr
class ModelNet(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array) -> jax.Array:
        obs_dim = obs.shape[-1]
        input = jnp.concatenate((obs, act), axis=-1)
        return mlp(self.hidden_sizes, obs_dim, self.activation, self.output_activation)(input)


@dataclass
@fix_repr
class QScoreNet(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array) -> jax.Array:
        act_dim = act.shape[-1]
        input = jnp.concatenate((obs, act), axis=-1)
        return mlp(self.hidden_sizes, act_dim, self.activation, self.output_activation)(input)


@dataclass
@fix_repr
class DiffusionPolicyNet(hk.Module):
    time_dim: int
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array, t: jax.Array) -> jax.Array:
        act_dim = act.shape[-1]
        te = scaled_sinusoidal_encoding(t, dim=self.time_dim, batch_shape=obs.shape[:-1])
        input = jnp.concatenate((obs, act, te), axis=-1)
        return mlp(self.hidden_sizes, act_dim, self.activation, self.output_activation)(input)

@dataclass
@fix_repr
class DACERPolicyNet(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    time_dim: int = 16
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array, t: jax.Array) -> jax.Array:
        act_dim = act.shape[-1]
        te = scaled_sinusoidal_encoding(t, dim=self.time_dim, batch_shape=obs.shape[:-1])
        te = hk.Linear(self.time_dim * 2)(te)
        te = self.activation(te)
        te = hk.Linear(self.time_dim)(te)
        input = jnp.concatenate((obs, act, te), axis=-1)
        return mlp(self.hidden_sizes, act_dim, self.activation, self.output_activation)(input)
    
@dataclass
@fix_repr
class DACERPolicyNet2(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    time_dim: int = 16
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array, r: jax.Array, t: jax.Array) -> jax.Array:
        act_dim = act.shape[-1]
        te1 = scaled_sinusoidal_encoding(r, dim=self.time_dim, batch_shape=obs.shape[:-1])
        te1 = hk.Linear(self.time_dim * 2)(te1)
        te1 = self.activation(te1)
        te1 = hk.Linear(self.time_dim)(te1)
        te2 = scaled_sinusoidal_encoding(t, dim=self.time_dim, batch_shape=obs.shape[:-1])
        te2 = hk.Linear(self.time_dim * 2)(te2)
        te2 = self.activation(te2)
        te2 = hk.Linear(self.time_dim)(te2)
        input = jnp.concatenate((obs, act, te1, te2), axis=-1)
        return mlp(self.hidden_sizes, act_dim, self.activation, self.output_activation)(input)
    
@dataclass
@fix_repr
class DACERPolicyNet_V(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    time_dim: int = 16
    feature_dim: int = 50
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array, t: jax.Array) -> jax.Array:
        act_dim = act.shape[-1]
        te = scaled_sinusoidal_encoding(t, dim=self.time_dim, batch_shape=obs.shape[:-1])
        te = hk.Linear(self.time_dim * 2)(te)
        te = self.activation(te)
        te = hk.Linear(self.time_dim)(te)
        x = hk.Linear(self.feature_dim)(obs)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.tanh(x)
        input = jnp.concatenate((x, act, te), axis=-1)
        return mlp(self.hidden_sizes, act_dim, self.activation, self.output_activation)(input)
    
@dataclass
@fix_repr
class DACERPolicyNet2_V(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    time_dim: int = 16
    feature_dim: int = 50
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array, r: jax.Array, t: jax.Array) -> jax.Array:
        act_dim = act.shape[-1]
        te1 = scaled_sinusoidal_encoding(r, dim=self.time_dim, batch_shape=obs.shape[:-1])
        te1 = hk.Linear(self.time_dim * 2)(te1)
        te1 = self.activation(te1)
        te1 = hk.Linear(self.time_dim)(te1)
        te2 = scaled_sinusoidal_encoding(t, dim=self.time_dim, batch_shape=obs.shape[:-1])
        te2 = hk.Linear(self.time_dim * 2)(te2)
        te2 = self.activation(te2)
        te2 = hk.Linear(self.time_dim)(te2)
        x = hk.Linear(self.feature_dim)(obs)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.tanh(x)
        input = jnp.concatenate((x, act, te1, te2), axis=-1)
        return mlp(self.hidden_sizes, act_dim, self.activation, self.output_activation)(input)

@dataclass
@fix_repr
class EncoderNet(hk.Module):
    repr_dim : int = 32 * 35 * 35
    name : str = None
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = obs.astype(jnp.float32) / 255.0 - 0.5
        x = jnp.reshape(x, (-1, 9, 84, 84))
        h = convnet(hidden_num=3, output_channels=32, kernel_shape=3, activation=jax.nn.relu)(x)
        h = jnp.reshape(h, [h.shape[0], -1])
        return jnp.squeeze(h)


def mlp(hidden_sizes: Sequence[int], output_size: int, activation: Activation, output_activation: Activation, *, squeeze_output: bool = False) -> Callable[[jax.Array], jax.Array]:
    layers = []
    for hidden_size in hidden_sizes:
        layers += [hk.Linear(hidden_size), activation]
    layers += [hk.Linear(output_size), output_activation]
    if squeeze_output:
        layers.append(partial(jnp.squeeze, axis=-1))
    return hk.Sequential(layers)

def convnet(hidden_num: int, output_channels: int, kernel_shape: int, activation: Activation):
    layers = []
    layers += [hk.Conv2D(output_channels=output_channels, kernel_shape=kernel_shape, stride=2, padding='VALID', data_format='NCHW'), activation]
    for _ in range(hidden_num):
        layers += [hk.Conv2D(output_channels=output_channels, kernel_shape=kernel_shape, stride=1, padding='VALID', data_format='NCHW'), activation]
    return hk.Sequential(layers)

def scaled_sinusoidal_encoding(t: jax.Array, *, dim: int, theta: int = 10000, batch_shape = None) -> jax.Array:
    assert dim % 2 == 0
    if batch_shape is not None:
        assert is_broadcastable(jnp.shape(t), batch_shape)

    scale = 1 / dim ** 0.5
    half_dim = dim // 2
    freq_seq = jnp.arange(half_dim) / half_dim
    inv_freq = theta ** -freq_seq

    emb = jnp.einsum('..., j -> ... j', t, inv_freq)
    emb = jnp.concatenate((
        jnp.sin(emb),
        jnp.cos(emb),
    ), axis=-1)
    emb *= scale

    if batch_shape is not None:
        emb = jnp.broadcast_to(emb, (*batch_shape, dim))

    return emb
