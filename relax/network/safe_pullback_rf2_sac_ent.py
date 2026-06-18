from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import haiku as hk
import math
import jax.scipy.stats

from relax.network.blocks import Activation, DACERPolicyNet, QNet
from relax.utils.flow import OTFlow
from relax.utils.jax_utils import random_key_from_data


class SafePullbackRF2Params(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    qp: hk.Params
    vp: hk.Params
    target_vp: hk.Params
    policy: hk.Params
    target_policy: hk.Params
    g: hk.Params
    log_alpha: jax.Array


@dataclass
class SafePullbackRF2SACENTNet:
    q: Callable
    qp: Callable
    vp: Callable
    policy: Callable
    g: Callable
    num_timesteps: int
    num_ent_timesteps: int
    num_timesteps_test: int
    act_dim: int
    num_particles: int
    target_entropy: float
    noise_scale: float
    noise_schedule: str
    alpha_value: float
    fixed_alpha: bool
    use_directional_noise: bool

    @property
    def flow(self):
        return OTFlow(self.num_timesteps)

    @property
    def flow_test(self):
        return OTFlow(self.num_timesteps_test)

    def get_qp(self, qp_params, obs, raw_action):
        return self.qp(qp_params, obs, raw_action)

    def get_vp(self, vp_params, obs):
        return self.vp(vp_params, obs)

    def get_exec_action_hat(self, g_params, obs, raw_action):
        return self.g(g_params, obs, raw_action)

    def _directional_obs_features(self, obs):
        if obs.shape[-1] == 10:
            goal_vec = -obs[..., 2:4]
            normal_vec = obs[..., 4:6]
            d_obs = obs[..., 6:7]
            return goal_vec, normal_vec, d_obs
        if obs.shape[-1] == 16:
            goal_vec = obs[..., 2:4]
            obstacle_block = obs[..., 7:]
            obstacle_block = obstacle_block.reshape(*obs.shape[:-1], 3, 3)
            rel = obstacle_block[..., :2]
            clear = obstacle_block[..., 2]
            idx = jnp.argmin(clear, axis=-1)
            normal_vec = jnp.take_along_axis(rel, idx[..., None, None], axis=-2).squeeze(axis=-2)
            d_obs = jnp.min(clear, axis=-1, keepdims=True)
            return goal_vec, normal_vec, d_obs
        goal_vec = obs[..., 2:4]
        normal_vec = obs[..., 4:6]
        d_obs = jnp.linalg.norm(normal_vec, axis=-1, keepdims=True)
        return goal_vec, normal_vec, d_obs

    def directional_noise(self, key, obs, base_std, return_components=False):
        if (not self.use_directional_noise) or self.act_dim != 2 or obs.shape[-1] < 8:
            noise = base_std * jax.random.normal(key, (*obs.shape[:-1], self.act_dim))
            if return_components:
                zero = jnp.zeros_like(noise)
                return noise, (zero, zero, zero, zero)
            return noise

        goal_vec, normal_vec, d_obs = self._directional_obs_features(obs)

        e_goal = goal_vec / (jnp.linalg.norm(goal_vec, axis=-1, keepdims=True) + 1e-6)
        e_normal = normal_vec / (jnp.linalg.norm(normal_vec, axis=-1, keepdims=True) + 1e-6)
        e_tangent = jnp.concatenate([-e_normal[..., 1:2], e_normal[..., 0:1]], axis=-1)

        near_scale = 0.5
        near = jnp.exp(-jnp.maximum(d_obs, 0.0) / near_scale)
        sigma_goal = base_std * (1.0 - 0.5 * near)
        sigma_tangent = base_std * (0.3 + 1.7 * near)
        sigma_normal = base_std * (1.0 - 0.8 * near)

        k1, k2, k3, k4 = jax.random.split(key, 4)
        eta_g = jax.random.normal(k1, (*obs.shape[:-1], 1))
        eta_t = jax.random.normal(k2, (*obs.shape[:-1], 1))
        eta_n = jax.random.normal(k3, (*obs.shape[:-1], 1))
        sign = jnp.where(jax.random.bernoulli(k4, 0.5, (*obs.shape[:-1], 1)), 1.0, -1.0)
        eta_n_safe = jnp.maximum(eta_n, -0.2)

        g_comp = sigma_goal * eta_g * e_goal
        t_comp = sigma_tangent * eta_t * sign * e_tangent
        n_comp = sigma_normal * eta_n_safe * e_normal
        noise = g_comp + t_comp + n_comp
        if return_components:
            return noise, (g_comp, t_comp, n_comp, near)
        return noise

    def get_action(self, key, policy_tuple, obs):
        policy_params, log_alpha, q1_params, q2_params = policy_tuple

        def model_fn(t, x):
            return self.policy(policy_params, obs, x, t)

        sample_key, noise_key = jax.random.split(key)
        act = self.flow.p_sample(sample_key, model_fn, (*obs.shape[:-1], self.act_dim)).clip(-1, 1)
        scale = jnp.float32(self.alpha_value) if self.fixed_alpha else jnp.exp(log_alpha)
        base_std = scale * self.noise_scale
        noise = self.directional_noise(noise_key, obs, base_std)
        return jnp.clip(act + noise, -1.0, 1.0)

    def get_action_ent(self, key, policy_tuple, obs):
        policy_params, log_alpha, q1_params, q2_params = policy_tuple
        sample_key, noise_key, ent_key = jax.random.split(key, 3)

        def model_fn(t, x):
            return self.policy(policy_params, obs, x, t)

        act = self.flow.p_sample(sample_key, model_fn, (*obs.shape[:-1], self.act_dim)).clip(-1, 1)
        log_prob = self.compute_log_likelihood(ent_key, policy_params, obs, act)
        entropy = -log_prob
        scale = jnp.float32(self.alpha_value) if self.fixed_alpha else jnp.exp(log_alpha)
        base_std = scale * self.noise_scale
        noise = self.directional_noise(noise_key, obs, base_std)
        return jnp.clip(act + noise, -1.0, 1.0), entropy

    def compute_log_likelihood(self, key, policy_params, obs, act):
        def model_fn(t, x):
            return self.policy(policy_params, obs, x, t)

        def log_p0(z):
            return jax.scipy.stats.norm.logpdf(z).sum(axis=-1)

        z = jax.random.normal(key, act.shape)

        def ode(state, t):
            f_t, _ = state
            u_t_fn = lambda x: model_fn(t, x)
            _, vjp_fn = jax.vjp(u_t_fn, f_t)
            vjp_z = vjp_fn(z)[0]
            trace_term = jnp.sum(vjp_z * z, axis=-1)
            return u_t_fn(f_t), -trace_term

        n = self.num_ent_timesteps
        dt = -1.0 / n

        def step(state, t):
            df, dg = ode(state, t)
            f, g = state
            return (f + df * dt, g + dg * dt), None

        timesteps = jnp.linspace(1.0, 1.0 / n, n)
        final, _ = jax.lax.scan(step, (act, jnp.zeros(act.shape[:-1])), timesteps)
        f0, g0 = final
        return log_p0(f0) - g0


def create_safe_pullback_rf2_sac_ent_net(
    key,
    obs_dim,
    act_dim,
    hidden_sizes,
    diffusion_hidden_sizes,
    activation=jax.nn.relu,
    num_timesteps=20,
    num_ent_timesteps=20,
    num_timesteps_test=20,
    num_particles=32,
    noise_scale=0.05,
    target_entropy_scale=0.9,
    alpha_value=0.01,
    fixed_alpha=True,
    init_alpha=0.01,
    use_directional_noise=True,
):
    q = hk.without_apply_rng(hk.transform(lambda obs, act: QNet(hidden_sizes, activation)(obs, act)))
    qp = hk.without_apply_rng(hk.transform(lambda obs, act: QNet(hidden_sizes, activation)(obs, act)))
    vp = hk.without_apply_rng(hk.transform(lambda obs: hk.nets.MLP((*hidden_sizes, 1), activation=activation)(obs).squeeze(-1)))
    policy = hk.without_apply_rng(hk.transform(lambda obs, act, t: DACERPolicyNet(diffusion_hidden_sizes, activation)(obs, act, t)))
    g = hk.without_apply_rng(hk.transform(lambda obs, act: jnp.tanh(hk.nets.MLP((*hidden_sizes, act_dim), activation=activation)(jnp.concatenate([obs, act], axis=-1)))))

    @jax.jit
    def init(key, obs, act):
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        q1 = q.init(k1, obs, act)
        q2 = q.init(k2, obs, act)
        qp_p = qp.init(k3, obs, act)
        vp_p = vp.init(k4, obs)
        pol = policy.init(k5, obs, act, 0)
        gp = g.init(k6, obs, act)
        return SafePullbackRF2Params(q1, q2, q1, q2, qp_p, vp_p, vp_p, pol, pol, gp, jnp.array(math.log(init_alpha), dtype=jnp.float32))

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)
    net = SafePullbackRF2SACENTNet(q=q.apply, qp=qp.apply, vp=vp.apply, policy=policy.apply, g=g.apply,
                                   num_timesteps=num_timesteps, num_ent_timesteps=num_ent_timesteps,
                                   num_timesteps_test=num_timesteps_test, act_dim=act_dim,
                                   target_entropy=-act_dim * target_entropy_scale, num_particles=num_particles,
                                   noise_scale=noise_scale, noise_schedule='linear',
                                   alpha_value=alpha_value, fixed_alpha=fixed_alpha,
                                   use_directional_noise=use_directional_noise)
    return net, params
