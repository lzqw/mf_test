from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import numpy as np
import optax

from relax.algorithm.base import Algorithm
from relax.network.safe_pullback_rf2_sac_ent import SafePullbackRF2SACENTNet, SafePullbackRF2Params
from relax.safety.obstacle_navigation_filter import ObstacleNavConfig, make_action_grid, project_action_jax_batched


class SafePullbackRF2OptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    qp: optax.OptState
    vp: optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState


class SafePullbackRF2TrainState(NamedTuple):
    params: SafePullbackRF2Params
    opt_state: SafePullbackRF2OptStates
    step: int
    entropy: float


class SafePullbackRF2SACENT(Algorithm):
    def __init__(self, agent: SafePullbackRF2SACENTNet, params: SafePullbackRF2Params, gamma=0.99, gamma_p=0.99,
                 lr=3e-4, alpha_lr=1e-2, tau=0.005, reward_scale=1.0, sample_k=64, lambda_p=1.0,
                 use_projection_critic=True, fixed_alpha=False, alpha_value=0.01,
                 lambda_p_warmup_steps=100000, lambda_d=0.5,
                 use_frpi_score=False, tau_c=1.0, mu_c=1.0, lambda_f=2.0,
                 use_tn_energy=False, tn_coef=1.0, sigma_n=0.2, sigma_t=1.0,
                 tn_r_min=0.02, tn_r_max=0.20, tn_clip=10.0, kappa_tn=1.0):
        self.agent = agent
        self.gamma = gamma
        self.gamma_p = gamma_p
        self.tau = tau
        self.reward_scale = reward_scale
        self.K = sample_k
        self.lambda_p = lambda_p
        self.use_projection_critic = use_projection_critic
        self.lambda_p_warmup_steps = lambda_p_warmup_steps
        self.lambda_d = lambda_d
        self.fixed_alpha = fixed_alpha
        self.alpha_value = alpha_value
        self.use_frpi_score = use_frpi_score
        self.tau_c = tau_c
        self.mu_c = mu_c
        self.lambda_f = lambda_f
        self.use_tn_energy = use_tn_energy
        self.tn_coef = tn_coef
        self.sigma_n = sigma_n
        self.sigma_t = sigma_t
        self.tn_r_min = tn_r_min
        self.tn_r_max = tn_r_max
        self.tn_clip = tn_clip
        self.kappa_tn = kappa_tn
        self.optim = optax.adam(lr)
        self.policy_optim = optax.adam(lr)
        self.alpha_optim = optax.adam(alpha_lr)
        self.cfg = ObstacleNavConfig()
        self.action_grid = jnp.asarray(make_action_grid(61))

        self.state = SafePullbackRF2TrainState(
            params=params,
            opt_state=SafePullbackRF2OptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                qp=self.optim.init(params.qp),
                vp=self.optim.init(params.vp),
                policy=self.policy_optim.init(params.policy),
                log_alpha=self.alpha_optim.init(params.log_alpha),
            ),
            step=0,
            entropy=0.0,
        )

        @jax.jit
        def _update(key, state, data):
            obs, exec_action, raw_action = data.obs, data.action, data.raw_action
            reward, next_obs, done = data.reward, data.next_obs, data.done
            projection_cost = data.projection_cost
            p = state.params
            o = state.opt_state

            k1, k2, k3 = jax.random.split(key, 3)
            raw_next_action, entropy = self.agent.get_action_ent(k1, (p.policy, p.log_alpha, p.q1, p.q2), next_obs)
            raw_next_action = jnp.clip(raw_next_action, -1.0, 1.0)
            exec_next_action, _, _ = project_action_jax_batched(next_obs, raw_next_action, self.action_grid, self.cfg)

            q1_t = self.agent.q(p.target_q1, next_obs, exec_next_action)
            q2_t = self.agent.q(p.target_q2, next_obs, exec_next_action)
            alpha = jnp.float32(self.alpha_value) if self.fixed_alpha else jnp.exp(p.log_alpha)
            q_backup = reward * self.reward_scale + (1.0 - done) * self.gamma * (jnp.minimum(q1_t, q2_t) - alpha * entropy)

            def qloss(qp, target):
                pred = self.agent.q(qp, obs, exec_action)
                return jnp.mean((pred - jax.lax.stop_gradient(target)) ** 2), pred

            (q1_loss, q1_pred), q1_grads = jax.value_and_grad(qloss, has_aux=True)(p.q1, q_backup)
            (q2_loss, q2_pred), q2_grads = jax.value_and_grad(qloss, has_aux=True)(p.q2, q_backup)

            vp_next = self.agent.get_vp(p.target_vp, next_obs)
            yp = projection_cost + self.gamma_p * (1.0 - done) * vp_next

            def qploss(qp):
                pred = self.agent.get_qp(qp, obs, raw_action)
                td_loss = jnp.mean((pred - jax.lax.stop_gradient(yp)) ** 2)
                policy_keys = jax.random.split(k3, 8)
                cf_policy = jax.vmap(
                    lambda sk: self.agent.get_action(sk, (p.policy, p.log_alpha, p.q1, p.q2), obs)
                )(policy_keys)
                cf_policy = jnp.swapaxes(cf_policy, 0, 1)
                cf_uniform = jax.random.uniform(k2, (obs.shape[0], 8, raw_action.shape[1]), minval=-1.0, maxval=1.0)
                cf_actions = jnp.concatenate([cf_policy, cf_uniform], axis=1)
                cf_obs = jnp.repeat(obs[:, None, :], cf_actions.shape[1], axis=1)
                cf_exec, _, _ = project_action_jax_batched(cf_obs, cf_actions, self.action_grid, self.cfg)
                d_cf = jnp.sum((cf_actions - cf_exec) ** 2, axis=-1)
                q_cf = self.agent.get_qp(qp, cf_obs, cf_actions)
                l_cf = jnp.mean((q_cf - jax.lax.stop_gradient(d_cf)) ** 2)
                lb = jnp.mean(jax.nn.relu(projection_cost - pred) ** 2)
                total = td_loss + 0.5 * l_cf + 0.5 * lb
                aux = dict(pred=pred, td_loss=td_loss, l_cf=l_cf, lb=lb)
                return total, aux

            if self.use_projection_critic:
                (qp_loss, qp_aux), qp_grads = jax.value_and_grad(qploss, has_aux=True)(p.qp)
                qp_pred = qp_aux["pred"]
            else:
                qp_loss, qp_pred = jnp.float32(0.0), jnp.zeros_like(reward)
                qp_grads = jax.tree_util.tree_map(jnp.zeros_like, p.qp)
                qp_aux = dict(td_loss=jnp.float32(0.0), l_cf=jnp.float32(0.0), lb=jnp.float32(0.0))

            def vploss(vp):
                pred = self.agent.get_vp(vp, obs)
                sample_keys = jax.random.split(k2, 8)
                policy_actions = jax.vmap(
                    lambda sk: self.agent.get_action(sk, (p.policy, p.log_alpha, p.q1, p.q2), obs)
                )(sample_keys)
                policy_actions = jnp.swapaxes(policy_actions, 0, 1)
                policy_obs = jnp.repeat(obs[:, None, :], 8, axis=1)
                target = jax.lax.stop_gradient(jnp.mean(self.agent.get_qp(p.qp, policy_obs, policy_actions), axis=1))
                return jnp.mean((pred - target) ** 2), pred

            if self.use_projection_critic:
                (vp_loss, vp_pred), vp_grads = jax.value_and_grad(vploss, has_aux=True)(p.vp)
            else:
                vp_loss, vp_pred = jnp.float32(0.0), jnp.zeros_like(reward)
                vp_grads = jax.tree_util.tree_map(jnp.zeros_like, p.vp)

            obs_rep = jnp.repeat(obs[:, None, :], self.K, axis=1)
            obs_flat = obs_rep.reshape(-1, obs.shape[-1])
            clean_model_fn = lambda tt, xx: self.agent.policy(p.target_policy, obs_flat, xx, tt)
            clean_flat = jnp.clip(self.agent.flow.p_sample(k1, clean_model_fn, (obs_flat.shape[0], raw_action.shape[-1])), -1.0, 1.0)
            clean = jax.lax.stop_gradient(clean_flat.reshape(obs.shape[0], self.K, raw_action.shape[-1]))
            noise = jax.random.normal(k2, shape=clean.shape)
            t = jax.random.uniform(k3, (obs.shape[0], self.K, 1), minval=1e-3, maxval=0.994)
            noisy = jnp.clip(t * clean + (1 - t) * noise, -1.0, 1.0)
            u = clean - noise
            exec_clean, _, _ = project_action_jax_batched(obs_rep, clean, self.action_grid, self.cfg)

            q_reward = jnp.minimum(self.agent.q(p.q1, obs_rep, exec_clean), self.agent.q(p.q2, obs_rep, exec_clean))
            q_proj = self.agent.get_qp(p.qp, obs_rep, clean) if self.use_projection_critic else jnp.zeros_like(q_reward)
            residual_clean = jnp.linalg.norm(exec_clean - clean, axis=-1)
            d_proj = residual_clean ** 2
            far_batch = jnp.mean((d_proj > 1e-8).astype(jnp.float32))
            apr_batch = jnp.mean(d_proj)
            lambda_p_current = self.lambda_p * jnp.minimum(1.0, state.step / jnp.maximum(self.lambda_p_warmup_steps, 1))

            c_cost = q_proj + self.mu_c * d_proj
            compatibility = jnp.clip(jnp.exp(-c_cost / jnp.maximum(self.tau_c, 1e-6)), 1e-6, 1.0)
            a_r = q_reward - jnp.mean(q_reward, axis=1, keepdims=True)
            a_f = compatibility - jnp.mean(compatibility, axis=1, keepdims=True)
            frpi_score = compatibility * a_r + self.lambda_f * a_f
            base_score = q_reward - lambda_p_current * q_proj - self.lambda_d * d_proj
            score = jax.lax.stop_gradient(jnp.where(self.use_frpi_score, frpi_score, base_score))
            critic = score / jnp.maximum(alpha, 1e-3)
            w = jax.nn.softmax(jax.lax.stop_gradient(critic), axis=1)

            obs_r = obs_flat
            clean_r = clean.reshape(-1, raw_action.shape[-1])
            noisy_r = noisy.reshape(-1, raw_action.shape[-1])
            t_r = t.reshape(-1)
            w_r = w.reshape(-1, 1)
            u_r = u.reshape(-1, raw_action.shape[-1])

            tn_energy = tn_normal_energy = tn_tangent_energy = tn_gate_mean = tn_residual_xt_mean = jnp.float32(0.0)

            def ploss(pp):
                denoiser = lambda tt, xx: self.agent.policy(pp, obs_r, xx, tt)
                flow_loss = self.agent.flow.reverse_weighted_p_loss2(denoiser, t_r, noisy_r, jax.lax.stop_gradient(w_r), jax.lax.stop_gradient(u_r))
                if self.use_tn_energy:
                    v_pred = self.agent.policy(pp, obs_r, noisy_r, t_r)
                    exec_xt, _, _ = project_action_jax_batched(obs_r, noisy_r, self.action_grid, self.cfg)
                    exec_xt = jax.lax.stop_gradient(exec_xt)
                    delta = jax.lax.stop_gradient(exec_xt - noisy_r)
                    r = jnp.linalg.norm(delta, axis=-1, keepdims=True)
                    n = delta / (r + 1e-6)
                    gate_denom = jnp.maximum(self.tn_r_max - self.tn_r_min, 1e-6)
                    gate = jnp.clip((r - self.tn_r_min) / gate_denom, 0.0, 1.0)
                    b = self.kappa_tn * delta
                    vn = jnp.sum(n * (v_pred - b), axis=-1, keepdims=True)
                    normal_sq = jnp.minimum(vn ** 2, self.tn_clip)
                    e_n = gate * normal_sq / (2.0 * jnp.maximum(self.sigma_n ** 2, 1e-6))
                    v_dot_n = jnp.sum(n * v_pred, axis=-1, keepdims=True)
                    v_norm2 = jnp.sum(v_pred ** 2, axis=-1, keepdims=True)
                    tangent_sq = jnp.maximum(v_norm2 - v_dot_n ** 2, 0.0)
                    e_t = tangent_sq / (2.0 * jnp.maximum(self.sigma_t ** 2, 1e-6))
                    tne = jnp.mean(e_n + e_t)
                    return flow_loss + alpha * self.tn_coef * tne, (flow_loss, tne, jnp.mean(e_n), jnp.mean(e_t), jnp.mean(gate), jnp.mean(r))
                return flow_loss, (flow_loss, jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0))

            (policy_loss, p_aux), policy_grads = jax.value_and_grad(ploss, has_aux=True)(p.policy)
            _, tn_energy, tn_normal_energy, tn_tangent_energy, tn_gate_mean, tn_residual_xt_mean = p_aux

            def aloss(log_alpha):
                return jnp.mean(log_alpha * (jnp.mean(entropy) - self.agent.target_entropy))

            alpha_grads = jax.grad(aloss)(p.log_alpha)

            def apply(optim, params, grads, st):
                upd, ns = optim.update(grads, st)
                return optax.apply_updates(params, upd), ns

            nq1, oq1 = apply(self.optim, p.q1, q1_grads, o.q1)
            nq2, oq2 = apply(self.optim, p.q2, q2_grads, o.q2)
            nqp, oqp = apply(self.optim, p.qp, qp_grads, o.qp)
            nvp, ovp = apply(self.optim, p.vp, vp_grads, o.vp)
            npol, opol = apply(self.policy_optim, p.policy, policy_grads, o.policy)
            nloga, ologa = apply(self.alpha_optim, p.log_alpha, alpha_grads, o.log_alpha)

            t_q1 = optax.incremental_update(nq1, p.target_q1, self.tau)
            t_q2 = optax.incremental_update(nq2, p.target_q2, self.tau)
            t_vp = optax.incremental_update(nvp, p.target_vp, self.tau)
            t_pol = optax.incremental_update(npol, p.target_policy, self.tau)

            ns = SafePullbackRF2TrainState(
                params=SafePullbackRF2Params(nq1, nq2, t_q1, t_q2, nqp, nvp, t_vp, npol, t_pol, nloga),
                opt_state=SafePullbackRF2OptStates(oq1, oq2, oqp, ovp, opol, ologa),
                step=state.step + 1,
                entropy=jnp.mean(entropy),
            )
            info = dict(q1_loss=q1_loss, q2_loss=q2_loss, qp_loss=qp_loss, vp_loss=vp_loss,
                        policy_loss=policy_loss, alpha=jnp.exp(nloga),
                        q_reward_mean=jnp.mean(q_reward), q_projection_mean=jnp.mean(q_proj),
                        projection_cost_batch=jnp.mean(projection_cost),
                        safe_pullback_score_mean=jnp.mean(score),
                        qp_td_loss=qp_aux["td_loss"], qp_cf_loss=qp_aux["l_cf"], qp_lb_loss=qp_aux["lb"],
                        lambda_p_current=lambda_p_current, FAR_batch=far_batch, APR_batch=apr_batch,
                        candidate_q_reward_mean=jnp.mean(q_reward),
                        candidate_q_projection_mean=jnp.mean(q_proj),
                        candidate_projection_residual_mean=jnp.mean(residual_clean),
                        frpi_score_mean=jnp.mean(frpi_score),
                        compatibility_mean=jnp.mean(compatibility),
                        compatibility_min=jnp.min(compatibility),
                        compatibility_max=jnp.max(compatibility),
                        tn_energy=tn_energy, tn_normal_energy=tn_normal_energy,
                        tn_tangent_energy=tn_tangent_energy, tn_gate_mean=tn_gate_mean,
                        tn_residual_xt_mean=tn_residual_xt_mean)
            return ns, info

        self._update = _update

    def update(self, key, data):
        self.state, info = self._update(key, self.state, data)
        return {k: float(v) for k, v in info.items()}

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self.agent.get_action(key, (self.state.params.policy, self.state.params.log_alpha, self.state.params.q1, self.state.params.q2), obs), dtype=np.float32)
