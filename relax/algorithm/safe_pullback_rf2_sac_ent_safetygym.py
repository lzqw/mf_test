from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import numpy as np
import optax

from relax.algorithm.base import Algorithm
from relax.network.safe_pullback_rf2_sac_ent import SafePullbackRF2SACENTNet, SafePullbackRF2Params


class SafePullbackRF2OptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    qp: optax.OptState
    vp: optax.OptState
    policy: optax.OptState
    g: optax.OptState
    log_alpha: optax.OptState


class SafePullbackRF2TrainState(NamedTuple):
    params: SafePullbackRF2Params
    opt_state: SafePullbackRF2OptStates
    step: int
    entropy: float


class SafePullbackRF2SACENTSafetyGym(Algorithm):
    def __init__(self, agent: SafePullbackRF2SACENTNet, params: SafePullbackRF2Params, gamma=0.99, gamma_p=0.99,
                 lr=3e-4, alpha_lr=1e-2, tau=0.005, reward_scale=1.0, sample_k=64, lambda_p=1.0,
                 use_projection_critic=True, fixed_alpha=False, alpha_value=0.01,
                 lambda_p_warmup_steps=100000, lambda_d=0.5,
                 use_frpi_score=False, tau_c=1.0, mu_c=1.0, lambda_f=2.0,
                 use_tn_energy=False, tn_coef=1.0, sigma_n=0.2, sigma_t=1.0,
                 tn_r_min=0.02, tn_r_max=0.20, tn_clip=10.0, kappa_tn=1.0,
                 entropy_reg_mode="legacy", candidate_temp=0.10,
                 beta_normal_entropy=1.0, min_effective_entropy=-20.0, target_effective_entropy=1.0,
                 normal_energy_coef=0.05, target_safe_energy=0.05,
                 safe_iso_coef=0.05, safe_energy_variant="normal_iso", weight_mix=0.05,
                 use_filter_surrogate=False, surrogate_warmup_steps=0, surrogate_loss_coef=1.0, lambda_raw_norm=0.0):
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
        self.entropy_reg_mode = entropy_reg_mode
        self.candidate_temp = candidate_temp
        self.beta_normal_entropy = beta_normal_entropy
        self.min_effective_entropy = min_effective_entropy
        self.target_effective_entropy = target_effective_entropy
        self.normal_energy_coef = normal_energy_coef
        self.target_safe_energy = target_safe_energy
        self.safe_iso_coef = safe_iso_coef
        self.safe_energy_variant = safe_energy_variant
        self.weight_mix = weight_mix
        self.use_filter_surrogate = use_filter_surrogate
        self.surrogate_warmup_steps = surrogate_warmup_steps
        self.surrogate_loss_coef = surrogate_loss_coef
        self.lambda_raw_norm = lambda_raw_norm
        self.optim = optax.adam(lr)
        self.policy_optim = optax.adam(lr)
        self.alpha_optim = optax.adam(alpha_lr)

        self.state = SafePullbackRF2TrainState(
            params=params,
            opt_state=SafePullbackRF2OptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                qp=self.optim.init(params.qp),
                vp=self.optim.init(params.vp),
                policy=self.policy_optim.init(params.policy),
                g=self.optim.init(params.g),
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

            (
                k_next,
                k_qp_policy,
                k_qp_uniform,
                k_vp_policy,
                k2,
            ) = jax.random.split(key, 5)
            def sample_action_with_safe_energy(sample_key, policy_params, obs_batch):
                bsz = obs_batch.shape[0]
                x = 0.5 * jax.random.normal(sample_key, (bsz, raw_action.shape[-1]))
                dt = 1.0 / self.agent.num_timesteps
                normal_e = jnp.zeros((bsz,), dtype=jnp.float32)
                tangent_e = jnp.zeros((bsz,), dtype=jnp.float32)
                iso_e = jnp.zeros((bsz,), dtype=jnp.float32)
                gate_sum = jnp.zeros((bsz,), dtype=jnp.float32)
                residual_sum = jnp.zeros((bsz,), dtype=jnp.float32)
                for k in range(self.agent.num_timesteps):
                    tau = jnp.full((bsz,), k * dt, dtype=jnp.float32)
                    v = self.agent.policy(policy_params, obs_batch, x, tau)
                    e_n, e_t, e_iso, gate, residual = compute_filter_aware_energy_at_points(obs_batch, x, v)
                    normal_e = normal_e + e_n.squeeze(-1) * dt
                    tangent_e = tangent_e + e_t.squeeze(-1) * dt
                    iso_e = iso_e + e_iso.squeeze(-1) * dt
                    gate_sum = gate_sum + gate.squeeze(-1)
                    residual_sum = residual_sum + residual.squeeze(-1)
                    x = x + v * dt
                action = jnp.clip(x, -1.0, 1.0)
                safe_e = jnp.where(
                    self.safe_energy_variant == "normal_tangent",
                    normal_e + tangent_e,
                    normal_e + self.safe_iso_coef * iso_e,
                )
                return action, safe_e, normal_e, tangent_e, iso_e, gate_sum / self.agent.num_timesteps, residual_sum / self.agent.num_timesteps

            entropy = jnp.zeros_like(reward)
            effective_entropy = jnp.zeros_like(reward)
            entropy_total = jnp.zeros_like(reward)
            normal_entropy_penalty = jnp.zeros_like(reward)
            safe_energy_next = jnp.zeros_like(reward)
            safe_n_next = safe_t_next = safe_iso_next = safe_gate_next = safe_residual_next = jnp.zeros_like(reward)
            alpha = jnp.float32(self.alpha_value) if self.fixed_alpha else jnp.exp(p.log_alpha)
            if self.entropy_reg_mode == "flac_tn":
                raw_next_action, entropy = self.agent.get_action_ent(k_next, (p.policy, p.log_alpha, p.q1, p.q2), next_obs)
                raw_next_action = jnp.clip(raw_next_action, -1.0, 1.0)
                safe_energy_next = self.agent.get_qp(p.qp, next_obs, raw_next_action)
            else:
                raw_next_action, entropy = self.agent.get_action_ent(k_next, (p.policy, p.log_alpha, p.q1, p.q2), next_obs)
                raw_next_action = jnp.clip(raw_next_action, -1.0, 1.0)
                entropy_total = entropy
            surrogate_ready = self.use_filter_surrogate and (state.step >= self.surrogate_warmup_steps)
            exec_next_hat = self.agent.get_exec_action_hat(p.g, next_obs, raw_next_action) if surrogate_ready else raw_next_action
            exec_next_hat = jax.lax.stop_gradient(exec_next_hat)
            q1_t = self.agent.q(p.target_q1, next_obs, exec_next_hat)
            q2_t = self.agent.q(p.target_q2, next_obs, exec_next_hat)
            min_q_t = jnp.minimum(q1_t, q2_t)
            if self.entropy_reg_mode == "legacy":
                q_backup = reward * self.reward_scale + (1.0 - done) * self.gamma * (min_q_t + alpha * entropy)
            elif self.entropy_reg_mode == "likelihood_tn":
                t_terminal = jnp.ones((next_obs.shape[0],), dtype=jnp.float32) * 0.999
                v_terminal = self.agent.policy(p.policy, next_obs, raw_next_action, t_terminal)
                e_n_terminal, _, _, _, _ = compute_filter_aware_energy_at_points(next_obs, raw_next_action, v_terminal)
                normal_entropy_penalty = e_n_terminal.squeeze(-1)
                effective_entropy = jnp.maximum(entropy_total - self.beta_normal_entropy * jax.lax.stop_gradient(normal_entropy_penalty), self.min_effective_entropy)
                q_backup = reward * self.reward_scale + (1.0 - done) * self.gamma * (min_q_t + alpha * effective_entropy)
            else:
                q_backup = reward * self.reward_scale + (1.0 - done) * self.gamma * (min_q_t - alpha * jax.lax.stop_gradient(safe_energy_next))

            def qloss(qp, target):
                pred = self.agent.q(qp, obs, exec_action)
                return jnp.mean((pred - jax.lax.stop_gradient(target)) ** 2), pred

            (q1_loss, q1_pred), q1_grads = jax.value_and_grad(qloss, has_aux=True)(p.q1, q_backup)
            (q2_loss, q2_pred), q2_grads = jax.value_and_grad(qloss, has_aux=True)(p.q2, q_backup)
            def gloss(gp):
                exec_hat = self.agent.get_exec_action_hat(gp, obs, raw_action)
                mse = jnp.mean((exec_hat - exec_action) ** 2)
                residual = jnp.mean(jnp.linalg.norm(exec_hat - exec_action, axis=-1))
                return mse, (mse, residual)
            (g_loss, g_aux), g_grads = jax.value_and_grad(gloss, has_aux=True)(p.g)
            g_grads = jax.tree_util.tree_map(lambda x: x * self.surrogate_loss_coef, g_grads)

            vp_next = self.agent.get_vp(p.target_vp, next_obs)
            yp = projection_cost + self.gamma_p * (1.0 - done) * vp_next

            def qploss(qp):
                pred = self.agent.get_qp(qp, obs, raw_action)
                td_loss = jnp.mean((pred - jax.lax.stop_gradient(yp)) ** 2)
                lb = jnp.mean(jax.nn.relu(projection_cost - pred) ** 2)
                total = td_loss + 0.5 * lb
                aux = dict(pred=pred, td_loss=td_loss, l_cf=jnp.float32(0.0), lb=lb)
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
                sample_keys = jax.random.split(k_vp_policy, 8)
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

            # -------- Candidate actions for Q-weighted flow update --------
            batch_size = obs.shape[0]
            act_dim = raw_action.shape[-1]
            k_t, k_local, k_rand, k_flow_noise = jax.random.split(k2, 4)
            K = self.K
            min_k = 8
            k_eff = jnp.maximum(K, min_k)

            n_local = int(max(self.K // 2, 1))
            n_fixed = 1
            n_uniform = int(max(self.K - n_fixed - n_local, 0))
            n_local = int(self.K - n_fixed - n_uniform)

            local_obs = jnp.repeat(obs[:, None, :], n_local, axis=1)
            local_raw = jnp.repeat(raw_action[:, None, :], n_local, axis=1)
            scale = alpha * self.agent.noise_scale
            local_noise = self.agent.directional_noise(k_local, local_obs, scale)
            local_clean = jnp.clip(local_raw + local_noise, -1.0, 1.0)

            uniform_clean = jax.random.uniform(k_rand, (batch_size, n_uniform, act_dim), minval=-1.0, maxval=1.0)
            clean = jnp.concatenate([raw_action[:, None, :], local_clean, uniform_clean], axis=1)
            clean = clean[:, :self.K, :]

            t = jax.random.uniform(k_t, (batch_size, self.K, 1), minval=1e-3, maxval=0.994)
            noise = jax.random.normal(k_flow_noise, clean.shape)
            noisy_rep = t * clean + (1.0 - t) * noise
            obs_rep = jnp.repeat(obs[:, None, :], self.K, axis=1)

            projection_residual_candidates = jnp.sqrt(jnp.maximum(self.agent.get_qp(p.qp, obs_rep, clean), 0.0))

            exec_clean_hat = self.agent.get_exec_action_hat(p.g, obs_rep, clean) if surrogate_ready else clean
            exec_clean_hat = jax.lax.stop_gradient(exec_clean_hat)
            q_reward_raw = jnp.minimum(self.agent.q(p.q1, obs_rep, clean), self.agent.q(p.q2, obs_rep, clean))
            q_reward = jnp.minimum(self.agent.q(p.q1, obs_rep, exec_clean_hat), self.agent.q(p.q2, obs_rep, exec_clean_hat))
            q_proj = self.agent.get_qp(p.qp, obs_rep, clean) if self.use_projection_critic else jnp.zeros_like(q_reward)

            progress_score = jnp.zeros_like(q_reward)
            warmup_steps = jnp.maximum(jnp.float32(self.lambda_p_warmup_steps), 1.0)
            lambda_eff = self.lambda_p * jnp.clip(jnp.float32(state.step) / warmup_steps, 0.0, 1.0)
            score = jax.lax.stop_gradient(q_reward + progress_score - lambda_eff * q_proj - self.lambda_raw_norm * jnp.sum(clean ** 2, axis=-1))

            candidate_temp = jnp.maximum(jnp.float32(self.candidate_temp), 1e-3)
            critic = score / candidate_temp
            w_soft = jnp.exp(critic - jax.nn.logsumexp(critic, axis=1, keepdims=True))
            uniform_w = jnp.ones_like(w_soft) / self.K
            w = (1.0 - self.weight_mix) * w_soft + self.weight_mix * uniform_w

            obs_r = obs_rep.reshape(-1, obs.shape[-1])
            clean_r = clean.reshape(-1, raw_action.shape[-1])
            noisy_r = noisy_rep.reshape(-1, raw_action.shape[-1])
            t_r = t.reshape(-1)
            w_r = w.reshape(-1, 1)
            u_r = clean_r - noisy_r

            weight_entropy = -jnp.mean(jnp.sum(w * jnp.log(w + 1e-8), axis=1))
            top_weight_mean = jnp.mean(jnp.max(w, axis=1))
            far_batch = jnp.mean((projection_residual_candidates > 1e-8).astype(jnp.float32))
            apr_batch = jnp.mean(projection_residual_candidates ** 2)

            tn_energy = tn_normal_energy = tn_tangent_energy = tn_gate_mean = tn_residual_xt_mean = jnp.float32(0.0)

            def ploss(pp):
                denoiser = lambda tt, xx: self.agent.policy(pp, obs_r, xx, tt)
                flow_loss = self.agent.flow.reverse_weighted_p_loss2(denoiser, t_r, noisy_r, jax.lax.stop_gradient(w_r), jax.lax.stop_gradient(u_r))
                qp_penalty = jnp.mean(self.agent.get_qp(p.qp, obs_r, noisy_r))
                residual = jnp.sqrt(jnp.maximum(self.agent.get_qp(p.qp, obs_r, noisy_r), 0.0))
                if self.use_tn_energy:
                    return flow_loss + alpha * self.tn_coef * qp_penalty, (flow_loss, qp_penalty, qp_penalty, jnp.float32(0.0), jnp.float32(0.0), jnp.mean(residual), qp_penalty)
                return flow_loss, (flow_loss, jnp.float32(0.0), qp_penalty, jnp.float32(0.0), jnp.float32(0.0), jnp.mean(residual), qp_penalty)

            (policy_loss, p_aux), policy_grads = jax.value_and_grad(ploss, has_aux=True)(p.policy)
            _, tn_energy, tn_normal_energy, tn_tangent_energy, tn_gate_mean, tn_residual_xt_mean, safe_energy_actor_mean = p_aux

            alpha_loss_val = jnp.float32(0.0)
            if self.fixed_alpha:
                alpha_grads = jax.tree_util.tree_map(jnp.zeros_like, p.log_alpha)
                nloga = p.log_alpha
                ologa = o.log_alpha
            else:
                def aloss(log_alpha):
                    a = jnp.exp(log_alpha)
                    if self.entropy_reg_mode == "legacy":
                        return jnp.mean(log_alpha * (jnp.mean(entropy) - self.agent.target_entropy))
                    if self.entropy_reg_mode == "likelihood_tn":
                        return a * jax.lax.stop_gradient(jnp.mean(effective_entropy) - self.target_effective_entropy)
                    return a * jax.lax.stop_gradient(self.target_safe_energy - jnp.mean(safe_energy_actor_mean))
                alpha_loss_val, alpha_grads = jax.value_and_grad(aloss)(p.log_alpha)

            def apply(optim, params, grads, st):
                upd, ns = optim.update(grads, st)
                return optax.apply_updates(params, upd), ns

            nq1, oq1 = apply(self.optim, p.q1, q1_grads, o.q1)
            nq2, oq2 = apply(self.optim, p.q2, q2_grads, o.q2)
            nqp, oqp = apply(self.optim, p.qp, qp_grads, o.qp)
            nvp, ovp = apply(self.optim, p.vp, vp_grads, o.vp)
            npol, opol = apply(self.policy_optim, p.policy, policy_grads, o.policy)
            ng, og = apply(self.optim, p.g, g_grads, o.g)
            if self.fixed_alpha:
                nloga, ologa = p.log_alpha, o.log_alpha
            else:
                nloga, ologa = apply(self.alpha_optim, p.log_alpha, alpha_grads, o.log_alpha)

            t_q1 = optax.incremental_update(nq1, p.target_q1, self.tau)
            t_q2 = optax.incremental_update(nq2, p.target_q2, self.tau)
            t_vp = optax.incremental_update(nvp, p.target_vp, self.tau)
            t_pol = optax.incremental_update(npol, p.target_policy, self.tau)

            ns = SafePullbackRF2TrainState(
                params=SafePullbackRF2Params(nq1, nq2, t_q1, t_q2, nqp, nvp, t_vp, npol, t_pol, ng, nloga),
                opt_state=SafePullbackRF2OptStates(oq1, oq2, oqp, ovp, opol, og, ologa),
                step=state.step + 1,
                entropy=jnp.mean(entropy),
            )
            info = dict(q1_loss=q1_loss, q2_loss=q2_loss, qp_loss=qp_loss, vp_loss=vp_loss,
                        policy_loss=policy_loss, alpha=(jnp.float32(self.alpha_value) if self.fixed_alpha else jnp.exp(nloga)),
                        alpha_loss=(jnp.float32(0.0) if self.fixed_alpha else alpha_loss_val),
                        q_reward_mean=jnp.mean(q_reward), q_projection_mean=jnp.mean(q_proj),
                        projection_cost_batch=jnp.mean(projection_cost),
                        safe_pullback_score_mean=jnp.mean(score),
                        qp_td_loss=qp_aux["td_loss"], qp_cf_loss=qp_aux["l_cf"], qp_lb_loss=qp_aux["lb"],
                        lambda_eff=lambda_eff, FAR_batch=far_batch, APR_batch=apr_batch,
                        candidate_q_reward_mean=jnp.mean(q_reward),
                        candidate_q_projection_mean=jnp.mean(q_proj),
                        projection_residual_candidate_mean=jnp.mean(projection_residual_candidates),
                        projection_residual_candidate_max=jnp.max(projection_residual_candidates),
                        progress_score_mean=jnp.mean(progress_score),
                        progress_score_max=jnp.max(progress_score),
                        weight_entropy=weight_entropy,
                        top_weight_mean=top_weight_mean,
                        clean_candidate_std=jnp.mean(jnp.std(clean, axis=1)),
                        clean_candidate_norm_mean=jnp.mean(jnp.linalg.norm(clean, axis=-1)),
                        exec_candidate_hat_norm_mean=jnp.mean(jnp.linalg.norm(exec_clean_hat, axis=-1)),
                        raw_action_batch_std=jnp.mean(jnp.std(raw_action, axis=0)),
                        g_loss=self.surrogate_loss_coef * g_loss, g_exec_mse=g_aux[0], g_exec_residual_mean=g_aux[1],
                        q_reward_raw_mean=jnp.mean(q_reward_raw), q_reward_exec_hat_mean=jnp.mean(q_reward),
                        tangent_candidate_fraction=jnp.float32((2.0 + n_local) / self.K),
                        uniform_candidate_fraction=jnp.float32(n_uniform / self.K),
                        goal_candidate_fraction=jnp.float32(1.0 / self.K),
                        tn_energy=tn_energy, tn_normal_energy=tn_normal_energy,
                        tn_tangent_energy=tn_tangent_energy, tn_gate_mean=tn_gate_mean,
                        tn_residual_xt_mean=tn_residual_xt_mean,
                        flow_loss=p_aux[0], candidate_temp=jnp.float32(self.candidate_temp),
                        entropy_reg_mode_id=jnp.float32(
                            0.0 if self.entropy_reg_mode == "legacy"
                            else 1.0 if self.entropy_reg_mode == "likelihood_tn"
                            else 2.0
                        ),
                        entropy_total_mean=jnp.mean(entropy_total),
                        effective_entropy_mean=jnp.mean(effective_entropy),
                        normal_entropy_penalty_mean=jnp.mean(normal_entropy_penalty),
                        target_effective_entropy=jnp.float32(self.target_effective_entropy),
                        safe_energy_actor_mean=jnp.mean(safe_energy_actor_mean),
                        safe_energy_next_mean=jnp.mean(safe_energy_next),
                        safe_energy_normal_mean=jnp.mean(safe_n_next),
                        safe_energy_tangent_mean=jnp.mean(safe_t_next),
                        safe_energy_iso_mean=jnp.mean(safe_iso_next),
                        safe_energy_gate_mean=jnp.mean(safe_gate_next),
                        safe_energy_residual_mean=jnp.mean(safe_residual_next),
                        target_safe_energy=jnp.float32(self.target_safe_energy))
            return ns, info

        self._update = _update

    def update(self, key, data):
        self.state, info = self._update(key, self.state, data)
        return {k: float(v) for k, v in info.items()}

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self.agent.get_action(key, (self.state.params.policy, self.state.params.log_alpha, self.state.params.q1, self.state.params.q2), obs), dtype=np.float32)
