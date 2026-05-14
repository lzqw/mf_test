from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import numpy as np
import optax

from relax.algorithm.base import Algorithm
from relax.network.safe_pullback_rf2_sac_ent import SafePullbackRF2SACENTNet, SafePullbackRF2Params
from relax.safety.humanoidbench_filter import project_action_jax_humanoid


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


class SafePullbackRF2SACENTHumanoid(Algorithm):
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
                 safe_iso_coef=0.05, safe_energy_variant="normal_iso", weight_mix=0.05, residual_radius=0.35, action_limit=1.0,
                 use_goal_candidate=False, high_level_max_delta=0.1,
                 policy_score_mode="linear_penalty", compat_tau=0.5, compat_mu=1.0, compat_lambda_f=2.0,
                 trigger_threshold=0.25):
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
        self.use_goal_candidate = use_goal_candidate
        self.high_level_max_delta = high_level_max_delta
        self.policy_score_mode = policy_score_mode
        self.compat_tau = compat_tau
        self.compat_mu = compat_mu
        self.compat_lambda_f = compat_lambda_f
        self.trigger_threshold = trigger_threshold
        self.optim = optax.adam(lr)
        self.policy_optim = optax.adam(lr)
        self.alpha_optim = optax.adam(alpha_lr)
        self.residual_radius = residual_radius
        self.action_limit = action_limit

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

            (
                k_next,
                k_qp_policy,
                k_qp_uniform,
                k_vp_policy,
                k2,
            ) = jax.random.split(key, 5)
            def compute_filter_aware_energy_at_points(obs_point, x_point, v_pred):
                x_for_filter = jnp.clip(x_point, -1.0, 1.0)
                exec_x, _, _ = project_action_jax_humanoid(x_for_filter, prior_action=None, residual_radius=self.residual_radius, action_limit=self.action_limit)
                exec_x = jax.lax.stop_gradient(exec_x)
                delta = jax.lax.stop_gradient(exec_x - x_for_filter)
                residual = jnp.linalg.norm(delta, axis=-1, keepdims=True)
                n = delta / (residual + 1e-6)
                gate = jnp.clip((residual - self.tn_r_min) / jnp.maximum(self.tn_r_max - self.tn_r_min, 1e-6), 0.0, 1.0)
                gate = jax.lax.stop_gradient(gate)
                b = self.kappa_tn * delta
                vn = jnp.sum(n * (v_pred - b), axis=-1, keepdims=True)
                normal_sq = jnp.minimum(vn ** 2, self.tn_clip)
                e_n = gate * normal_sq / (2.0 * jnp.maximum(self.sigma_n ** 2, 1e-6))
                v_dot_n = jnp.sum(n * v_pred, axis=-1, keepdims=True)
                v_norm2 = jnp.sum(v_pred ** 2, axis=-1, keepdims=True)
                tangent_sq = jnp.maximum(v_norm2 - v_dot_n ** 2, 0.0)
                e_t = tangent_sq / (2.0 * jnp.maximum(self.sigma_t ** 2, 1e-6))
                e_iso = 0.5 * v_norm2
                return e_n, e_t, e_iso, gate, residual

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
                raw_next_action, safe_energy_next, safe_n_next, safe_t_next, safe_iso_next, safe_gate_next, safe_residual_next = sample_action_with_safe_energy(k_next, p.policy, next_obs)
            else:
                raw_next_action, entropy = self.agent.get_action_ent(k_next, (p.policy, p.log_alpha, p.q1, p.q2), next_obs)
                raw_next_action = jnp.clip(raw_next_action, -1.0, 1.0)
                entropy_total = entropy
            exec_next_action, _, _ = project_action_jax_humanoid(raw_next_action, prior_action=None, residual_radius=self.residual_radius, action_limit=self.action_limit)

            q1_t = self.agent.q(p.target_q1, next_obs, exec_next_action)
            q2_t = self.agent.q(p.target_q2, next_obs, exec_next_action)
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

            vp_next = self.agent.get_vp(p.target_vp, next_obs)
            yp = projection_cost + self.gamma_p * (1.0 - done) * vp_next

            def qploss(qp):
                pred = self.agent.get_qp(qp, obs, raw_action)
                td_loss = jnp.mean((pred - jax.lax.stop_gradient(yp)) ** 2)
                policy_keys = jax.random.split(k_qp_policy, 8)
                cf_policy = jax.vmap(
                    lambda sk: self.agent.get_action(sk, (p.policy, p.log_alpha, p.q1, p.q2), obs)
                )(policy_keys)
                cf_policy = jnp.swapaxes(cf_policy, 0, 1)
                cf_uniform = jax.random.uniform(k_qp_uniform, (obs.shape[0], 8, raw_action.shape[1]), minval=-1.0, maxval=1.0)
                cf_actions = jnp.concatenate([cf_policy, cf_uniform], axis=1)
                cf_obs = jnp.repeat(obs[:, None, :], cf_actions.shape[1], axis=1)
                cf_exec, _, _ = project_action_jax_humanoid(cf_actions, prior_action=None, residual_radius=self.residual_radius, action_limit=self.action_limit)
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

            use_goal_candidate = bool(self.use_goal_candidate) and (act_dim == 3) and (obs.shape[-1] >= 9)
            n_fixed = 3 if use_goal_candidate else 2
            n_local = int(max(K // 2, 1))
            n_uniform = int(max(K - n_fixed - n_local, 0))

            a_zero = jnp.zeros_like(raw_action)
            if use_goal_candidate:
                goal_minus_last_target = obs[..., -6:-3]
                max_delta = jnp.maximum(jnp.float32(self.high_level_max_delta), 1e-6)
                a_goal = jnp.clip(goal_minus_last_target / max_delta, -1.0, 1.0)
            local_noise = self.agent.noise_scale * jax.random.normal(k_local, (batch_size, n_local, act_dim))
            local_clean = jnp.clip(raw_action[:, None, :] + local_noise, -1.0, 1.0)
            uniform_clean = jax.random.uniform(k_rand, (batch_size, n_uniform, act_dim), minval=-1.0, maxval=1.0)
            fixed_clean = [raw_action[:, None, :], a_zero[:, None, :]]
            if use_goal_candidate:
                fixed_clean.append(a_goal[:, None, :])
            clean = jnp.concatenate(fixed_clean + [local_clean, uniform_clean], axis=1)
            clean = clean[:, :K, :]

            t = jax.random.uniform(k_t, (batch_size, self.K, 1), minval=1e-3, maxval=0.994)
            noise = jax.random.normal(k_flow_noise, clean.shape)
            noisy_rep = t * clean + (1.0 - t) * noise
            obs_rep = jnp.repeat(obs[:, None, :], self.K, axis=1)

            compat_mode = (act_dim == 3) and (obs.shape[-1] >= 9)
            if compat_mode:
                goal_minus_last_target = obs[..., -6:-3]
                max_delta = jnp.maximum(jnp.float32(self.high_level_max_delta), 1e-6)
                a_ref = jnp.clip(goal_minus_last_target / max_delta, -1.0, 1.0)
                a_ref_rep = a_ref[:, None, :]
                ref_diff = clean - a_ref_rep
                ref_dist = jnp.linalg.norm(ref_diff, axis=-1)
                trigger = ref_dist > jnp.float32(self.trigger_threshold)
                exec_clean = jnp.where(trigger[..., None], a_ref_rep, clean)
                projection_residual_candidates = jnp.linalg.norm(exec_clean - clean, axis=-1)
                candidate_dS = projection_residual_candidates ** 2
            else:
                exec_clean, _, projection_residual_candidates = project_action_jax_humanoid(clean, prior_action=None, residual_radius=self.residual_radius, action_limit=self.action_limit)
                candidate_dS = projection_residual_candidates ** 2
                ref_dist = projection_residual_candidates
                trigger = projection_residual_candidates > 1e-8

            q_reward = jnp.minimum(self.agent.q(p.q1, obs_rep, exec_clean), self.agent.q(p.q2, obs_rep, exec_clean))
            q_proj = self.agent.get_qp(p.qp, obs_rep, clean) if self.use_projection_critic else jnp.zeros_like(q_reward)

            warmup_steps = jnp.maximum(jnp.float32(self.lambda_p_warmup_steps), 1.0)
            lambda_eff = self.lambda_p * jnp.clip(jnp.float32(state.step) / warmup_steps, 0.0, 1.0)

            F = jnp.zeros_like(q_reward)
            Ar = jnp.zeros_like(q_reward)
            AF = jnp.zeros_like(q_reward)
            if self.policy_score_mode == "linear_penalty":
                score = q_reward - lambda_eff * q_proj
            elif self.policy_score_mode == "hard_region":
                q_r_adv = q_reward - jnp.mean(q_reward, axis=1, keepdims=True)
                q_s_adv = q_proj - jnp.mean(q_proj, axis=1, keepdims=True)
                score = jnp.where(trigger, -lambda_eff * q_s_adv, q_r_adv)
            elif self.policy_score_mode == "compat_gate":
                C = q_proj + jnp.float32(self.compat_mu) * jax.lax.stop_gradient(candidate_dS)
                F = jnp.exp(-C / jnp.maximum(jnp.float32(self.compat_tau), 1e-6))
                F = jnp.clip(F, 1e-6, 1.0)
                Ar = q_reward - jnp.mean(q_reward, axis=1, keepdims=True)
                AF = F - jnp.mean(F, axis=1, keepdims=True)
                score = F * Ar + jnp.float32(self.compat_lambda_f) * AF
            else:
                score = q_reward - lambda_eff * q_proj

            score = jax.lax.stop_gradient(score)

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
            apr_batch = jnp.mean(candidate_dS)

            tn_energy = tn_normal_energy = tn_tangent_energy = tn_gate_mean = tn_residual_xt_mean = jnp.float32(0.0)

            def ploss(pp):
                denoiser = lambda tt, xx: self.agent.policy(pp, obs_r, xx, tt)
                flow_loss = self.agent.flow.reverse_weighted_p_loss2(denoiser, t_r, noisy_r, jax.lax.stop_gradient(w_r), jax.lax.stop_gradient(u_r))
                if self.entropy_reg_mode == "legacy" and self.use_tn_energy:
                    v_pred = self.agent.policy(pp, obs_r, noisy_r, t_r)
                    noisy_r_for_filter = jnp.clip(noisy_r, -1.0, 1.0)
                    exec_xt, _, _ = project_action_jax_humanoid(noisy_r_for_filter, prior_action=None, residual_radius=self.residual_radius, action_limit=self.action_limit)
                    exec_xt = jax.lax.stop_gradient(exec_xt)
                    delta = jax.lax.stop_gradient(exec_xt - noisy_r_for_filter)
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
                    return flow_loss + alpha * self.tn_coef * tne, (flow_loss, tne, jnp.mean(e_n), jnp.mean(e_t), jnp.mean(gate), jnp.mean(r), tne)
                v_pred = self.agent.policy(pp, obs_r, noisy_r, t_r)
                e_n, e_t, e_iso, gate, residual = compute_filter_aware_energy_at_points(obs_r, noisy_r, v_pred)
                actor_normal = jnp.mean(e_n)
                actor_safe = jnp.mean(jnp.where(self.safe_energy_variant == "normal_tangent", e_n + e_t, e_n + self.safe_iso_coef * e_iso))
                if self.entropy_reg_mode == "likelihood_tn" and self.use_tn_energy:
                    return flow_loss + self.normal_energy_coef * actor_normal, (flow_loss, actor_normal, jnp.mean(e_n), jnp.mean(e_t), jnp.mean(gate), jnp.mean(residual), actor_safe)
                if self.entropy_reg_mode == "flac_tn" and self.use_tn_energy:
                    return flow_loss + alpha * actor_safe, (flow_loss, actor_safe, jnp.mean(e_n), jnp.mean(e_t), jnp.mean(gate), jnp.mean(residual), actor_safe)
                return flow_loss, (flow_loss, jnp.float32(0.0), jnp.mean(e_n), jnp.mean(e_t), jnp.mean(gate), jnp.mean(residual), actor_safe)

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
            if self.fixed_alpha:
                nloga, ologa = p.log_alpha, o.log_alpha
            else:
                nloga, ologa = apply(self.alpha_optim, p.log_alpha, alpha_grads, o.log_alpha)

            t_q1 = optax.incremental_update(nq1, p.target_q1, self.tau)
            t_q2 = optax.incremental_update(nq2, p.target_q2, self.tau)
            t_vp = optax.incremental_update(nvp, p.target_vp, self.tau)
            t_pol = optax.incremental_update(npol, p.target_policy, self.tau)

            ns = SafePullbackRF2TrainState(
                params=SafePullbackRF2Params(nq1, nq2, t_q1, t_q2, nqp, nvp, t_vp, npol, t_pol, p.g, nloga),
                opt_state=SafePullbackRF2OptStates(oq1, oq2, oqp, ovp, opol, ologa),
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
                        candidate_trigger_rate=jnp.mean(trigger.astype(jnp.float32)),
                        candidate_dS_mean=jnp.mean(candidate_dS),
                        candidate_ref_dist_mean=jnp.mean(ref_dist),
                        compat_F_mean=jnp.mean(F),
                        compat_F_min=jnp.min(F),
                        compat_Ar_mean=jnp.mean(Ar),
                        compat_AF_mean=jnp.mean(AF),
                        policy_score_mode_id=jnp.float32(
                            0.0 if self.policy_score_mode == "linear_penalty"
                            else 1.0 if self.policy_score_mode == "hard_region"
                            else 2.0
                        ),
                        projection_residual_candidate_mean=jnp.mean(projection_residual_candidates),
                        projection_residual_candidate_max=jnp.max(projection_residual_candidates),
                        weight_entropy=weight_entropy,
                        top_weight_mean=top_weight_mean,
                        clean_candidate_std=jnp.mean(jnp.std(clean, axis=1)),
                        raw_action_batch_std=jnp.mean(jnp.std(raw_action, axis=0)),
                        uniform_candidate_fraction=jnp.float32(n_uniform / self.K),
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
