import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import jax
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.safe_obstacle_navigation_2d import SafeObstacleNavigation2DEnv
from relax.algorithm.safe_pullback_rf2_sac_ent import SafePullbackRF2SACENT
from relax.network.safe_pullback_rf2_sac_ent import create_safe_pullback_rf2_sac_ent_net


def goal_controller(state, goal):
    d = goal - state
    return np.clip(d / (np.linalg.norm(d) + 1e-6), -1.0, 1.0).astype(np.float32)


def make_algo(args, obs_dim=8, act_dim=2):
    key = jax.random.PRNGKey(args.seed)
    net, params = create_safe_pullback_rf2_sac_ent_net(
        key, obs_dim, act_dim, hidden_sizes=[256, 256, 256], diffusion_hidden_sizes=[256, 256, 256],
        num_timesteps=args.diffusion_steps, num_ent_timesteps=args.num_ent_timesteps,
        alpha_value=args.alpha_value, fixed_alpha=args.fixed_alpha, init_alpha=args.init_alpha,
    )
    return SafePullbackRF2SACENT(
        net, params, gamma=args.gamma, gamma_p=args.gamma_p, lr=args.lr, alpha_lr=args.alpha_lr,
        sample_k=args.sample_k, lambda_p=args.lambda_p, use_projection_critic=args.use_projection_critic,
        fixed_alpha=args.fixed_alpha, alpha_value=args.alpha_value,
        lambda_p_warmup_steps=args.lambda_p_warmup_steps, lambda_d=args.lambda_d,
        use_frpi_score=args.use_frpi_score, tau_c=args.tau_c, mu_c=args.mu_c, lambda_f=args.lambda_f,
        use_tn_energy=args.use_tn_energy, tn_coef=args.tn_coef, sigma_n=args.sigma_n, sigma_t=args.sigma_t,
        tn_r_min=args.tn_r_min, tn_r_max=args.tn_r_max, tn_clip=args.tn_clip, kappa_tn=args.kappa_tn,
    )


def classify_route(pos_traj):
    near_idx = np.where(np.abs(pos_traj[:, 0]) < 0.5)[0]
    y_mean = np.mean(pos_traj[near_idx, 1]) if len(near_idx) > 0 else np.mean(pos_traj[:, 1])
    return "upper" if y_mean > 0 else "lower"


def load_agent(checkpoint_path, algo):
    with open(checkpoint_path, 'rb') as f:
        ckpt = pickle.load(f)
    saved_args = ckpt.get('args')
    if saved_args is None:
        saved_args = {
            'seed': ckpt['seed'], 'diffusion_steps': 10, 'num_ent_timesteps': 10, 'alpha_value': 0.01,
            'fixed_alpha': False, 'init_alpha': 0.01, 'gamma': 0.99, 'gamma_p': 0.99, 'lr': 3e-4,
            'alpha_lr': 1e-2, 'sample_k': 64, 'lambda_p': 1.0, 'use_projection_critic': True,
            'lambda_p_warmup_steps': 100000, 'lambda_d': 0.5,
            'use_frpi_score': False, 'tau_c': 1.0, 'mu_c': 1.0, 'lambda_f': 2.0,
            'use_tn_energy': False, 'tn_coef': 1.0, 'sigma_n': 0.2, 'sigma_t': 1.0,
            'tn_r_min': 0.02, 'tn_r_max': 0.20, 'tn_clip': 10.0, 'kappa_tn': 1.0,
        }
    defaults = {'use_frpi_score': False, 'tau_c': 1.0, 'mu_c': 1.0, 'lambda_f': 2.0, 'use_tn_energy': False, 'tn_coef': 1.0, 'sigma_n': 0.2, 'sigma_t': 1.0, 'tn_r_min': 0.02, 'tn_r_max': 0.20, 'tn_clip': 10.0, 'kappa_tn': 1.0}
    for k, v in defaults.items():
        saved_args.setdefault(k, v)
    args = argparse.Namespace(**saved_args)
    agent = make_algo(args)
    agent.state = ckpt['agent_state']
    return agent


def run_evaluation(agent, algo, eval_episodes=200, seed=0, effective_entropy_tau_res=0.1):
    env = SafeObstacleNavigation2DEnv(use_filter=algo != 'rf2_no_filter', seed=seed)
    key = jax.random.PRNGKey(seed + 123)
    T, N = env.episode_len, eval_episodes

    is_success = np.zeros((N,), bool)
    episode_return = np.zeros((N,), np.float32)
    filter_active = np.zeros((N, T), bool)
    safe_violation = np.zeros((N, T), bool)
    projection_residual = np.zeros((N, T), np.float32)
    distance_to_obstacle = np.zeros((N, T), np.float32)
    positions = np.zeros((N, T + 1, 2), np.float32)
    time_to_goal = np.full((N,), T, np.int32)
    valid_lengths = np.full((N,), T + 1, np.int32)

    for i in range(N):
        obs, _ = env.reset(seed=seed + i)
        positions[i, 0] = env.state
        for t in range(T):
            if algo == 'goal_filter':
                raw = goal_controller(env.state, env.goal)
            else:
                key, ak = jax.random.split(key)
                raw = np.asarray(agent.get_action(ak, obs[None, :])[0])
            nobs, r, term, trunc, info = env.step(raw)
            filter_active[i, t] = info['filter_active']
            safe_violation[i, t] = info['safe_violation']
            projection_residual[i, t] = info['projection_residual']
            distance_to_obstacle[i, t] = info['distance_to_obstacle']
            episode_return[i] += r
            positions[i, t + 1] = env.state
            obs = nobs
            if term and not is_success[i]:
                is_success[i] = True
                time_to_goal[i] = t + 1
            if term or trunc:
                valid_lengths[i] = t + 2
                positions[i, t + 1:] = positions[i, t + 1]
                break
    routes = [classify_route(positions[i, :time_to_goal[i] + 1]) for i in range(N) if is_success[i]]
    upper = routes.count('upper')
    lower = routes.count('lower')
    ns = max(len(routes), 1)
    p_up, p_low = upper / ns, lower / ns
    route_entropy = -(p_up * np.log(p_up + 1e-8) + p_low * np.log(p_low + 1e-8))
    success_idx = np.where(is_success)[0]
    if success_idx.size > 0:
        succ_routes = [classify_route(positions[i, :time_to_goal[i] + 1]) for i in success_idx]
        succ_res = np.array([np.mean(projection_residual[i, :time_to_goal[i]]) for i in success_idx], dtype=np.float32)
        weights = np.exp(-succ_res / max(effective_entropy_tau_res, 1e-6))
        wsum = max(float(np.sum(weights)), 1e-8)
        p_up_eff = float(np.sum(weights * (np.array(succ_routes) == 'upper').astype(np.float32)) / wsum)
        p_low_eff = float(np.sum(weights * (np.array(succ_routes) == 'lower').astype(np.float32)) / wsum)
    else:
        p_up_eff, p_low_eff = 0.0, 0.0
    effective_route_entropy = -(p_up_eff * np.log(p_up_eff + 1e-8) + p_low_eff * np.log(p_low_eff + 1e-8))

    step_mask = (np.arange(T)[None, :] < (valid_lengths - 1)[:, None])
    valid_step_count = max(int(step_mask.sum()), 1)
    return {
        'success_rate': float(np.mean(is_success)),
        'collision_rate': float(np.mean(np.any((distance_to_obstacle < 0.0) & step_mask, axis=1))),
        'return': float(np.mean(episode_return)),
        'FAR': float(np.sum(filter_active * step_mask) / valid_step_count),
        'APR': float(np.sum(projection_residual * step_mask) / valid_step_count),
        'feasible_raw_action_ratio': float(np.sum((1 - safe_violation.astype(np.float32)) * step_mask) / valid_step_count),
        'route_entropy': float(route_entropy),
        'effective_route_upper_ratio': float(p_up_eff),
        'effective_route_lower_ratio': float(p_low_eff),
        'effective_route_entropy': float(effective_route_entropy),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--algo', required=True)
    p.add_argument('--eval_episodes', type=int, default=200)
    p.add_argument('--save_dir', required=True)
    p.add_argument('--effective_entropy_tau_res', type=float, default=0.1)
    args = p.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    env = SafeObstacleNavigation2DEnv(use_filter=args.algo != 'rf2_no_filter', seed=0)
    agent = None if args.algo == 'goal_filter' else load_agent(args.checkpoint, args.algo)
    key = jax.random.PRNGKey(123)

    T, N = env.episode_len, args.eval_episodes
    positions = np.zeros((N, T + 1, 2), np.float32)
    obs_all = np.zeros((N, T + 1, 8), np.float32)
    raw_actions = np.zeros((N, T, 2), np.float32)
    exec_actions = np.zeros((N, T, 2), np.float32)
    rewards = np.zeros((N, T), np.float32)
    state_violation = np.zeros((N, T), bool)
    tightened_violation = np.zeros((N, T), bool)
    safe_violation = np.zeros((N, T), bool)
    filter_active = np.zeros((N, T), bool)
    projection_residual = np.zeros((N, T), np.float32)
    projection_cost = np.zeros((N, T), np.float32)
    distance_to_goal = np.zeros((N, T), np.float32)
    distance_to_obstacle = np.zeros((N, T), np.float32)
    is_success = np.zeros((N,), bool)
    time_to_goal = np.full((N,), T, np.int32)
    valid_lengths = np.full((N,), T + 1, np.int32)
    episode_return = np.zeros((N,), np.float32)

    for i in range(N):
        obs, _ = env.reset(seed=i)
        positions[i, 0] = env.state
        obs_all[i, 0] = obs
        for t in range(T):
            if args.algo == 'goal_filter':
                raw = goal_controller(env.state, env.goal)
            else:
                key, ak = jax.random.split(key)
                raw = np.asarray(agent.get_action(ak, obs[None, :])[0])
            nobs, r, term, trunc, info = env.step(raw)
            raw_actions[i, t] = info['raw_action']
            exec_actions[i, t] = info['exec_action']
            rewards[i, t] = r
            state_violation[i, t] = info['state_violation']
            tightened_violation[i, t] = info['tightened_violation']
            safe_violation[i, t] = info['safe_violation']
            filter_active[i, t] = info['filter_active']
            projection_residual[i, t] = info['projection_residual']
            projection_cost[i, t] = info['projection_cost']
            distance_to_goal[i, t] = info['distance_to_goal']
            distance_to_obstacle[i, t] = info['distance_to_obstacle']
            episode_return[i] += r
            positions[i, t + 1] = env.state
            obs_all[i, t + 1] = nobs
            obs = nobs
            if term and not is_success[i]:
                is_success[i] = True
                time_to_goal[i] = t + 1
            if term or trunc:
                valid_lengths[i] = t + 2
                positions[i, t + 1:] = positions[i, t + 1]
                obs_all[i, t + 1:] = obs_all[i, t + 1]
                break

    np.savez(save_dir / 'rollouts.npz', positions=positions, obs=obs_all, raw_actions=raw_actions, exec_actions=exec_actions,
             rewards=rewards, state_violation=state_violation, tightened_violation=tightened_violation,
             safe_violation=safe_violation, filter_active=filter_active, projection_residual=projection_residual,
             projection_cost=projection_cost, distance_to_goal=distance_to_goal, distance_to_obstacle=distance_to_obstacle,
             is_success=is_success, time_to_goal=time_to_goal, valid_lengths=valid_lengths, episode_return=episode_return)

    routes = [classify_route(positions[i, :time_to_goal[i] + 1]) for i in range(N) if is_success[i]]
    upper = routes.count('upper')
    lower = routes.count('lower')
    ns = max(len(routes), 1)
    p_up, p_low = upper / ns, lower / ns
    route_entropy = -(p_up * np.log(p_up + 1e-8) + p_low * np.log(p_low + 1e-8))

    step_mask = (np.arange(T)[None, :] < (valid_lengths - 1)[:, None])
    valid_step_count = max(int(step_mask.sum()), 1)
    success_idx = np.where(is_success)[0]
    if success_idx.size > 0:
        succ_routes = [classify_route(positions[i, :time_to_goal[i] + 1]) for i in success_idx]
        succ_res = np.array([np.mean(projection_residual[i, :time_to_goal[i]]) for i in success_idx], dtype=np.float32)
        weights = np.exp(-succ_res / max(args.effective_entropy_tau_res, 1e-6))
        wsum = max(float(np.sum(weights)), 1e-8)
        p_up_eff = float(np.sum(weights * (np.array(succ_routes) == 'upper').astype(np.float32)) / wsum)
        p_low_eff = float(np.sum(weights * (np.array(succ_routes) == 'lower').astype(np.float32)) / wsum)
    else:
        p_up_eff, p_low_eff = 0.0, 0.0
    effective_route_entropy = -(p_up_eff * np.log(p_up_eff + 1e-8) + p_low_eff * np.log(p_low_eff + 1e-8))

    summary = {
        'success_rate': float(np.mean(is_success)),
        'collision_rate': float(np.mean(np.any((distance_to_obstacle < 0.0) & step_mask, axis=1))),
        'state_violation_rate': float(np.sum(state_violation * step_mask) / valid_step_count),
        'episode_return_mean': float(np.mean(episode_return)),
        'episode_return_std': float(np.std(episode_return)),
        'time_to_goal_mean': float(np.mean(time_to_goal)),
        'filter_activation_rate': float(np.sum(filter_active * step_mask) / valid_step_count),
        'avg_projection_residual': float(np.sum(projection_residual * step_mask) / valid_step_count),
        'feasible_raw_action_ratio': float(np.sum((1 - safe_violation.astype(np.float32)) * step_mask) / valid_step_count),
        'route_upper_ratio': float(p_up),
        'route_lower_ratio': float(p_low),
        'route_entropy': float(route_entropy),
        'effective_route_upper_ratio': float(p_up_eff),
        'effective_route_lower_ratio': float(p_low_eff),
        'effective_route_entropy': float(effective_route_entropy),
    }
    (save_dir / 'summary.json').write_text(json.dumps(summary, indent=2))

    with open(save_dir / 'metrics.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)


if __name__ == '__main__':
    main()
