import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

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
        noise_scale=getattr(args, "policy_noise_scale", 0.3),
    )
    return SafePullbackRF2SACENT(
        net, params, gamma=args.gamma, gamma_p=args.gamma_p, lr=args.lr, alpha_lr=args.alpha_lr,
        sample_k=args.sample_k, lambda_p=args.lambda_p, use_projection_critic=args.use_projection_critic,
        fixed_alpha=args.fixed_alpha, alpha_value=args.alpha_value,
        lambda_p_warmup_steps=args.lambda_p_warmup_steps, lambda_d=args.lambda_d,
        use_frpi_score=args.use_frpi_score, tau_c=args.tau_c, mu_c=args.mu_c, lambda_f=args.lambda_f,
        use_tn_energy=args.use_tn_energy, tn_coef=args.tn_coef, sigma_n=args.sigma_n, sigma_t=args.sigma_t,
        tn_r_min=args.tn_r_min, tn_r_max=args.tn_r_max, tn_clip=args.tn_clip, kappa_tn=args.kappa_tn,
        entropy_reg_mode=getattr(args, "entropy_reg_mode", "legacy"),
        candidate_temp=getattr(args, "candidate_temp", 0.10),
        beta_normal_entropy=getattr(args, "beta_normal_entropy", 1.0),
        min_effective_entropy=getattr(args, "min_effective_entropy", -20.0),
        target_effective_entropy=getattr(args, "target_effective_entropy", 1.0),
        normal_energy_coef=getattr(args, "normal_energy_coef", 0.05),
        target_safe_energy=getattr(args, "target_safe_energy", 0.05),
        safe_iso_coef=getattr(args, "safe_iso_coef", 0.05),
        safe_energy_variant=getattr(args, "safe_energy_variant", "normal_iso"),
        weight_mix=getattr(args, "weight_mix", 0.05),
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
            'policy_noise_scale': 0.3, 'start_y_range': 0.4, 'weight_mix': 0.05,
            'entropy_reg_mode': 'legacy', 'candidate_temp': 0.10, 'beta_normal_entropy': 1.0,
            'min_effective_entropy': -20.0, 'target_effective_entropy': 1.0,
            'normal_energy_coef': 0.05, 'target_safe_energy': 0.05, 'safe_iso_coef': 0.05,
            'safe_energy_variant': 'normal_iso',
        }
    defaults = {'use_frpi_score': False, 'tau_c': 1.0, 'mu_c': 1.0, 'lambda_f': 2.0, 'use_tn_energy': False, 'tn_coef': 1.0, 'sigma_n': 0.2, 'sigma_t': 1.0, 'tn_r_min': 0.02, 'tn_r_max': 0.20, 'tn_clip': 10.0, 'kappa_tn': 1.0, 'policy_noise_scale': 0.3, 'start_y_range': 0.4, 'weight_mix': 0.05, 'entropy_reg_mode': 'legacy', 'candidate_temp': 0.10, 'beta_normal_entropy': 1.0, 'min_effective_entropy': -20.0, 'target_effective_entropy': 1.0, 'normal_energy_coef': 0.05, 'target_safe_energy': 0.05, 'safe_iso_coef': 0.05, 'safe_energy_variant': 'normal_iso'}
    for k, v in defaults.items():
        saved_args.setdefault(k, v)
    args = argparse.Namespace(**saved_args)
    agent = make_algo(args)
    agent.state = ckpt['agent_state']
    return agent


def _compute_summary(positions, distance_to_obstacle, state_violation, safe_violation, filter_active,
                     projection_residual, raw_actions, exec_actions, distance_to_goal, is_success, time_to_goal, valid_lengths, episode_return,
                     effective_entropy_tau_res=0.1):
    n, t_max = distance_to_obstacle.shape
    routes = [classify_route(positions[i, :time_to_goal[i] + 1]) for i in range(n) if is_success[i]]
    upper = routes.count('upper')
    lower = routes.count('lower')
    ns = max(len(routes), 1)
    p_up, p_low = upper / ns, lower / ns
    route_entropy = -(p_up * np.log(p_up + 1e-8) + p_low * np.log(p_low + 1e-8))

    step_mask = (np.arange(t_max)[None, :] < (valid_lengths - 1)[:, None])
    valid_step_count = max(int(step_mask.sum()), 1)
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
    all_routes = [classify_route(positions[i, :max(valid_lengths[i],2)]) for i in range(n)]
    upper_all = all_routes.count('upper')
    lower_all = all_routes.count('lower')
    p_up_all = upper_all / max(n,1)
    p_low_all = lower_all / max(n,1)
    route_entropy_all = -(p_up_all * np.log(p_up_all + 1e-8) + p_low_all * np.log(p_low_all + 1e-8))

    final_distance_to_goal = np.zeros((n,), dtype=np.float32)
    min_distance_to_goal = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        step_count = max(int(valid_lengths[i]) - 1, 1)
        valid_dist = distance_to_goal[i, :step_count]
        final_distance_to_goal[i] = valid_dist[-1]
        min_distance_to_goal[i] = np.min(valid_dist)

    return {
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
        'return': float(np.mean(episode_return)),
        'FAR': float(np.sum(filter_active * step_mask) / valid_step_count),
        'APR': float(np.sum(projection_residual * step_mask) / valid_step_count),
        'final_distance_to_goal_mean': float(np.mean(final_distance_to_goal)),
        'final_distance_to_goal_std': float(np.std(final_distance_to_goal)),
        'min_distance_to_goal_mean': float(np.mean(min_distance_to_goal)),
        'min_distance_to_goal_std': float(np.std(min_distance_to_goal)),
        'raw_action_std': float(np.mean([np.std(raw_actions[i, :max(valid_lengths[i]-1,1)], axis=0).mean() for i in range(n)])),
        'exec_action_std': float(np.mean([np.std(exec_actions[i, :max(valid_lengths[i]-1,1)], axis=0).mean() for i in range(n)])),
        'projection_residual_mean_nonzero': float(np.mean(projection_residual[projection_residual > 1e-8])) if np.any(projection_residual > 1e-8) else 0.0,
        'filter_activation_episode_rate': float(np.mean([np.any(filter_active[i, :max(valid_lengths[i]-1,1)]) for i in range(n)])),
        'route_entropy_all_episodes': float(route_entropy_all),
        'route_upper_ratio_all_episodes': float(p_up_all),
        'route_lower_ratio_all_episodes': float(p_low_all),
    }


def collect_eval_rollouts(agent, algo, eval_episodes=200, seed=0, effective_entropy_tau_res=0.1, start_y_range=0.4):
    env = SafeObstacleNavigation2DEnv(use_filter=algo != 'rf2_no_filter', seed=seed, start_y_range=start_y_range)
    key = jax.random.PRNGKey(seed + 123)
    t_max, n = env.episode_len, eval_episodes

    positions = np.zeros((n, t_max + 1, 2), np.float32)
    obs_all = np.zeros((n, t_max + 1, 8), np.float32)
    raw_actions = np.zeros((n, t_max, 2), np.float32)
    exec_actions = np.zeros((n, t_max, 2), np.float32)
    rewards = np.zeros((n, t_max), np.float32)
    state_violation = np.zeros((n, t_max), bool)
    tightened_violation = np.zeros((n, t_max), bool)
    safe_violation = np.zeros((n, t_max), bool)
    filter_active = np.zeros((n, t_max), bool)
    projection_residual = np.zeros((n, t_max), np.float32)
    projection_cost = np.zeros((n, t_max), np.float32)
    distance_to_goal = np.zeros((n, t_max), np.float32)
    distance_to_obstacle = np.zeros((n, t_max), np.float32)
    is_success = np.zeros((n,), bool)
    time_to_goal = np.full((n,), t_max, np.int32)
    valid_lengths = np.full((n,), t_max + 1, np.int32)
    episode_return = np.zeros((n,), np.float32)

    for i in range(n):
        obs, _ = env.reset(seed=seed + i)
        positions[i, 0] = env.state
        obs_all[i, 0] = obs
        for t in range(t_max):
            if algo == 'goal_filter':
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

    rollout_data = {
        'positions': positions,
        'obs': obs_all,
        'raw_actions': raw_actions,
        'exec_actions': exec_actions,
        'rewards': rewards,
        'state_violation': state_violation,
        'tightened_violation': tightened_violation,
        'safe_violation': safe_violation,
        'filter_active': filter_active,
        'projection_residual': projection_residual,
        'projection_cost': projection_cost,
        'distance_to_goal': distance_to_goal,
        'distance_to_obstacle': distance_to_obstacle,
        'is_success': is_success,
        'time_to_goal': time_to_goal,
        'valid_lengths': valid_lengths,
        'episode_return': episode_return,
    }
    summary = _compute_summary(
        positions=positions,
        distance_to_obstacle=distance_to_obstacle,
        state_violation=state_violation,
        safe_violation=safe_violation,
        filter_active=filter_active,
        projection_residual=projection_residual,
        raw_actions=raw_actions,
        exec_actions=exec_actions,
        distance_to_goal=distance_to_goal,
        is_success=is_success,
        time_to_goal=time_to_goal,
        valid_lengths=valid_lengths,
        episode_return=episode_return,
        effective_entropy_tau_res=effective_entropy_tau_res,
    )
    return rollout_data, summary


def plot_eval_trajectories(save_dir, positions, is_success, time_to_goal, valid_lengths, projection_residual=None,
                           filter_active=None, route_labels=None, max_episodes=100, success_only=False,
                           failure_only=False, individual=False, dpi=200, fmt="png", title=None):
    del projection_residual, filter_active
    env = SafeObstacleNavigation2DEnv(seed=0)
    cfg = env.cfg
    x_min, x_max = cfg.x_min, cfg.x_max
    y_min, y_max = cfg.y_min, cfg.y_max
    obs_center = np.asarray(cfg.obstacle_center)
    obs_radius = cfg.obstacle_radius
    eps_obs = cfg.eps_obs
    goal = np.asarray(env.goal)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    n = positions.shape[0]

    if route_labels is None:
        route_labels = []
        for i in range(n):
            if is_success[i]:
                end_t = min(int(time_to_goal[i]) + 1, int(valid_lengths[i]))
                route_labels.append(classify_route(positions[i, :end_t]))
            else:
                route_labels.append('failure')

    indices = list(range(n))
    if success_only:
        indices = [i for i in indices if is_success[i]]
    if failure_only:
        indices = [i for i in indices if not is_success[i]]

    if not indices:
        mode = 'success' if success_only else 'failure' if failure_only else 'all'
        print(f"No episodes available for {mode} trajectory plot.")
    else:
        if max_episodes > 0:
            indices = indices[:max_episodes]

        def draw_base(ax):
            ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, lw=1.2, ec='black', label='workspace'))
            ax.add_patch(Circle(obs_center, obs_radius, color='gray', alpha=0.3, label='obstacle'))
            ax.add_patch(Circle(obs_center, obs_radius + eps_obs, fill=False, ec='gray', ls='--', lw=1.2, label='tightened obstacle'))
            ax.scatter(goal[0], goal[1], c='tab:green', marker='*', s=90, label='goal')
            ax.set_xlim(x_min - 0.2, x_max + 0.2)
            ax.set_ylim(y_min - 0.2, y_max + 0.2)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.2)

        fig, ax = plt.subplots(figsize=(7, 5))
        draw_base(ax)

        shown = set()
        for i in indices:
            vl = int(max(valid_lengths[i], 2))
            traj = positions[i, :vl]
            route = route_labels[i]
            if is_success[i]:
                color = 'tab:blue' if route == 'upper' else 'tab:orange'
                name = f"success-{route}"
                marker = 'x'
                alpha = 0.65
            else:
                color = 'lightgray'
                name = 'failure'
                marker = '^'
                alpha = 0.45
            label = name if name not in shown else None
            shown.add(name)
            ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=alpha, lw=1.2, label=label)
            ax.scatter(traj[0, 0], traj[0, 1], c=color, s=12, alpha=0.85)
            ax.scatter(traj[-1, 0], traj[-1, 1], c=color, s=22, marker=marker, alpha=0.85)

        if title:
            ax.set_title(title)
        ax.legend(loc='best', fontsize=8)
        suffix = 'all'
        if success_only:
            suffix = 'success'
        elif failure_only:
            suffix = 'failure'
        fig.tight_layout()
        fig.savefig(save_dir / f"trajectories_{suffix}.{fmt}", dpi=dpi)
        plt.close(fig)

    if individual:
        indiv_dir = save_dir / 'trajectories_individual'
        indiv_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            if success_only and not is_success[i]:
                continue
            if failure_only and is_success[i]:
                continue
            vl = int(max(valid_lengths[i], 2))
            traj = positions[i, :vl]
            route = route_labels[i]
            status = 'success' if is_success[i] else 'failure'
            color = 'tab:blue' if route == 'upper' else 'tab:orange' if route == 'lower' else 'lightgray'
            marker = 'x' if is_success[i] else '^'

            fig, ax = plt.subplots(figsize=(7, 5))
            draw_base(ax)
            ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.65, lw=1.2, label=f"{status}-{route}")
            ax.scatter(traj[0, 0], traj[0, 1], c=color, s=15)
            ax.scatter(traj[-1, 0], traj[-1, 1], c=color, s=24, marker=marker)
            ax.legend(loc='best', fontsize=8)
            if title:
                ax.set_title(f"{title} | episode {i}")
            fig.tight_layout()
            fig.savefig(indiv_dir / f"episode_{i:04d}_{status}_{route}.{fmt}", dpi=dpi)
            plt.close(fig)


def run_evaluation(agent, algo, eval_episodes=200, seed=0, effective_entropy_tau_res=0.1, start_y_range=0.4):
    _, summary = collect_eval_rollouts(agent, algo, eval_episodes=eval_episodes, seed=seed,
                                       effective_entropy_tau_res=effective_entropy_tau_res, start_y_range=start_y_range)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--algo', required=True)
    p.add_argument('--eval_episodes', type=int, default=200)
    p.add_argument('--save_dir', required=True)
    p.add_argument('--effective_entropy_tau_res', type=float, default=0.1)
    p.add_argument('--start_y_range', type=float, default=0.4)
    p.add_argument('--save_plots', action='store_true', default=False)
    p.add_argument('--plot_max_episodes', type=int, default=100)
    p.add_argument('--plot_individual_episodes', action='store_true', default=False)
    p.add_argument('--plot_success_only', action='store_true', default=False)
    p.add_argument('--plot_failure_only', action='store_true', default=False)
    p.add_argument('--plot_dpi', type=int, default=200)
    p.add_argument('--plot_format', choices=['png', 'pdf', 'svg'], default='png')
    args = p.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    agent = None if args.algo == 'goal_filter' else load_agent(args.checkpoint, args.algo)
    rollout_data, summary = collect_eval_rollouts(
        agent, args.algo, eval_episodes=args.eval_episodes, seed=0,
        effective_entropy_tau_res=args.effective_entropy_tau_res,
        start_y_range=args.start_y_range,
    )

    np.savez(save_dir / 'rollouts.npz', **rollout_data)
    (save_dir / 'summary.json').write_text(json.dumps(summary, indent=2))

    with open(save_dir / 'metrics.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    if args.save_plots:
        plot_eval_trajectories(
            save_dir=save_dir,
            positions=rollout_data['positions'],
            is_success=rollout_data['is_success'],
            time_to_goal=rollout_data['time_to_goal'],
            valid_lengths=rollout_data['valid_lengths'],
            projection_residual=rollout_data['projection_residual'],
            filter_active=rollout_data['filter_active'],
            max_episodes=args.plot_max_episodes,
            success_only=args.plot_success_only,
            failure_only=args.plot_failure_only,
            individual=args.plot_individual_episodes,
            dpi=args.plot_dpi,
            fmt=args.plot_format,
            title=f"{args.algo} eval trajectories",
        )


if __name__ == '__main__':
    main()
