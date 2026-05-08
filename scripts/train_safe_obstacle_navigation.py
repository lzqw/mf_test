import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
try:
    from tensorboardX import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        class SummaryWriter:
            def __init__(self, *args, **kwargs):
                pass

            def add_scalar(self, *args, **kwargs):
                pass

            def flush(self):
                pass

            def close(self):
                pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.safe_obstacle_navigation_2d import SafeObstacleNavigation2DEnv
from relax.algorithm.safe_pullback_rf2_sac_ent import SafePullbackRF2SACENT
from relax.network.safe_pullback_rf2_sac_ent import create_safe_pullback_rf2_sac_ent_net
from scripts.safe_pullback_experience import SafePullbackExperience
from eval.eval_safe_obstacle_navigation import collect_eval_rollouts, plot_eval_trajectories, run_evaluation


class Batch(NamedTuple):
    obs: jnp.ndarray
    raw_action: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    next_obs: jnp.ndarray
    projection_residual: jnp.ndarray
    projection_cost: jnp.ndarray


def goal_controller(state, goal):
    d = goal - state
    return np.clip(d / (np.linalg.norm(d) + 1e-6), -1.0, 1.0).astype(np.float32)


def make_algo(args, obs_dim=8, act_dim=2):
    key = jax.random.PRNGKey(args.seed)
    net, params = create_safe_pullback_rf2_sac_ent_net(
        key, obs_dim, act_dim, hidden_sizes=[256, 256, 256], diffusion_hidden_sizes=[256, 256, 256],
        num_timesteps=args.diffusion_steps, num_ent_timesteps=args.num_ent_timesteps,
        alpha_value=args.alpha_value, fixed_alpha=args.fixed_alpha, init_alpha=args.init_alpha,
        noise_scale=args.policy_noise_scale,
    )
    return SafePullbackRF2SACENT(
        net, params, gamma=args.gamma, gamma_p=args.gamma_p, lr=args.lr, alpha_lr=args.alpha_lr,
        sample_k=args.sample_k, lambda_p=args.lambda_p, use_projection_critic=args.use_projection_critic,
        fixed_alpha=args.fixed_alpha, alpha_value=args.alpha_value,
        lambda_p_warmup_steps=args.lambda_p_warmup_steps, lambda_d=args.lambda_d,
        use_frpi_score=args.use_frpi_score, tau_c=args.tau_c, mu_c=args.mu_c, lambda_f=args.lambda_f,
        use_tn_energy=args.use_tn_energy, tn_coef=args.tn_coef, sigma_n=args.sigma_n, sigma_t=args.sigma_t,
        tn_r_min=args.tn_r_min, tn_r_max=args.tn_r_max, tn_clip=args.tn_clip, kappa_tn=args.kappa_tn,
        entropy_reg_mode=args.entropy_reg_mode, candidate_temp=args.candidate_temp,
        beta_normal_entropy=args.beta_normal_entropy, min_effective_entropy=args.min_effective_entropy,
        target_effective_entropy=args.target_effective_entropy, normal_energy_coef=args.normal_energy_coef,
        target_safe_energy=args.target_safe_energy, safe_iso_coef=args.safe_iso_coef,
        safe_energy_variant=args.safe_energy_variant, weight_mix=args.weight_mix,
    )


def configure_algo_mode(args):
    if args.algo == 'rf2_filter':
        args.use_filter = True
        args.use_projection_critic = False
        args.lambda_p = 0.0
    elif args.algo == 'safe_pullback_rf2':
        args.use_filter = True
        args.use_projection_critic = True
    elif args.algo == 'safe_pullback_rf2_no_entropy':
        args.use_filter = True
        args.use_projection_critic = True
        args.fixed_alpha = True
        args.alpha_value = 0.01
    elif args.algo == 'rf2_no_filter':
        args.use_filter = False
        args.use_projection_critic = False
        args.lambda_p = 0.0
    elif args.algo == 'goal_filter':
        args.use_filter = True
        args.use_projection_critic = False
        args.lambda_p = 0.0
    else:
        raise ValueError(f'Unsupported algo mode: {args.algo}')


def sample_batch(buf, batch_size):
    idx = np.random.randint(0, len(buf), size=batch_size)
    items = [buf[i] for i in idx]
    return Batch(
        obs=jnp.asarray(np.stack([x.obs for x in items])),
        raw_action=jnp.asarray(np.stack([x.raw_action for x in items])),
        action=jnp.asarray(np.stack([x.action for x in items])),
        reward=jnp.asarray(np.stack([x.reward for x in items])),
        done=jnp.asarray(np.stack([x.done for x in items]).astype(np.float32)),
        next_obs=jnp.asarray(np.stack([x.next_obs for x in items])),
        projection_residual=jnp.asarray(np.stack([x.projection_residual for x in items])),
        projection_cost=jnp.asarray(np.stack([x.projection_cost for x in items])),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--algo', default='safe_pullback_rf2')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--total_steps', type=int, default=200000)
    p.add_argument('--start_steps', type=int, default=20000)
    p.add_argument('--update_after', type=int, default=10000)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--eval_interval', type=int, default=5000)
    p.add_argument('--noise_sigma_x', type=float, default=0.01)
    p.add_argument('--noise_sigma_y', type=float, default=0.01)
    p.add_argument('--start_y_range', type=float, default=0.4)
    p.add_argument('--policy_noise_scale', type=float, default=0.3)
    p.add_argument('--weight_mix', type=float, default=0.05)
    p.add_argument('--log_dir', required=True)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--alpha_lr', type=float, default=1e-2)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--gamma_p', type=float, default=0.99)
    p.add_argument('--sample_k', type=int, default=256)
    p.add_argument('--lambda_p', type=float, default=0.1)
    p.add_argument('--use_projection_critic', action='store_true', default=False)
    p.add_argument('--fixed_alpha', action='store_true', default=False)
    p.add_argument('--alpha_value', type=float, default=0.1)
    p.add_argument('--init_alpha', type=float, default=0.1)
    p.add_argument('--diffusion_steps', type=int, default=10)
    p.add_argument('--num_ent_timesteps', type=int, default=10)
    p.add_argument('--lambda_p_warmup_steps', type=int, default=100000)
    p.add_argument('--lambda_d', type=float, default=0.5)
    p.add_argument('--eval_episodes', type=int, default=100)
    p.add_argument('--use_frpi_score', action='store_true', default=False)
    p.add_argument('--tau_c', type=float, default=1.0)
    p.add_argument('--mu_c', type=float, default=1.0)
    p.add_argument('--lambda_f', type=float, default=2.0)
    p.add_argument('--use_tn_energy', action='store_true', default=False)
    p.add_argument('--tn_coef', type=float, default=1.0)
    p.add_argument('--sigma_n', type=float, default=0.2)
    p.add_argument('--sigma_t', type=float, default=1.0)
    p.add_argument('--tn_r_min', type=float, default=0.02)
    p.add_argument('--tn_r_max', type=float, default=0.20)
    p.add_argument('--tn_clip', type=float, default=10.0)
    p.add_argument('--kappa_tn', type=float, default=1.0)
    p.add_argument('--entropy_reg_mode', choices=['legacy', 'likelihood_tn', 'flac_tn'], default='legacy')
    p.add_argument('--candidate_temp', type=float, default=0.10)
    p.add_argument('--beta_normal_entropy', type=float, default=1.0)
    p.add_argument('--min_effective_entropy', type=float, default=-20.0)
    p.add_argument('--target_effective_entropy', type=float, default=1.0)
    p.add_argument('--normal_energy_coef', type=float, default=0.05)
    p.add_argument('--target_safe_energy', type=float, default=0.05)
    p.add_argument('--safe_iso_coef', type=float, default=0.05)
    p.add_argument('--safe_energy_variant', choices=['normal_iso', 'normal_tangent'], default='normal_iso')
    p.add_argument('--save_eval_plots', action='store_true', default=False)
    p.add_argument('--save_eval_rollouts', action='store_true', default=False)
    p.add_argument('--eval_plot_max_episodes', type=int, default=50)
    p.add_argument('--eval_plot_individual_episodes', action='store_true', default=False)
    p.add_argument('--eval_plot_format', choices=['png', 'pdf', 'svg'], default='png')
    p.add_argument('--eval_plot_dpi', type=int, default=200)
    args = p.parse_args()
    configure_algo_mode(args)

    np.random.seed(args.seed)
    env = SafeObstacleNavigation2DEnv(noise_sigma=(args.noise_sigma_x, args.noise_sigma_y), use_filter=args.use_filter, seed=args.seed, start_y_range=args.start_y_range)
    agent = make_algo(args)
    key = jax.random.PRNGKey(args.seed + 7)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir / "tb"))
    with open(log_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    buffer = []
    train_log = []
    eval_log = []

    obs, _ = env.reset(seed=args.seed)
    try:
        for step in range(1, args.total_steps + 1):
            if args.algo == 'goal_filter':
                raw_action = goal_controller(env.state, env.goal)
            elif step < args.start_steps:
                raw_action = env.action_space.sample()
            else:
                key, ak = jax.random.split(key)
                raw_action = np.asarray(agent.get_action(ak, obs[None, :])[0])

            next_obs, reward, terminated, truncated, info = env.step(raw_action)
            exp = SafePullbackExperience.create(obs, raw_action, info['exec_action'], reward, terminated, truncated, next_obs, info)
            buffer.append(exp)
            if len(buffer) > 1_000_000:
                buffer.pop(0)
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()

            writer.add_scalar("train/buffer_size", float(len(buffer)), step)
            env_scalar_keys = (
                ("projection_residual", "train_env/projection_residual"),
                ("projection_cost", "train_env/projection_cost"),
                ("filter_active", "train_env/filter_active"),
                ("safe_violation", "train_env/safe_violation"),
                ("state_violation", "train_env/state_violation"),
                ("reward", "train_env/reward"),
            )
            for info_key, tag in env_scalar_keys:
                if info_key in info:
                    writer.add_scalar(tag, float(info[info_key]), step)
            writer.add_scalar("train_env/reward", float(reward), step)

            if step >= args.update_after and len(buffer) >= args.batch_size and args.algo != 'goal_filter':
                key, uk = jax.random.split(key)
                batch = sample_batch(buffer, args.batch_size)
                out = agent.update(uk, batch)
                out['step'] = step
                train_log.append(out)
                for k, v in out.items():
                    if k == "step":
                        continue
                    writer.add_scalar(f"train/{k}", float(v), step)
                writer.flush()

            if step % args.eval_interval == 0:
                if args.save_eval_plots or args.save_eval_rollouts:
                    eval_agent = None if args.algo == 'goal_filter' else agent
                    rollout_data, eval_result = collect_eval_rollouts(
                        eval_agent, args.algo, args.eval_episodes, seed=args.seed + step, start_y_range=args.start_y_range
                    )
                    step_eval_dir = log_dir / "eval_rollouts" / f"step_{step:08d}"
                    step_eval_dir.mkdir(parents=True, exist_ok=True)
                    (step_eval_dir / "summary.json").write_text(json.dumps(eval_result, indent=2))
                    if args.save_eval_rollouts:
                        np.savez(step_eval_dir / "rollouts.npz", **rollout_data)
                    if args.save_eval_plots:
                        plot_eval_trajectories(
                            save_dir=step_eval_dir,
                            positions=rollout_data["positions"],
                            is_success=rollout_data["is_success"],
                            time_to_goal=rollout_data["time_to_goal"],
                            valid_lengths=rollout_data["valid_lengths"],
                            projection_residual=rollout_data.get("projection_residual"),
                            filter_active=rollout_data.get("filter_active"),
                            max_episodes=args.eval_plot_max_episodes,
                            individual=args.eval_plot_individual_episodes,
                            dpi=args.eval_plot_dpi,
                            fmt=args.eval_plot_format,
                            title=f"{args.algo} trajectories at step {step}",
                        )
                else:
                    if args.algo == 'goal_filter':
                        eval_result = run_evaluation(None, args.algo, args.eval_episodes, seed=args.seed + step, start_y_range=args.start_y_range)
                    else:
                        eval_result = run_evaluation(agent, args.algo, args.eval_episodes, seed=args.seed + step, start_y_range=args.start_y_range)
                eval_result['step'] = step
                eval_log.append(eval_result)
                for k, v in eval_result.items():
                    if k == "step":
                        continue
                    writer.add_scalar(f"eval/{k}", float(v), step)
                writer.flush()
                print(
                    f"[step {step}] eval_return={eval_result.get('return', float('nan')):.4f}, "
                    f"success={eval_result.get('success_rate', float('nan')):.4f}, "
                    f"FAR={eval_result.get('FAR', float('nan')):.4f}, "
                    f"APR={eval_result.get('APR', float('nan')):.4f}"
                )
    except KeyboardInterrupt:
        print("Training interrupted. Saving logs and checkpoint...")
    finally:
        with open(log_dir / 'train_metrics.pkl', 'wb') as f:
            pickle.dump(train_log, f)
        with open(log_dir / 'checkpoint.pkl', 'wb') as f:
            pickle.dump({'algo': args.algo, 'seed': args.seed, 'args': vars(args), 'agent_state': agent.state}, f)
        with open(log_dir / 'eval_metrics.pkl', 'wb') as f:
            pickle.dump(eval_log, f)
        writer.flush()
        writer.close()


if __name__ == '__main__':
    main()
