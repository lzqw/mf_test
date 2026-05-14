import argparse
import pickle
import sys
from pathlib import Path

import jax
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.safety_gym_safe_wrapper import SafeSafetyGymWrapper
from scripts.safetygym_eval_viz import collect_scene, plot_safetygym_eval_diagnostics, plot_safetygym_eval_trajectory, save_records
from scripts.train_safe_safetygym import make_algo


def make_dummy_args(seed=0):
    return argparse.Namespace(
        seed=seed, diffusion_steps=10, num_ent_timesteps=10, alpha_value=0.1, fixed_alpha=False, init_alpha=0.1,
        policy_noise_scale=0.3, gamma=0.99, gamma_p=0.99, lr=3e-4, alpha_lr=1e-2, sample_k=256, lambda_p=0.0,
        use_projection_critic=False, lambda_p_warmup_steps=0, use_tn_energy=False, entropy_reg_mode='legacy',
        use_filter_surrogate=False, surrogate_warmup_steps=0, surrogate_loss_coef=1.0, lambda_raw_norm=0.0,
        use_directional_noise=False,
    )


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env_id', default='SafetyPointGoal1-v0')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--render_mode', default=None)
    p.add_argument('--use_filter', action='store_true')
    p.add_argument('--filter_type', default='hybrid')
    p.add_argument('--save_dir', type=str, default=None)
    p.add_argument('--save_trajectory', action='store_true')
    p.add_argument('--max_steps', type=int, default=1000)
    p.add_argument('--arrow_stride', type=int, default=25)
    p.add_argument('--plot_diagnostics', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    save_dir = Path(args.save_dir) if args.save_dir else Path(args.checkpoint).resolve().parent / 'visualizations'
    if args.save_trajectory:
        save_dir.mkdir(parents=True, exist_ok=True)

    env = SafeSafetyGymWrapper(env_id=args.env_id, use_filter=args.use_filter, filter_type=args.filter_type, render_mode=args.render_mode)
    agent = make_algo(make_dummy_args(args.seed), env.observation_space.shape[0], env.action_space.shape[0])
    agent.state = pickle.load(open(args.checkpoint, 'rb'))

    safe_filter = getattr(env, 'safe_filter', None)
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        scene = collect_scene(env, safe_filter) if safe_filter is not None else {'hazards': [], 'objects': [], 'goal': None}
        done = False
        ret = 0.0
        costs, fars, aprs, sv = [], [], [], []
        records = []
        info = {}
        l = 0
        while not done and l < args.max_steps:
            raw_action = np.asarray(agent.get_action(jax.random.PRNGKey(args.seed + ep * 10000 + l), obs[None])[0])
            obs, reward, term, trunc, info = env.step(raw_action)
            done = term or trunc
            exec_action = np.asarray(info.get('exec_action', raw_action), dtype=np.float32)
            ego = safe_filter._extract_ego_state_from_env(env) if safe_filter is not None else {'pos': np.zeros(2, np.float32)}
            records.append({
                't': l, 'ego_x': float(ego['pos'][0]), 'ego_y': float(ego['pos'][1]), 'raw_action': raw_action, 'exec_action': exec_action,
                'reward': float(reward), 'cost': float(info.get('cost', 0.0)), 'filter_active': float(info.get('filter_active', 0.0)),
                'projection_residual': float(info.get('projection_residual', 0.0)), 'raw_action_norm': float(info.get('raw_action_norm', 0.0)),
                'exec_action_norm': float(info.get('exec_action_norm', 0.0)), 'current_min_h': float(info.get('current_min_h', np.nan)),
                'predicted_min_h': float(info.get('predicted_min_h', np.nan)), 'emergency_active': float(info.get('emergency_active', 0.0)),
                'safe_candidate_ratio': float(info.get('safe_candidate_ratio', np.nan)),
            })
            l += 1
            ret += float(reward)
            costs.append(float(info.get('cost', 0.0))); fars.append(float(info.get('filter_active', 0.0))); aprs.append(float(info.get('projection_residual', 0.0))); sv.append(float(info.get('safety_violation', 0.0)))

        succ = float(info.get('is_success', info.get('success', info.get('goal_met', info.get('task_success', 0.0)))))
        summary = dict(return_=ret, cost_return=float(np.sum(costs)), episode_length=l, success=succ, FAR=float(np.mean(fars)) if fars else 0.0, APR=float(np.mean(aprs)) if aprs else 0.0, safety_violation=float(np.mean(sv)) if sv else 0.0)
        print(summary)

        if args.save_trajectory:
            prefix = save_dir / f'ep{ep:03d}'
            save_records(records, prefix)
            title = f"{args.env_id} ep={ep} return={summary['return_']:.2f} cost={summary['cost_return']:.2f} FAR={summary['FAR']:.3f} APR={summary['APR']:.3f} success={summary['success']:.2f}"
            plot_safetygym_eval_trajectory(records, scene, save_path=str(prefix) + '_trajectory.png', title=title, arrow_stride=args.arrow_stride)
            if args.plot_diagnostics:
                plot_safetygym_eval_diagnostics(records, save_path=str(prefix) + '_diagnostics.png', title=title)
    if args.save_trajectory:
        print(f'[saved] {save_dir}')
