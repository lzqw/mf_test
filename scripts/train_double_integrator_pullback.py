import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import NamedTuple

import jax
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

from envs.safe_obstacle_double_integrator_2d import SafeObstacleDoubleIntegrator2DEnv
from scripts.safe_pullback_experience import SafePullbackExperience
from eval.eval_double_integrator_pullback import (
    make_algo,
    obs_to_algo_obs,
    to_real_action,
)


class Batch(NamedTuple):
    obs: np.ndarray
    raw_action: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    next_obs: np.ndarray
    projection_residual: np.ndarray
    projection_cost: np.ndarray
    safe_violation: np.ndarray
    filter_active: np.ndarray
    state_violation: np.ndarray
    is_success: np.ndarray
    cost: np.ndarray


def make_batch(buf, batch_size):
    idx = np.random.randint(0, len(buf), size=batch_size)
    items = [buf[i] for i in idx]
    return Batch(
        obs=np.stack([x.obs for x in items]).astype(np.float32),
        raw_action=np.stack([x.raw_action for x in items]).astype(np.float32),
        action=np.stack([x.action for x in items]).astype(np.float32),
        reward=np.asarray([x.reward for x in items], dtype=np.float32),
        done=np.asarray([x.done for x in items], dtype=np.float32),
        next_obs=np.stack([x.next_obs for x in items]).astype(np.float32),
        projection_residual=np.asarray([x.projection_residual for x in items], dtype=np.float32),
        projection_cost=np.asarray([x.projection_cost for x in items], dtype=np.float32),
        safe_violation=np.asarray([x.safe_violation for x in items], dtype=bool),
        filter_active=np.asarray([x.filter_active for x in items], dtype=bool),
        state_violation=np.asarray([x.state_violation for x in items], dtype=bool),
        is_success=np.asarray([x.is_success for x in items], dtype=bool),
        cost=np.asarray([x.cost for x in items], dtype=np.float32),
    )


def make_algo_from_args(args):
    return make_algo(args)


def configure_algo_mode(args):
    if args.algo == "vanilla_flow":
        args.use_projection_critic = False
        args.lambda_p = 0.0
        args.use_tn_energy = False
        args.entropy_reg_mode = "legacy"
        args.safe_energy_variant = "normal_iso"
        args.safe_iso_coef = args.safe_iso_coef
        args.normal_energy_coef = 0.0
        args.target_safe_energy = 0.0
        args.tn_coef = 1.0
        args.weight_mix = 0.05
    elif args.algo == "curvature_flow":
        args.use_projection_critic = False
        args.lambda_p = 0.0
        args.use_tn_energy = True
        args.entropy_reg_mode = "legacy"
        args.safe_energy_variant = "normal_tangent"
        args.safe_iso_coef = max(0.0, args.curvature_robust_iso)
        args.normal_energy_coef = args.curvature_normal_energy_coef
        args.target_safe_energy = args.curvature_target_safe_energy
        args.tn_coef = args.curvature_tn_coef
        args.sigma_n = args.curvature_sigma_n
        args.sigma_t = args.curvature_sigma_t
        args.weight_mix = args.weight_mix
    else:
        raise ValueError(f"Unsupported algo: {args.algo}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["vanilla_flow", "curvature_flow"], default="vanilla_flow")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total_steps", type=int, default=300000)
    p.add_argument("--start_steps", type=int, default=20000)
    p.add_argument("--update_after", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--eval_interval", type=int, default=5000)
    p.add_argument("--outdir", required=True)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--alpha_lr", type=float, default=1e-2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gamma_p", type=float, default=0.99)
    p.add_argument("--sample_k", type=int, default=256)
    p.add_argument("--start_y_range", type=float, default=0.45)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--a_max", type=float, default=3.0)
    p.add_argument("--v_max", type=float, default=2.0)
    p.add_argument("--damping", type=float, default=0.98)
    p.add_argument("--episode_len", type=int, default=200)
    p.add_argument("--diffusion_steps", type=int, default=10)
    p.add_argument("--num_ent_timesteps", type=int, default=10)
    p.add_argument("--alpha_value", type=float, default=0.1)
    p.add_argument("--init_alpha", type=float, default=0.1)
    p.add_argument("--policy_noise_scale", type=float, default=0.3)
    p.add_argument("--hidden_sizes", type=str, default="256,256,256")
    p.add_argument("--diffusion_hidden_sizes", type=str, default="256,256,256")
    p.add_argument("--weight_mix", type=float, default=0.05)
    p.add_argument("--update_every", type=int, default=1)
    p.add_argument("--use_projection_critic", action="store_true", default=False)
    p.add_argument("--fixed_alpha", action="store_true", default=False)
    p.add_argument("--lambda_p_warmup_steps", type=int, default=100000)
    p.add_argument("--lambda_d", type=float, default=0.5)
    p.add_argument("--use_frpi_score", action="store_true", default=False)
    p.add_argument("--tau_c", type=float, default=1.0)
    p.add_argument("--mu_c", type=float, default=1.0)
    p.add_argument("--lambda_f", type=float, default=2.0)
    p.add_argument("--use_tn_energy", action="store_true", default=False)
    p.add_argument("--tn_coef", type=float, default=1.0)
    p.add_argument("--sigma_n", type=float, default=0.2)
    p.add_argument("--sigma_t", type=float, default=1.0)
    p.add_argument("--tn_r_min", type=float, default=0.02)
    p.add_argument("--tn_r_max", type=float, default=0.20)
    p.add_argument("--tn_clip", type=float, default=10.0)
    p.add_argument("--kappa_tn", type=float, default=1.0)
    p.add_argument("--candidate_temp", type=float, default=0.10)
    p.add_argument("--beta_normal_entropy", type=float, default=1.0)
    p.add_argument("--min_effective_entropy", type=float, default=-20.0)
    p.add_argument("--target_effective_entropy", type=float, default=1.0)
    p.add_argument("--normal_energy_coef", type=float, default=0.05)
    p.add_argument("--target_safe_energy", type=float, default=0.05)
    p.add_argument("--safe_iso_coef", type=float, default=0.2)
    p.add_argument("--entropy_reg_mode", choices=["legacy", "likelihood_tn", "flac_tn"], default="legacy")
    # curvature-only knobs
    p.add_argument("--curvature_sigma_n", type=float, default=0.10)
    p.add_argument("--curvature_sigma_t", type=float, default=1.0)
    p.add_argument("--curvature_tn_coef", type=float, default=0.5)
    p.add_argument("--curvature_normal_energy_coef", type=float, default=0.2)
    p.add_argument("--curvature_target_safe_energy", type=float, default=0.08)
    p.add_argument("--curvature_robust_iso", type=float, default=0.2)
    p.add_argument("--eval_episodes", type=int, default=100)
    p.add_argument("--eval_delta", type=float, default=0.0)
    p.add_argument("--save_eval_rollouts", action="store_true", default=False)
    p.add_argument("--reward_progress_coef", type=float, default=8.0)
    p.add_argument("--reward_success_bonus", type=float, default=100.0)
    p.add_argument("--reward_collision_penalty", type=float, default=100.0)
    p.add_argument("--reward_near_obs_coef", type=float, default=8.0)
    p.add_argument("--reward_safety_buffer", type=float, default=0.20)
    p.add_argument("--reward_action_coef", type=float, default=0.03)
    p.add_argument("--reward_speed_coef", type=float, default=0.01)
    p.add_argument("--reward_time_coef", type=float, default=0.01)
    args = p.parse_args()

    configure_algo_mode(args)

    np.random.seed(args.seed)
    env_reward_cfg = dict(
        progress_coef=args.reward_progress_coef,
        success_bonus=args.reward_success_bonus,
        collision_penalty=args.reward_collision_penalty,
        near_obs_coef=args.reward_near_obs_coef,
        safety_buffer=args.reward_safety_buffer,
        action_coef=args.reward_action_coef,
        speed_coef=args.reward_speed_coef,
        time_coef=args.reward_time_coef,
    )
    env = SafeObstacleDoubleIntegrator2DEnv(
        noise_sigma=(0.0, 0.0),
        use_filter=True,
        seed=args.seed,
        start_y_range=args.start_y_range,
        dt=args.dt,
        a_max=args.a_max,
        v_max=args.v_max,
        damping=args.damping,
        episode_len=args.episode_len,
        reward_cfg=env_reward_cfg,
    )
    env.set_action_gain(1.0 - float(args.eval_delta))

    agent = make_algo_from_args(args)

    key = jax.random.PRNGKey(args.seed + 17)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out_dir / "tb"))
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    buffer = []
    train_log = []
    eval_log = []

    obs_real, _ = env.reset(seed=args.seed)
    obs = obs_to_algo_obs(obs_real)

    try:
        for step in range(1, args.total_steps + 1):
            if step < args.start_steps:
                raw_action_algo = env.action_space.sample().astype(np.float32)
            else:
                key, ak = jax.random.split(key)
                raw_action_algo = np.asarray(agent.get_action(ak, obs[None, :])[0], dtype=np.float32)

            raw_action_real = to_real_action(raw_action_algo)
            next_obs_real, reward, terminated, truncated, info = env.step(raw_action_real)
            next_obs_algo = obs_to_algo_obs(next_obs_real)

            exec_action_algo = to_real_action(np.asarray(info["exec_action"], dtype=np.float32))
            projection_residual = float(np.linalg.norm(exec_action_algo - raw_action_algo))
            projection_cost = float(projection_residual ** 2)

            exp = SafePullbackExperience(
                obs=obs,
                raw_action=raw_action_algo.astype(np.float32),
                action=exec_action_algo.astype(np.float32),
                reward=float(reward),
                done=np.bool_(terminated or truncated),
                next_obs=next_obs_algo,
                projection_residual=projection_residual,
                projection_cost=projection_cost,
                safe_violation=bool(info["safe_violation"]),
                filter_active=bool(info["filter_activated"]),
                state_violation=bool(info["state_violation"]),
                is_success=bool(info["success"]),
                cost=0.0,
            )
            buffer.append(exp)
            if len(buffer) > 1_000_000:
                buffer.pop(0)

            obs = next_obs_algo
            if terminated or truncated:
                obs_real, _ = env.reset(seed=args.seed + step)
                obs = obs_to_algo_obs(obs_real)

            writer.add_scalar("train/buffer_size", float(len(buffer)), step)

            env_scalar_keys = (
                ("projection_cost", "train_env/projection_cost"),
                ("filter_active", "train_env/filter_active"),
                ("safe_violation", "train_env/safe_violation"),
                ("state_violation", "train_env/state_violation"),
                ("reward", "train_env/reward"),
            )
            info_scalar = {
                "projection_cost": projection_cost,
                "filter_active": bool(info["filter_activated"]),
                "safe_violation": bool(info["safe_violation"]),
                "state_violation": bool(info["state_violation"]),
                "reward": float(reward),
            }
            for info_key, tag in env_scalar_keys:
                writer.add_scalar(tag, float(info_scalar[info_key]), step)

            writer.add_scalar("train_env/reward", float(reward), step)

            if (
                step >= args.update_after
                and step % args.update_every == 0
                and len(buffer) >= args.batch_size
            ):
                key, uk = jax.random.split(key)
                batch = make_batch(buffer, args.batch_size)
                out = agent.update(uk, batch)
                out["step"] = step
                train_log.append(out)
                for k, v in out.items():
                    if k == "step":
                        continue
                    writer.add_scalar(f"train/{k}", float(v), step)
                writer.flush()

            if step % args.eval_interval == 0:
                from eval.eval_double_integrator_pullback import collect_double_integrator_eval_rollouts

                eval_agent = agent
                eval_rollouts, eval_summary = collect_double_integrator_eval_rollouts(
                    eval_agent,
                    args.algo,
                    episodes=args.eval_episodes,
                    seed=args.seed + step,
                    delta=float(args.eval_delta),
                    start_y_range=args.start_y_range,
                    dt=args.dt,
                    a_max=args.a_max,
                    use_handcrafted_controller=False,
                )
                eval_dir = out_dir / "eval_rollouts" / f"step_{step:08d}"
                if args.save_eval_rollouts:
                    eval_dir.mkdir(parents=True, exist_ok=True)
                    np.savez(eval_dir / "rollouts.npz", **eval_rollouts)
                    (eval_dir / "summary.json").write_text(json.dumps(eval_summary, indent=2))
                eval_summary["step"] = step
                eval_log.append(eval_summary)
                for k, v in eval_summary.items():
                    if k == "step":
                        continue
                    writer.add_scalar(f"eval/{k}", float(v), step)
                writer.flush()

                print(
                    f"[step {step}] eval return={eval_summary.get('return_mean', float('nan')):.4f}, "
                    f"success={eval_summary.get('success_rate', float('nan')):.4f}, "
                    f"collision={eval_summary.get('collision_rate', float('nan')):.4f}, "
                    f"h_min={eval_summary.get('h_min_mean', float('nan')):.4f}"
                )

    finally:
        with open(out_dir / "train_metrics.pkl", "wb") as f:
            pickle.dump(train_log, f)
        with open(out_dir / "checkpoint.pkl", "wb") as f:
            pickle.dump({
                "algo": args.algo,
                "seed": args.seed,
                "args": vars(args),
                "agent_state": agent.state,
                "outdir": str(out_dir),
            }, f)
        with open(out_dir / "eval_metrics.pkl", "wb") as f:
            pickle.dump(eval_log, f)

        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
