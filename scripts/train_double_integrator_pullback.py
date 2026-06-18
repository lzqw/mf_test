import argparse
import csv
import json
import pickle
import sys
import time
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
from eval.eval_double_integrator_pullback import (
    CHECKPOINT_OBS_DIM_MISMATCH_MSG,
    make_algo,
    obs_to_algo_obs,
    to_real_action,
)
from scripts.safe_pullback_experience import SafePullbackExperience


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


RUNTIME_OVERRIDE_KEYS = {
    "checkpoint_name",
    "eval_interval",
    "max_walltime_sec",
    "outdir",
    "resume_checkpoint",
    "save_interval",
    "total_steps",
}


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
    return make_algo(args, obs_dim=getattr(args, "obs_dim", 10))


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


def _save_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("")
        return
    headers = []
    for row in rows:
        for key in row.keys():
            if key not in headers:
                headers.append(key)
    if "step" in headers:
        headers = ["step"] + [h for h in headers if h != "step"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_pickle_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "rb") as f:
        data = pickle.load(f)
    return list(data) if isinstance(data, list) else []


def _metric_source_dirs(out_dir, resume_checkpoint):
    dirs = [Path(out_dir)]
    if resume_checkpoint:
        ckpt_dir = Path(resume_checkpoint).resolve().parent
        if ckpt_dir not in dirs:
            dirs.append(ckpt_dir)
    return dirs


def _load_existing_metrics(out_dir, resume_checkpoint):
    for src_dir in _metric_source_dirs(out_dir, resume_checkpoint):
        train_path = src_dir / "train_metrics.pkl"
        eval_path = src_dir / "eval_metrics.pkl"
        if train_path.exists() or eval_path.exists():
            return _load_pickle_rows(train_path), _load_pickle_rows(eval_path)
    return [], []


def _checkpoint_payload(agent, args, step, out_dir, action_dim=2):
    return {
        "algo": args.algo,
        "seed": args.seed,
        "step": int(step),
        "total_steps": int(args.total_steps),
        "obs_dim": int(args.obs_dim),
        "action_dim": int(action_dim),
        "args": vars(args),
        "agent_state": jax.device_get(agent.state),
        "outdir": str(out_dir),
    }


def _save_checkpoint(path, payload):
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def _write_metrics(out_dir, train_log, eval_log, interrupted=False):
    out_dir = Path(out_dir)
    with open(out_dir / "train_metrics.pkl", "wb") as f:
        pickle.dump(train_log, f)
    with open(out_dir / "eval_metrics.pkl", "wb") as f:
        pickle.dump(eval_log, f)
    _save_csv(out_dir / "train_metrics.csv", train_log)
    _save_csv(out_dir / "eval_metrics.csv", eval_log)
    if interrupted:
        with open(out_dir / "train_metrics_interrupted.pkl", "wb") as f:
            pickle.dump(train_log, f)


def _write_training_state(out_dir, step, args, last_checkpoint):
    state = {
        "step": int(step),
        "total_steps": int(args.total_steps),
        "algo": args.algo,
        "seed": int(args.seed),
        "obs_dim": int(args.obs_dim),
        "action_dim": 2,
        "last_checkpoint": str(last_checkpoint),
    }
    (Path(out_dir) / "training_state.json").write_text(json.dumps(state, indent=2))


def _save_periodic_state(agent, args, out_dir, step, train_log, eval_log):
    out_dir = Path(out_dir)
    step_checkpoint = out_dir / f"checkpoint_step_{step:08d}.pkl"
    latest_checkpoint = out_dir / "checkpoint_latest.pkl"
    payload = _checkpoint_payload(agent, args, step, out_dir)
    _save_checkpoint(step_checkpoint, payload)
    _save_checkpoint(latest_checkpoint, payload)
    _write_metrics(out_dir, train_log, eval_log, interrupted=False)
    _write_training_state(out_dir, step, args, latest_checkpoint)
    return latest_checkpoint


def _save_final_state(agent, args, out_dir, step, train_log, eval_log):
    out_dir = Path(out_dir)
    final_checkpoint = out_dir / args.checkpoint_name
    latest_checkpoint = out_dir / "checkpoint_latest.pkl"
    payload = _checkpoint_payload(agent, args, step, out_dir)
    _save_checkpoint(final_checkpoint, payload)
    _save_checkpoint(latest_checkpoint, payload)
    _write_metrics(out_dir, train_log, eval_log, interrupted=False)
    _write_training_state(out_dir, step, args, latest_checkpoint)
    return final_checkpoint


def _save_interrupted_state(agent, args, out_dir, step, train_log, eval_log):
    out_dir = Path(out_dir)
    interrupted_checkpoint = out_dir / f"checkpoint_interrupted_step_{step:08d}.pkl"
    latest_checkpoint = out_dir / "checkpoint_latest.pkl"
    payload = _checkpoint_payload(agent, args, step, out_dir)
    _save_checkpoint(interrupted_checkpoint, payload)
    _save_checkpoint(latest_checkpoint, payload)
    _write_metrics(out_dir, train_log, eval_log, interrupted=True)
    _write_training_state(out_dir, step, args, latest_checkpoint)
    return interrupted_checkpoint


def _restore_args_from_checkpoint(args, checkpoint_args):
    for key, value in checkpoint_args.items():
        if key in RUNTIME_OVERRIDE_KEYS:
            continue
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def _load_resume_checkpoint(args):
    checkpoint_path = Path(args.resume_checkpoint)
    try:
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
    except Exception:
        cpu_device = jax.devices("cpu")[0]
        with open(checkpoint_path, "rb") as f:
            with jax.default_device(cpu_device):
                ckpt = pickle.load(f)
    saved_args = ckpt.get("args", {})
    saved_obs_dim = saved_args.get("obs_dim", ckpt.get("obs_dim", None))
    if saved_obs_dim is None or int(saved_obs_dim) <= 0:
        raise ValueError(CHECKPOINT_OBS_DIM_MISMATCH_MSG)
    checkpoint_algo = ckpt.get("algo", saved_args.get("algo", None))
    if checkpoint_algo is not None and checkpoint_algo != args.algo:
        raise ValueError(f"resume checkpoint algo mismatch: checkpoint={checkpoint_algo}, requested={args.algo}")

    args = _restore_args_from_checkpoint(args, saved_args)
    args.obs_dim = int(saved_obs_dim)
    configure_algo_mode(args)
    agent = make_algo_from_args(args)
    try:
        agent.state = ckpt["agent_state"]
    except Exception as exc:
        raise ValueError(CHECKPOINT_OBS_DIM_MISMATCH_MSG) from exc
    start_step = int(ckpt.get("step", 0) or 0)
    return agent, start_step, args


def _make_env(args):
    env_reward_cfg = dict(
        progress_coef=args.reward_progress_coef,
        success_bonus=args.reward_success_bonus,
        collision_penalty=args.reward_collision_penalty,
        near_obs_coef=args.reward_near_obs_coef,
        safety_buffer=args.reward_safety_buffer,
        action_coef=args.reward_action_coef,
        speed_coef=args.reward_speed_coef,
        time_coef=args.reward_time_coef,
        route_softmin_beta=args.reward_route_softmin_beta,
        route_start_bias_scale=args.reward_route_start_bias_scale,
        goal_progress_mix=args.reward_goal_progress_mix,
        terminal_goal_bonus_radius=args.terminal_goal_bonus_radius,
        terminal_goal_bonus_coef=args.terminal_goal_bonus_coef,
    )
    env = SafeObstacleDoubleIntegrator2DEnv(
        noise_sigma=(0.0, 0.0),
        use_filter=True,
        seed=args.seed,
        map_id=args.map_id,
        route_variant=args.route_variant,
        obs_mode=args.obs_mode,
        start_y_range=args.start_y_range,
        dt=args.dt,
        a_max=args.a_max,
        v_max=args.v_max,
        damping=args.damping,
        episode_len=args.episode_len,
        goal_radius=args.goal_radius,
        reward_mode=args.reward_mode,
        reward_cfg=env_reward_cfg,
    )
    env.set_action_gain(1.0 - float(args.eval_delta))
    return env


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["vanilla_flow", "curvature_flow"], default="vanilla_flow")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total_steps", type=int, default=300000)
    p.add_argument("--start_steps", type=int, default=20000)
    p.add_argument("--update_after", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--eval_interval", type=int, default=5000)
    p.add_argument("--save_interval", type=int, default=5000)
    p.add_argument("--outdir", required=True)
    p.add_argument("--resume_checkpoint", type=str, default="")
    p.add_argument("--checkpoint_name", type=str, default="checkpoint.pkl")
    p.add_argument("--max_walltime_sec", type=float, default=0.0)
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
    p.add_argument("--goal_radius", type=float, default=0.18)
    p.add_argument("--map_id", choices=["single_circle", "three_circles"], default="single_circle")
    p.add_argument("--route_variant", type=str, default="baseline")
    p.add_argument("--obs_mode", choices=["single_obstacle", "all_obstacles"], default=None)
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
    p.add_argument("--curvature_sigma_n", type=float, default=0.10)
    p.add_argument("--curvature_sigma_t", type=float, default=1.0)
    p.add_argument("--curvature_tn_coef", type=float, default=0.5)
    p.add_argument("--curvature_normal_energy_coef", type=float, default=0.2)
    p.add_argument("--curvature_target_safe_energy", type=float, default=0.08)
    p.add_argument("--curvature_robust_iso", type=float, default=0.2)
    p.add_argument("--eval_episodes", type=int, default=100)
    p.add_argument("--eval_delta", type=float, default=0.0)
    p.add_argument("--obs_dim", type=int, default=10)
    p.add_argument("--save_eval_rollouts", action="store_true", default=False)
    p.add_argument("--reward_progress_coef", type=float, default=8.0)
    p.add_argument("--reward_success_bonus", type=float, default=100.0)
    p.add_argument("--reward_collision_penalty", type=float, default=100.0)
    p.add_argument("--reward_near_obs_coef", type=float, default=8.0)
    p.add_argument("--reward_safety_buffer", type=float, default=0.20)
    p.add_argument("--reward_action_coef", type=float, default=0.03)
    p.add_argument("--reward_speed_coef", type=float, default=0.01)
    p.add_argument("--reward_time_coef", type=float, default=0.01)
    p.add_argument("--reward_route_softmin_beta", type=float, default=0.0)
    p.add_argument("--reward_route_start_bias_scale", type=float, default=0.0)
    p.add_argument("--reward_goal_progress_mix", type=float, default=0.0)
    p.add_argument("--terminal_goal_bonus_radius", type=float, default=0.0)
    p.add_argument("--terminal_goal_bonus_coef", type=float, default=0.0)
    p.add_argument("--reward_mode", choices=["goal_progress", "symmetric_path_progress", "multi_route_progress"], default="goal_progress")
    return p.parse_args()


def main():
    args = parse_args()
    args.outdir = str(Path(args.outdir))
    args.resume_checkpoint = str(args.resume_checkpoint or "")
    args.obs_dim = int(args.obs_dim)
    configure_algo_mode(args)
    np.random.seed(args.seed)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out_dir / "tb"))
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    train_log, eval_log = _load_existing_metrics(out_dir, args.resume_checkpoint)

    env = _make_env(args)
    if args.resume_checkpoint:
        agent, start_step, args = _load_resume_checkpoint(args)
        env = _make_env(args)
    else:
        args.obs_dim = int(env.observation_space.shape[0])
        agent = make_algo_from_args(args)
        start_step = 0

    args.obs_dim = int(env.observation_space.shape[0])
    args.outdir = str(out_dir)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    key = jax.random.PRNGKey(args.seed + 17 + start_step)
    obs_real, _ = env.reset(seed=args.seed + start_step)
    obs = obs_to_algo_obs(obs_real)

    buffer = []
    last_step = int(start_step)
    periodic_saved = False
    interrupted = False
    walltime_stop = False
    last_checkpoint = out_dir / "checkpoint_latest.pkl"
    train_start_time = time.perf_counter()

    try:
        for step in range(start_step + 1, args.total_steps + 1):
            last_step = step

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
            if step % 100 == 0:
                print(
                    f"[train] step={step}/{args.total_steps}, reward={float(reward):.4f}, "
                    f"buffer={len(buffer)}, terminated={terminated}, truncated={truncated}",
                    flush=True,
                )

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
            info_scalar = {
                "projection_cost": projection_cost,
                "filter_active": bool(info["filter_activated"]),
                "safe_violation": bool(info["safe_violation"]),
                "state_violation": bool(info["state_violation"]),
                "reward": float(reward),
            }
            for info_key, tag in (
                ("projection_cost", "train_env/projection_cost"),
                ("filter_active", "train_env/filter_active"),
                ("safe_violation", "train_env/safe_violation"),
                ("state_violation", "train_env/state_violation"),
                ("reward", "train_env/reward"),
            ):
                writer.add_scalar(tag, float(info_scalar[info_key]), step)

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
                for key_name, value in out.items():
                    if key_name == "step":
                        continue
                    writer.add_scalar(f"train/{key_name}", float(value), step)
                writer.flush()

            if args.eval_interval > 0 and step % args.eval_interval == 0:
                from eval.eval_double_integrator_pullback import collect_double_integrator_eval_rollouts

                eval_rollouts, eval_summary = collect_double_integrator_eval_rollouts(
                    agent,
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
                for key_name, value in eval_summary.items():
                    if key_name == "step":
                        continue
                    writer.add_scalar(f"eval/{key_name}", float(value), step)
                writer.flush()
                print(
                    f"[step {step}] eval return={eval_summary.get('return_mean', float('nan')):.4f}, "
                    f"success={eval_summary.get('success_rate', float('nan')):.4f}, "
                    f"collision={eval_summary.get('collision_rate', float('nan')):.4f}, "
                    f"h_min={eval_summary.get('h_min_mean', float('nan')):.4f}",
                    flush=True,
                )

            if args.save_interval > 0 and step % args.save_interval == 0:
                last_checkpoint = _save_periodic_state(agent, args, out_dir, step, train_log, eval_log)
                periodic_saved = True
                print(f"[save] step={step}, checkpoint={last_checkpoint}", flush=True)

            if args.max_walltime_sec > 0.0:
                elapsed = time.perf_counter() - train_start_time
                if elapsed >= float(args.max_walltime_sec):
                    walltime_stop = True
                    print(
                        f"[stop] reached max walltime at step={step}, elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                    break
    except BaseException:
        interrupted = True
        raise
    finally:
        if interrupted:
            last_checkpoint = _save_interrupted_state(agent, args, out_dir, last_step, train_log, eval_log)
        else:
            if not periodic_saved and last_step > start_step:
                last_checkpoint = _save_periodic_state(agent, args, out_dir, last_step, train_log, eval_log)
            last_checkpoint = _save_final_state(agent, args, out_dir, last_step, train_log, eval_log)
            if walltime_stop:
                print(f"[final] walltime stop checkpoint saved to {last_checkpoint}", flush=True)
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
