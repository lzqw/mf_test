import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.safety_gym_safe_wrapper import SafeSafetyGymWrapper
from scripts.safetygym_eval_viz import collect_scene, plot_safetygym_eval_diagnostics, plot_safetygym_eval_trajectory, save_records

SUPPORTED_ENVS = {
    "SafetyPointGoal1-v0",
    "SafetyPointPush1-v0",
    "SafetyCarGoal1-v0",
    "SafetyCarPush1-v0",
}


def _safe_info_float(info: Dict, key: str, default: float = 0.0) -> float:
    v = info.get(key, default)
    if isinstance(v, (str, bytes)):
        return float(default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def save_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def evaluate(model, env, eval_episodes: int, seed: int) -> Dict[str, float]:
    per_ep = []
    for ep in range(eval_episodes):
        obs, _ = env.reset(seed=seed + 1000 + ep)
        done = False
        ep_ret = 0.0
        ep_len = 0
        costs = []
        safety_viol = []
        constraint_viol = []
        raw_norms = []
        exec_norms = []
        goal_dists = []
        goal_met_steps = []
        goal_reached_steps = []
        info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += float(reward)
            ep_len += 1
            costs.append(_safe_info_float(info, "cost", 0.0))
            safety_viol.append(_safe_info_float(info, "safety_violation", 0.0))
            constraint_viol.append(_safe_info_float(info, "constraint_violation", 0.0))
            raw_norms.append(_safe_info_float(info, "raw_action_norm", 0.0))
            exec_norms.append(_safe_info_float(info, "exec_action_norm", 0.0))
            goal_dists.append(_safe_info_float(info, "goal_dist", np.nan))
            goal_met_steps.append(_safe_info_float(info, "goal_met", 0.0))
            goal_reached_steps.append(_safe_info_float(info, "goal_reached_by_dist", 0.0))

        goal_met = _safe_info_float(info, "goal_met", 0.0)
        goal_reached = _safe_info_float(info, "goal_reached_by_dist", 0.0)
        success = max(_safe_info_float(info, "is_success", 0.0), goal_met, goal_reached)

        per_ep.append(
            {
                "return_": ep_ret,
                "episode_length": float(ep_len),
                "success_rate": success,
                "cost_return": float(np.sum(costs)) if costs else 0.0,
                "cost_rate": float(np.mean(np.asarray(costs) > 0.0)) if costs else 0.0,
                "safety_violation_rate": float(np.mean(safety_viol)) if safety_viol else 0.0,
                "constraint_violation_rate": float(np.mean(constraint_viol)) if constraint_viol else 0.0,
                "raw_action_norm": float(np.mean(raw_norms)) if raw_norms else 0.0,
                "exec_action_norm": float(np.mean(exec_norms)) if exec_norms else 0.0,
                "goal_dist_mean": float(np.nanmean(np.asarray(goal_dists, dtype=np.float32))) if goal_dists else np.nan,
                "goal_dist_final": _safe_info_float(info, "goal_dist", np.nan),
                "goal_met_rate": goal_met,
                "goal_reached_by_dist_rate": goal_reached,
            }
        )

    return {k: float(np.mean([row[k] for row in per_ep])) for k in per_ep[0].keys()}


def rollout_eval_trajectories(model, env, args, step: int) -> Optional[Path]:
    out_dir = Path(args.log_dir) / "eval_trajectories" / f"step_{step}"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_filter = getattr(env, "safe_filter", None)
    for ep in range(args.eval_traj_episodes):
        obs, _ = env.reset(seed=args.seed + 2000 + step + ep)
        scene = collect_scene(env, safe_filter) if safe_filter is not None else {"hazards": [], "objects": [], "goal": None}
        records = []
        done = False
        t = 0
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            ego = safe_filter._extract_ego_state_from_env(env) if safe_filter is not None else {"pos": np.zeros(2, np.float32)}
            records.append(
                {
                    "t": t,
                    "ego_x": float(ego["pos"][0]),
                    "ego_y": float(ego["pos"][1]),
                    "raw_action": np.asarray(info.get("raw_action", action), dtype=np.float32),
                    "exec_action": np.asarray(info.get("exec_action", action), dtype=np.float32),
                    "reward": float(reward),
                    "cost": _safe_info_float(info, "cost", 0.0),
                    "projection_residual": _safe_info_float(info, "projection_residual", 0.0),
                    "raw_action_norm": _safe_info_float(info, "raw_action_norm", 0.0),
                    "exec_action_norm": _safe_info_float(info, "exec_action_norm", 0.0),
                    "emergency_active": _safe_info_float(info, "emergency_active", 0.0),
                    "goal_dist": _safe_info_float(info, "goal_dist", np.nan),
                    "goal_met": _safe_info_float(info, "goal_met", 0.0),
                    "goal_reached_by_dist": _safe_info_float(info, "goal_reached_by_dist", 0.0),
                }
            )
            t += 1
        prefix = out_dir / f"ep{ep:03d}"
        save_records(records, prefix)
        goal_met = _safe_info_float(info, "goal_met", 0.0)
        goal_reached = _safe_info_float(info, "goal_reached_by_dist", 0.0)
        succ = max(_safe_info_float(info, "is_success", 0.0), goal_met, goal_reached)
        final_goal_dist = _safe_info_float(info, "goal_dist", np.nan)
        title = f"{args.env_id} step={step} ep={ep} final_goal_dist={final_goal_dist:.3f} goal_met={goal_met:.2f} success={succ:.2f}"
        plot_safetygym_eval_trajectory(records, scene, save_path=str(prefix) + "_trajectory.png", title=title, arrow_stride=args.eval_traj_stride)
        plot_safetygym_eval_diagnostics(records, save_path=str(prefix) + "_diagnostics.png", title=title)
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="SafetyPointGoal1-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=50000)
    parser.add_argument("--eval_interval", type=int, default=5000)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--buffer_size", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--train_freq", type=int, default=1)
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--learning_starts", type=int, default=5000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_eval_trajectories", action="store_true")
    parser.add_argument("--eval_traj_episodes", type=int, default=1)
    parser.add_argument("--eval_traj_stride", type=int, default=25)
    args = parser.parse_args()

    if args.env_id not in SUPPORTED_ENVS:
        raise ValueError(f"Unsupported env_id={args.env_id}, supported={sorted(SUPPORTED_ENVS)}")

    try:
        from stable_baselines3 import SAC
    except Exception:
        print('Failed to import stable-baselines3. Please install with: pip install "stable-baselines3[extra]"')
        raise

    np.random.seed(args.seed)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    env = SafeSafetyGymWrapper(env_id=args.env_id, use_filter=False, filter_type="none")
    eval_env = SafeSafetyGymWrapper(env_id=args.env_id, use_filter=False, filter_type="none")

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        learning_starts=args.learning_starts,
        tensorboard_log=str(log_dir / "tb"),
        seed=args.seed,
        verbose=1,
        device=args.device,
    )

    eval_metrics: List[Dict] = []
    best_return = -float("inf")
    best_cost = float("inf")
    best_safety = float("inf")

    def persist():
        with (log_dir / "eval_metrics.pkl").open("wb") as f:
            pickle.dump(eval_metrics, f)
        save_csv(log_dir / "eval_metrics.csv", eval_metrics)

    step = 0
    try:
        while step < args.total_steps:
            chunk = min(args.eval_interval, args.total_steps - step)
            model.learn(total_timesteps=chunk, reset_num_timesteps=False, tb_log_name="sac")
            step += chunk
            metrics = evaluate(model, eval_env, args.eval_episodes, args.seed)
            metrics["step"] = step
            eval_metrics.append(metrics)
            persist()
            model.save(str(log_dir / "model_latest"))

            if metrics["return_"] > best_return:
                best_return = metrics["return_"]
                model.save(str(log_dir / "best_return_model"))
            if metrics["cost_return"] < best_cost:
                best_cost = metrics["cost_return"]
                model.save(str(log_dir / "best_cost_model"))
            safety_score = metrics["cost_return"] + 10.0 * metrics["safety_violation_rate"]
            if safety_score < best_safety:
                best_safety = safety_score
                model.save(str(log_dir / "best_safety_model"))

            if args.save_eval_trajectories:
                rollout_eval_trajectories(model, eval_env, args, step)

            print(f"[Eval] step={step} metrics={metrics}")
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Saving latest outputs...")
    finally:
        model.save(str(log_dir / "model_latest"))
        persist()


if __name__ == "__main__":
    main()
