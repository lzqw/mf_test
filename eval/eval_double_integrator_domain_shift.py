import argparse
import csv
import json
from pathlib import Path

import numpy as np

from eval.eval_double_integrator_pullback import (
    build_env_kwargs,
    collect_double_integrator_eval_rollouts,
    load_double_integrator_agent,
)


def parse_deltas(raw):
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    out = []
    for p in parts:
        out.append(float(p))
    return out


def per_episode_violation_rates(rollouts):
    state_violation = np.asarray(rollouts["state_violation"], dtype=bool)
    tight_violation = np.asarray(rollouts["tight_violation"], dtype=bool)
    valid_lengths = np.asarray(rollouts["valid_lengths"], dtype=np.int32)
    out = np.zeros(state_violation.shape[0], dtype=np.float32)
    for i in range(out.shape[0]):
        horizon = max(int(valid_lengths[i]) - 1, 1)
        flags = state_violation[i, :horizon] | tight_violation[i, :horizon]
        out[i] = float(np.mean(flags.astype(np.float32)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vanilla_checkpoint", required=True)
    p.add_argument("--curvature_checkpoint", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument(
        "--delta_grid",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 0.3, 0.4],
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start_y_range", type=float, default=0.45)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--a_max", type=float, default=3.0)
    p.add_argument("--v_max", type=float, default=None)
    p.add_argument("--damping", type=float, default=None)
    p.add_argument("--episode_len", type=int, default=None)
    p.add_argument("--eps_obs", type=float, default=None)
    p.add_argument("--map_id", choices=["single_circle", "three_circles"], default="single_circle")
    p.add_argument("--route_variant", type=str, default="baseline")
    p.add_argument("--obs_mode", choices=["single_obstacle", "all_obstacles"], default=None)
    p.add_argument("--reward_mode", choices=["goal_progress", "symmetric_path_progress", "multi_route_progress"], default="goal_progress")
    p.add_argument("--goal_radius", type=float, default=None)
    p.add_argument("--reward_progress_coef", type=float, default=None)
    p.add_argument("--reward_success_bonus", type=float, default=None)
    p.add_argument("--reward_collision_penalty", type=float, default=None)
    p.add_argument("--reward_near_obs_coef", type=float, default=None)
    p.add_argument("--reward_safety_buffer", type=float, default=None)
    p.add_argument("--reward_action_coef", type=float, default=None)
    p.add_argument("--reward_speed_coef", type=float, default=None)
    p.add_argument("--reward_time_coef", type=float, default=None)
    p.add_argument("--reward_route_softmin_beta", type=float, default=None)
    p.add_argument("--reward_route_start_bias_scale", type=float, default=None)
    p.add_argument("--reward_goal_progress_mix", type=float, default=None)
    p.add_argument("--terminal_goal_bonus_radius", type=float, default=None)
    p.add_argument("--terminal_goal_bonus_coef", type=float, default=None)
    p.add_argument("--save_rollouts", action="store_true", default=False)
    args = p.parse_args()

    deltas = parse_deltas(args.delta_grid)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    vanilla, vanilla_saved = load_double_integrator_agent(args.vanilla_checkpoint)
    curvature, curvature_saved = load_double_integrator_agent(args.curvature_checkpoint)
    vanilla_env = build_env_kwargs(args, vanilla_saved)
    curvature_env = build_env_kwargs(args, curvature_saved)

    rows = []

    for delta in deltas:
        for method, agent in [("Vanilla Flow", vanilla), ("Curvature-Shaped Flow", curvature)]:
            alias = "vanilla_flow" if method.startswith("Vanilla") else "curvature_flow"
            rollouts, summary = collect_double_integrator_eval_rollouts(
                agent,
                alias,
                episodes=args.episodes,
                seed=args.seed + int(delta * 1000),
                delta=delta,
                start_y_range=vanilla_env["start_y_range"] if alias == "vanilla_flow" else curvature_env["start_y_range"],
                dt=vanilla_env["dt"] if alias == "vanilla_flow" else curvature_env["dt"],
                a_max=vanilla_env["a_max"] if alias == "vanilla_flow" else curvature_env["a_max"],
                use_handcrafted_controller=False,
                env_kwargs={
                    "v_max": (vanilla_env if alias == "vanilla_flow" else curvature_env)["v_max"],
                    "damping": (vanilla_env if alias == "vanilla_flow" else curvature_env)["damping"],
                    "episode_len": (vanilla_env if alias == "vanilla_flow" else curvature_env)["episode_len"],
                    "goal_radius": (vanilla_env if alias == "vanilla_flow" else curvature_env)["goal_radius"],
                    "eps_obs": (vanilla_env if alias == "vanilla_flow" else curvature_env)["eps_obs"],
                    "map_id": (vanilla_env if alias == "vanilla_flow" else curvature_env)["map_id"],
                    "route_variant": (vanilla_env if alias == "vanilla_flow" else curvature_env)["route_variant"],
                    "obs_mode": (vanilla_env if alias == "vanilla_flow" else curvature_env)["obs_mode"],
                    "reward_mode": (vanilla_env if alias == "vanilla_flow" else curvature_env)["reward_mode"],
                    "reward_cfg": (vanilla_env if alias == "vanilla_flow" else curvature_env)["reward_cfg"],
                },
            )
            row = dict(
                delta=float(delta),
                method=method,
                J_eval_mean=float(summary.get("J_eval_mean", 0.0)),
                J_eval_std=float(summary.get("J_eval_std", 0.0)),
                violation_rate_mean=float(summary.get("violation_rate", 0.0)),
                h_min_mean=float(summary.get("h_min_mean", 0.0)),
                h_min_std=float(summary.get("h_min_std", 0.0)),
                success_rate=float(summary.get("success_rate", 0.0)),
                collision_rate=float(summary.get("collision_rate", 0.0)),
                route_upper_fraction=float(summary.get("route_upper_fraction", 0.0)),
                route_lower_fraction=float(summary.get("route_lower_fraction", 0.0)),
                route_mixed_fraction=float(summary.get("route_mixed_fraction", 0.0)),
                return_mean=float(summary.get("return_mean", 0.0)),
                return_std=float(summary.get("return_std", 0.0)),
            )
            row["violation_rate_std"] = float(np.std(per_episode_violation_rates(rollouts), ddof=0))
            rows.append(row)

            if args.save_rollouts:
                npz_path = outdir / f"rollouts_delta_{delta:.1f}_{alias}.npz"
                np.savez(npz_path, **rollouts)

    csv_path = outdir / "domain_shift_rollout.csv"
    with open(csv_path, "w", newline="") as f:
        keys = [
            "delta",
            "method",
            "J_eval_mean",
            "J_eval_std",
            "violation_rate_mean",
            "violation_rate_std",
            "h_min_mean",
            "h_min_std",
            "success_rate",
            "collision_rate",
            "route_upper_fraction",
            "route_lower_fraction",
            "route_mixed_fraction",
            "return_mean",
            "return_std",
        ]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(outdir / "summary.json", "w") as f:
        json.dump({"rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
