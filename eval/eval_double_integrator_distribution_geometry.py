import argparse
import csv
import json
from pathlib import Path

import jax
import numpy as np

from envs.safe_obstacle_double_integrator_2d import SafeObstacleDoubleIntegrator2DEnv
from eval.eval_double_integrator_pullback import build_env_kwargs, load_double_integrator_agent, obs_to_algo_obs
from relax.utils.pullback_geometry import (
    empirical_covariance_local,
    local_normal_tangent,
    multi_obstacle_curvature_covariances_double_integrator,
    to_normal_tangent_frame,
)


def sample_policy_actions(agent, obs, num_samples, seed):
    if agent is None:
        return np.zeros((num_samples, 2), dtype=np.float32)
    obs_algo = obs_to_algo_obs(obs)[None, :]
    actions = np.zeros((num_samples, 2), dtype=np.float32)
    key = jax.random.PRNGKey(seed + 11)
    for i in range(num_samples):
        key, k = jax.random.split(key)
        raw = np.asarray(agent.get_action(k, obs_algo)[0], dtype=np.float32)
        actions[i] = np.clip(raw, -1.0, 1.0)
    return actions


def _cov_stats_local(cov_local):
    cov_local = 0.5 * (np.asarray(cov_local, dtype=np.float64) + np.asarray(cov_local, dtype=np.float64).T)
    normal_var = float(cov_local[0, 0])
    tangent_var = float(cov_local[1, 1])
    denom = max(normal_var + tangent_var, 1e-12)
    nlr = normal_var / denom
    tcr = tangent_var / denom
    return normal_var, tangent_var, nlr, tcr


def _row(state_id, obstacle_id, method, pos, vel, clearance, cov_local):
    normal_var, tangent_var, nlr, tcr = _cov_stats_local(cov_local)
    return dict(
        state_id=int(state_id),
        obstacle_id=int(obstacle_id),
        method=method,
        px=float(pos[0]),
        py=float(pos[1]),
        vx=float(vel[0]),
        vy=float(vel[1]),
        clearance=float(clearance),
        normal_var=float(normal_var),
        tangent_var=float(tangent_var),
        nlr=float(nlr),
        tcr=float(tcr),
        trace=float(np.trace(cov_local)),
        det=float(np.linalg.det(cov_local)),
    )


def _make_states_single(env, num_states):
    radius_eval = env.obstacle_radii[0] + env.eps_obs + 0.04
    angles = np.linspace(0.0, 2.0 * np.pi, num_states, endpoint=False)
    states = []
    for angle in angles:
        center = env.obstacle_centers[0]
        pos = center + radius_eval * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        states.append((0, pos))
    return states


def _make_states_multi(env, num_states_per_obstacle):
    states = []
    for obs_id, (center, radius) in enumerate(zip(env.obstacle_centers, env.obstacle_radii)):
        radius_eval = float(radius + env.eps_obs + 0.05)
        angles = np.linspace(0.0, 2.0 * np.pi, num_states_per_obstacle, endpoint=False)
        for angle in angles:
            pos = center + radius_eval * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            states.append((obs_id, pos))
    return states


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vanilla_checkpoint", required=True)
    p.add_argument("--curvature_checkpoint", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--num_states", type=int, default=8)
    p.add_argument("--num_states_per_obstacle", type=int, default=4)
    p.add_argument("--num_action_samples", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start_y_range", type=float, default=None)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--a_max", type=float, default=None)
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
    p.add_argument("--lambda_safe", type=float, default=4.0)
    p.add_argument("--lambda_eps", type=float, default=0.05)
    p.add_argument("--lambda_clip", type=float, default=120.0)
    p.add_argument("--lambda_robust", type=float, default=40.0)
    p.add_argument("--robust_iso", type=float, default=0.25)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    vanilla_agent, vanilla_saved = load_double_integrator_agent(args.vanilla_checkpoint)
    curvature_agent, curvature_saved = load_double_integrator_agent(args.curvature_checkpoint)
    env_kwargs = build_env_kwargs(args, curvature_saved)
    env = SafeObstacleDoubleIntegrator2DEnv(
        noise_sigma=(0.0, 0.0),
        use_filter=False,
        seed=args.seed,
        map_id=env_kwargs["map_id"],
        route_variant=env_kwargs["route_variant"],
        obs_mode=env_kwargs["obs_mode"],
        start_y_range=env_kwargs["start_y_range"],
        dt=env_kwargs["dt"],
        a_max=env_kwargs["a_max"],
        v_max=env_kwargs["v_max"],
        damping=env_kwargs["damping"],
        episode_len=env_kwargs["episode_len"],
        goal_radius=env_kwargs["goal_radius"],
        eps_obs=env_kwargs["eps_obs"],
        reward_mode=env_kwargs["reward_mode"],
        reward_cfg=env_kwargs["reward_cfg"],
    )

    if env.map_id == "three_circles":
        states = _make_states_multi(env, args.num_states_per_obstacle)
    else:
        states = _make_states_single(env, args.num_states)

    rows = []
    sample_rows = []
    vel = np.zeros(2, dtype=np.float32)

    for sid, (anchor_obstacle_id, pos) in enumerate(states):
        state = np.array([pos[0], pos[1], vel[0], vel[1]], dtype=np.float32)
        obs = env._get_obs_from_state(state)
        stats = multi_obstacle_curvature_covariances_double_integrator(
            pos=pos,
            vel=vel,
            obstacle_centers=env.obstacle_centers,
            obstacle_radii=env.obstacle_radii,
            dt=env.dt,
            a_max=env.a_max,
            lambda_scale=args.lambda_safe,
            lambda_eps=args.lambda_eps,
            lambda_clip=args.lambda_clip,
            lambda_robust=args.lambda_robust,
            robust_iso=args.robust_iso,
        )
        nearest_id = int(stats["nearest_obstacle_id"])
        clearance = float(stats["clearances"][nearest_id])
        normal = np.asarray(stats["normal"], dtype=np.float64)
        tangent = np.asarray(stats["tangent"], dtype=np.float64)

        vanilla_actions = sample_policy_actions(vanilla_agent, obs, args.num_action_samples, args.seed + sid)
        curvature_actions = sample_policy_actions(curvature_agent, obs, args.num_action_samples, args.seed + 10000 + sid)
        cov_vanilla_local, *_ = empirical_covariance_local(vanilla_actions, pos=pos, center=env.obstacle_centers[nearest_id])
        cov_curvature_local, *_ = empirical_covariance_local(curvature_actions, pos=pos, center=env.obstacle_centers[nearest_id])

        rows.append(_row(sid, nearest_id, "Nominal", pos, vel, clearance, stats["cov_nominal_local"]))
        rows.append(_row(sid, nearest_id, "Safety-shaped", pos, vel, clearance, stats["cov_safety_local"]))
        rows.append(_row(sid, nearest_id, "Robust-shaped", pos, vel, clearance, stats["cov_robust_local"]))
        rows.append(_row(sid, nearest_id, "Vanilla Flow", pos, vel, clearance, cov_vanilla_local))
        rows.append(_row(sid, nearest_id, "Curvature-Shaped Flow", pos, vel, clearance, cov_curvature_local))

        sample_rows.append(
            dict(
                state_id=int(sid),
                obstacle_id=int(nearest_id),
                anchor_obstacle_id=int(anchor_obstacle_id),
                px=float(pos[0]),
                py=float(pos[1]),
                clearance=float(clearance),
                normal=normal,
                tangent=tangent,
                actions_vanilla=vanilla_actions,
                actions_curvature=curvature_actions,
                actions_vanilla_local=to_normal_tangent_frame(vanilla_actions, normal, tangent),
                actions_curvature_local=to_normal_tangent_frame(curvature_actions, normal, tangent),
                Sigma_nominal=stats["cov_nominal"],
                Sigma_safe=stats["cov_safety"],
                Sigma_robust=stats["cov_robust"],
                Sigma_nominal_local=stats["cov_nominal_local"],
                Sigma_safe_local=stats["cov_safety_local"],
                Sigma_robust_local=stats["cov_robust_local"],
                Sigma_vanilla_local=cov_vanilla_local,
                Sigma_curvature_local=cov_curvature_local,
            )
        )

    fieldnames = list(rows[0].keys())
    with open(outdir / "distribution_geometry.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    np.savez(
        outdir / "distribution_geometry.npz",
        rows=np.array([str(r) for r in rows], dtype=object),
        samples=np.array(sample_rows, dtype=object),
    )
    with open(outdir / "distribution_geometry.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
