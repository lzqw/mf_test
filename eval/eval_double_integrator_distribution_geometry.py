import argparse
import json
import pickle
from pathlib import Path

import jax
import numpy as np

from envs.safe_obstacle_double_integrator_2d import SafeObstacleDoubleIntegrator2DEnv
from relax.utils.curvature import (
    covariance_from_curvature,
    covariance_variances,
    default_double_integrator,
    default_lqr_config,
    full_robust_curvature,
    nominal_curvature,
    pbar_tau,
    safety_curvature,
    solve_discounted_riccati,
)
from relax.utils.curvature import nlr_tcr
from eval.eval_double_integrator_pullback import (
    load_double_integrator_agent,
    obs_to_algo_obs,
)


def make_state_observation(pos, goal, obstacle_center):
    pos = np.asarray(pos, dtype=np.float32)
    rel_goal = goal - pos
    rel_obs = pos - obstacle_center
    clearance = float(np.linalg.norm(rel_obs) - 0.8)
    d_goal = float(np.linalg.norm(rel_goal))
    obs = np.array(
        [
            float(pos[0]),
            float(pos[1]),
            0.0,
            0.0,
            float(rel_goal[0]),
            float(rel_goal[1]),
            float(rel_obs[0]),
            float(rel_obs[1]),
            clearance,
            d_goal,
        ],
        dtype=np.float32,
    )
    return obs


def sample_policy_actions(agent, obs, num_samples, seed, goal_obs=False):
    if agent is None:
        return np.zeros((num_samples, 2), dtype=np.float32)
    obs_algo = obs_to_algo_obs(obs)
    obs_algo = obs_algo[None, :]
    rng = np.random.default_rng(seed)
    actions = np.zeros((num_samples, 2), dtype=np.float32)
    key = jax.random.PRNGKey(seed + 11)
    for i in range(num_samples):
        key, k = jax.random.split(key)
        raw = np.asarray(agent.get_action(k, obs_algo)[0], dtype=np.float32)
        actions[i] = np.clip(raw, -1.0, 1.0)
    # ensure diversity even if JAX RNG saturates
    if np.linalg.matrix_rank(np.cov(actions.T) if num_samples > 2 else np.eye(2)) == 0:
        actions = actions + 1e-3 * rng.normal(size=actions.shape)
    return np.clip(actions, -1.0, 1.0)


def make_state_grid(r_eval, num_states):
    angles = np.linspace(0.0, 2.0 * np.pi, num_states, endpoint=False)
    states = []
    for a in angles:
        px = r_eval * np.cos(a)
        py = r_eval * np.sin(a)
        if abs(py) < 0.12:
            py *= 1.8
        states.append((float(px), float(py)))
    return states


def estimate_sigma_cov_from_actions(actions):
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError("actions must be [N,2]")
    mean = actions.mean(axis=0, keepdims=True)
    centered = actions - mean
    if centered.shape[0] <= 1:
        return np.eye(2, dtype=np.float64)
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    return 0.5 * (cov + cov.T)


def compute_diagnostic_row(state_id, method, obs, Sigma, D, clearance):
    n, t = covariance_variances(Sigma, D)
    nl, tr = nlr_tcr(Sigma, D)
    row = dict(
        state_id=state_id,
        method=method,
        px=float(obs[0]),
        py=float(obs[1]),
        vx=float(obs[2]),
        vy=float(obs[3]),
        clearance=float(clearance),
        normal_var=float(n),
        tangent_var=float(t),
        nlr=float(nl),
        tcr=float(tr),
        trace=float(np.trace(Sigma)),
        det=float(np.linalg.det(Sigma)),
    )
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vanilla_checkpoint", required=True)
    p.add_argument("--curvature_checkpoint", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--num_states", type=int, default=8)
    p.add_argument("--num_action_samples", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--a_max", type=float, default=3.0)
    p.add_argument("--lambda_safe", type=float, default=100.0)
    p.add_argument("--robust_iso", type=float, default=0.2)
    p.add_argument("--obs_radius", type=float, default=0.8)
    p.add_argument("--eps_obs", type=float, default=0.08)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    goal = np.array([2.6, 0.0], dtype=np.float32)
    env = SafeObstacleDoubleIntegrator2DEnv(
        noise_sigma=(0.0, 0.0),
        use_filter=False,
        dt=args.dt,
        a_max=args.a_max,
    )
    env.goal = goal

    vanilla_agent = load_double_integrator_agent(args.vanilla_checkpoint)
    curvature_agent = load_double_integrator_agent(args.curvature_checkpoint)

    cfg = default_lqr_config()
    cfg["dt"] = args.dt
    A, B = default_double_integrator(dt=cfg["dt"])
    P, K = solve_discounted_riccati(A, B, cfg["Q"], cfg["R"], rho=cfg["rho"])
    rho = float(cfg["rho"])
    M0 = nominal_curvature(P, B, cfg["R"], cfg["rho"])
    Sigma0 = covariance_from_curvature(M0)
    SigmaIso = covariance_from_curvature(M0)

    E = B
    Phi_u = np.eye(2)
    W_P = E.T @ P @ E
    tau = rho * np.max(np.linalg.eigvalsh(E.T @ P @ E)) * 1.1 + 1e-3

    r_eval = args.obs_radius + args.eps_obs + 0.04
    states = make_state_grid(r_eval, args.num_states)

    rows = []
    rows_samples = []
    for sid, (px, py) in enumerate(states):
        obs = make_state_observation(np.array([px, py], dtype=np.float32), goal=goal, obstacle_center=np.array([0.0, 0.0], dtype=np.float32))
        clearance = float(obs[8])
        d = np.array([obs[0], obs[1]], dtype=np.float64)
        if np.linalg.norm(d) < 1e-8:
            d = np.array([1.0, 0.0], dtype=np.float64)
        d_unit = d / np.linalg.norm(d)
        C_active = np.zeros((1, 4), dtype=np.float64)
        C_active[0, :2] = d_unit

        Sigma_nom = Sigma0
        M_nom = M0
        M_safe = M0 + safety_curvature(B, C_active, np.array([[args.lambda_safe]], dtype=np.float64))
        Sigma_safe = covariance_from_curvature(M_safe)

        Lambda_b = np.array([[args.lambda_safe]], dtype=np.float64)
        robust_tau = cfg["rho"] * np.max(np.linalg.eigvalsh(E.T @ P @ E)) * 1.1 + 1e-3
        M_robust, _, _ = full_robust_curvature(
            P,
            B,
            cfg["R"],
            C_active,
            Lambda_b,
            rho,
            E,
            Phi_u,
            W_P,
            robust_tau,
            eps=args.eps_obs,
        )
        Sigma_robust = covariance_from_curvature(M_robust)

        D_mat = B.T @ C_active.T

        rows.append(compute_diagnostic_row(sid, "Nominal LQR", obs, Sigma_nom, D_mat, clearance))
        rows.append(compute_diagnostic_row(sid, "Safety-shaped", obs, Sigma_safe, D_mat, clearance))
        rows.append(compute_diagnostic_row(sid, "Robust-shaped", obs, Sigma_robust, D_mat, clearance))

        act_v = sample_policy_actions(vanilla_agent, obs, args.num_action_samples, args.seed + sid, goal_obs=False)
        act_c = sample_policy_actions(curvature_agent, obs, args.num_action_samples, args.seed + 10_000 + sid, goal_obs=False)

        Sigma_v = estimate_sigma_cov_from_actions(act_v)
        Sigma_c = estimate_sigma_cov_from_actions(act_c)
        rows.append(compute_diagnostic_row(sid, "Vanilla Flow", obs, Sigma_v, D_mat, clearance))
        rows.append(compute_diagnostic_row(sid, "Curvature-Shaped Flow", obs, Sigma_c, D_mat, clearance))

        rows_samples.append({
            "state_id": sid,
            "px": float(obs[0]),
            "py": float(obs[1]),
            "actions_vanilla": act_v,
            "actions_curvature": act_c,
            "Sigma_nominal": Sigma_nom,
            "Sigma_safe": Sigma_safe,
            "Sigma_robust": Sigma_robust,
            "Sigma_vanilla": Sigma_v,
            "Sigma_curvature": Sigma_c,
            "clearance": clearance,
            "D": D_mat,
        })

    header = list(rows[0].keys())
    with open(outdir / "distribution_geometry.csv", "w", newline="") as f:
        import csv

        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

    np.savez(
        outdir / "distribution_geometry.npz",
        rows=np.array([str(r) for r in rows], dtype=object),
        samples=np.array(rows_samples, dtype=object),
        states=np.array(states, dtype=np.float32),
    )
    with open(outdir / "distribution_geometry.json", "w") as f:
        json.dump({
            "num_states": int(args.num_states),
            "num_action_samples": int(args.num_action_samples),
            "rows": rows,
        }, f, indent=2)


if __name__ == "__main__":
    main()
