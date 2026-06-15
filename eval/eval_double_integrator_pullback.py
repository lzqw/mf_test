import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import importlib.metadata as _metadata

try:
    _orig_entry_points = _metadata.entry_points

    def _no_gymnasium_plugins(*args, **kwargs):
        if "group" in kwargs and kwargs["group"] == "gymnasium.envs":
            return []
        return _orig_entry_points(*args, **kwargs)

    _metadata.entry_points = _no_gymnasium_plugins
except Exception:
    pass

import jax
import matplotlib

matplotlib.use("Agg")
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.safe_obstacle_double_integrator_2d import SafeObstacleDoubleIntegrator2DEnv


def make_algo_kwargs(args):
    return dict(
        gamma=args.gamma,
        gamma_p=args.gamma_p,
        lr=args.lr,
        alpha_lr=args.alpha_lr,
        sample_k=args.sample_k,
        lambda_p=args.lambda_p,
        use_projection_critic=args.use_projection_critic,
        fixed_alpha=args.fixed_alpha,
        alpha_value=args.alpha_value,
        lambda_p_warmup_steps=args.lambda_p_warmup_steps,
        lambda_d=args.lambda_d,
        use_frpi_score=args.use_frpi_score,
        tau_c=args.tau_c,
        mu_c=args.mu_c,
        lambda_f=args.lambda_f,
        use_tn_energy=args.use_tn_energy,
        tn_coef=args.tn_coef,
        sigma_n=args.sigma_n,
        sigma_t=args.sigma_t,
        tn_r_min=args.tn_r_min,
        tn_r_max=args.tn_r_max,
        tn_clip=args.tn_clip,
        kappa_tn=args.kappa_tn,
        entropy_reg_mode=args.entropy_reg_mode,
        candidate_temp=args.candidate_temp,
        beta_normal_entropy=args.beta_normal_entropy,
        min_effective_entropy=args.min_effective_entropy,
        target_effective_entropy=args.target_effective_entropy,
        normal_energy_coef=args.normal_energy_coef,
        target_safe_energy=args.target_safe_energy,
        safe_iso_coef=args.safe_iso_coef,
        safe_energy_variant=args.safe_energy_variant,
        weight_mix=args.weight_mix,
    )


def make_algo(args, obs_dim=8, act_dim=2):
    from relax.algorithm.safe_pullback_rf2_sac_ent import SafePullbackRF2SACENT
    from relax.network.safe_pullback_rf2_sac_ent import create_safe_pullback_rf2_sac_ent_net
    key = jax.random.PRNGKey(args.seed)
    hidden_sizes = getattr(args, "hidden_sizes", "256,256,256")
    hidden_sizes = [int(x) for x in str(hidden_sizes).split(",") if str(x).strip()]
    if len(hidden_sizes) == 0:
        hidden_sizes = [256, 256, 256]
    diffusion_hidden_sizes = getattr(args, "diffusion_hidden_sizes", "256,256,256")
    diffusion_hidden_sizes = [int(x) for x in str(diffusion_hidden_sizes).split(",") if str(x).strip()]
    if len(diffusion_hidden_sizes) == 0:
        diffusion_hidden_sizes = [256, 256, 256]
    net, params = create_safe_pullback_rf2_sac_ent_net(
        key,
        obs_dim,
        act_dim,
        hidden_sizes=hidden_sizes,
        diffusion_hidden_sizes=diffusion_hidden_sizes,
        num_timesteps=args.diffusion_steps,
        num_ent_timesteps=args.num_ent_timesteps,
        alpha_value=args.alpha_value,
        fixed_alpha=args.fixed_alpha,
        init_alpha=args.init_alpha,
        noise_scale=args.policy_noise_scale,
    )
    return SafePullbackRF2SACENT(net, params, **make_algo_kwargs(args))


def obs_to_algo_obs(obs_real):
    obs_real = np.asarray(obs_real, dtype=np.float32)
    if obs_real.shape != (10,):
        raise ValueError("Expected 10-dim real double-integrator observation.")

    px, py = float(obs_real[0]), float(obs_real[1])
    goal_rel_x, goal_rel_y = float(obs_real[4]), float(obs_real[5])
    clear = float(obs_real[8])
    d_goal = float(obs_real[9])

    # Mirror x so that algorithm's hard-coded safe-pullback goal remains at -2.6.
    goal_alg_x = float(goal_rel_x)
    goal_alg_y = -goal_rel_y
    rel_obs_alg_x = -px
    rel_obs_alg_y = py
    return np.array(
        [-px, py, goal_alg_x, goal_alg_y, rel_obs_alg_x, rel_obs_alg_y, clear, d_goal],
        dtype=np.float32,
    )


def to_real_action(action_algo):
    # Inverse mirror-x transform for actions.
    a = np.asarray(action_algo, dtype=np.float32).copy()
    a = np.clip(a, -1.0, 1.0)
    a[0] = -a[0]
    return a


def goal_controller(obs_real, goal=np.array([2.6, 0.0], dtype=np.float32)):
    p = obs_real[:2]
    v = obs_real[2:4]
    d = goal - p
    d_norm = float(np.linalg.norm(d) + 1e-6)
    u = d / d_norm
    damp = 0.1
    v_term = -0.12 * v
    raw = np.clip(u + v_term, -1.0, 1.0).astype(np.float32)
    raw = np.clip(raw + np.array([0.0, 0.0], dtype=np.float32), -1.0, 1.0)
    return raw


def _classify_route(positions):
    traj = np.asarray(positions, dtype=np.float32)
    if traj.shape[0] == 0:
        return "unknown"
    near = np.where(np.abs(traj[:, 0]) < 0.5)[0]
    if len(near) == 0:
        near = np.arange(traj.shape[0])
    y_mean = float(np.mean(traj[near, 1]))
    return "upper" if y_mean >= 0.0 else "lower"


def collect_double_integrator_eval_rollouts(
    agent,
    algo,
    episodes=200,
    seed=0,
    delta=0.0,
    start_y_range=0.45,
    dt=0.1,
    a_max=3.0,
    use_handcrafted_controller=False,
    goal=np.array([2.6, 0.0], dtype=np.float32),
):
    env = SafeObstacleDoubleIntegrator2DEnv(
        noise_sigma=(0.0, 0.0),
        use_filter=algo != "vanilla_flow" or agent is not None,
        seed=seed,
        start_y_range=start_y_range,
        dt=dt,
        a_max=a_max,
    )
    env.set_action_gain(1.0 - float(delta))

    t_max = env.episode_len
    positions = np.zeros((episodes, t_max + 1, 2), dtype=np.float32)
    obs_all = np.zeros((episodes, t_max + 1, 10), dtype=np.float32)
    raw_actions = np.zeros((episodes, t_max, 2), dtype=np.float32)
    exec_actions = np.zeros((episodes, t_max, 2), dtype=np.float32)
    rewards = np.zeros((episodes, t_max), dtype=np.float32)
    state_violation = np.zeros((episodes, t_max), dtype=bool)
    tight_violation = np.zeros((episodes, t_max), dtype=bool)
    safe_violation = np.zeros((episodes, t_max), dtype=bool)
    filter_active = np.zeros((episodes, t_max), dtype=bool)
    filter_fallback = np.zeros((episodes, t_max), dtype=bool)
    projection_residual = np.zeros((episodes, t_max), dtype=np.float32)
    distance_to_goal = np.zeros((episodes, t_max), dtype=np.float32)
    distance_to_obstacle = np.zeros((episodes, t_max), dtype=np.float32)
    is_success = np.zeros((episodes,), dtype=bool)
    time_to_goal = np.full((episodes,), t_max, dtype=np.int32)
    valid_lengths = np.full((episodes,), t_max + 1, dtype=np.int32)
    episode_return = np.zeros((episodes,), dtype=np.float32)

    for i in range(episodes):
        obs_real, _ = env.reset(seed=seed + i + 1)
        obs_all[i, 0] = obs_real
        positions[i, 0] = obs_real[:2]
        key = jax.random.PRNGKey(seed + 999 + i)

        for t in range(t_max):
            obs_algo = obs_to_algo_obs(obs_real)
            if use_handcrafted_controller or agent is None:
                raw_algo = goal_controller(obs_real, goal=goal)
            else:
                key, k = jax.random.split(key)
                raw_algo = np.asarray(agent.get_action(k, obs_algo[None, :])[0], dtype=np.float32)
            raw_real = to_real_action(raw_algo)

            obs_next_real, r, term, trunc, info = env.step(raw_real)
            obs_all[i, t + 1] = obs_next_real
            positions[i, t + 1] = obs_next_real[:2]
            raw_actions[i, t] = raw_algo
            exec_actions[i, t] = to_real_action(np.asarray(info["exec_action"], dtype=np.float32))
            rewards[i, t] = float(r)
            state_violation[i, t] = bool(info["state_violation"])
            tight_violation[i, t] = bool(info["tightened_violation"])
            safe_violation[i, t] = bool(info["safe_violation"])
            filter_active[i, t] = bool(info["filter_activated"])
            filter_fallback[i, t] = bool(info["filter_fallback"])
            projection_residual[i, t] = float(np.linalg.norm(raw_algo - exec_actions[i, t]))
            distance_to_goal[i, t] = float(info["distance_to_goal"])
            distance_to_obstacle[i, t] = float(info["clearance"])

            episode_return[i] += float(r)
            if term and not is_success[i]:
                is_success[i] = True
                time_to_goal[i] = t + 1
            if term or trunc:
                valid_lengths[i] = t + 2
                positions[i, t + 1:] = positions[i, t + 1]
                obs_all[i, t + 1:] = obs_all[i, t + 1]
                break
            obs_real = obs_next_real

    # Metrics
    collision = distance_to_obstacle <= 0.0
    h_min = np.array([np.min(distance_to_obstacle[i, : valid_lengths[i] - 1]) for i in range(episodes)], dtype=np.float32)
    violation_mean = np.mean((state_violation | tight_violation), axis=1)

    route_tags = []
    for i in range(episodes):
        if is_success[i]:
            route = _classify_route(positions[i, : time_to_goal[i] + 1])
        else:
            route = "unknown"
        route_tags.append(route)

    success_idx = np.where(is_success)[0]
    upper = int(np.sum([route_tags[i] == "upper" for i in success_idx]))
    lower = int(np.sum([route_tags[i] == "lower" for i in success_idx]))
    total_success = int(np.sum(is_success))
    if total_success > 0:
        route_upper_fraction = upper / max(total_success, 1)
        route_lower_fraction = lower / max(total_success, 1)
    else:
        route_upper_fraction = 0.0
        route_lower_fraction = 0.0

    step_mask = np.arange(t_max)[None, :] < (valid_lengths - 1)[:, None]
    mask_count = max(int(np.sum(step_mask)), 1)

    summary = {
        "return_mean": float(np.mean(episode_return)),
        "return_std": float(np.std(episode_return)),
        "success_rate": float(np.mean(is_success)),
        "collision_rate": float(np.mean(np.any(collision, axis=1))),
        "violation_rate": float(np.sum(violation_mean * 1.0) / episodes),
        "h_min_mean": float(np.mean(h_min)),
        "h_min_std": float(np.std(h_min)),
        "J_eval_mean": float(np.mean(-episode_return)),
        "J_eval_std": float(np.std(-episode_return)),
        "filter_activation_rate": float(np.mean(filter_active[step_mask]) if mask_count > 0 else 0.0),
        "route_upper_fraction": float(route_upper_fraction),
        "route_lower_fraction": float(route_lower_fraction),
        "filter_fallback_rate": float(np.mean(filter_fallback[step_mask]) if mask_count > 0 else 0.0),
        "min_margin": float(np.min(distance_to_obstacle)),
        "max_margin": float(np.max(distance_to_obstacle)),
    }

    rollout_data = dict(
        positions=positions,
        obs=obs_all,
        raw_actions=raw_actions,
        exec_actions=exec_actions,
        rewards=rewards,
        state_violation=state_violation,
        tight_violation=tight_violation,
        safe_violation=safe_violation,
        filter_active=filter_active,
        filter_fallback=filter_fallback,
        projection_residual=projection_residual,
        distance_to_goal=distance_to_goal,
        distance_to_obstacle=distance_to_obstacle,
        is_success=is_success,
        route_tags=np.asarray(route_tags),
        time_to_goal=time_to_goal,
        valid_lengths=valid_lengths,
        episode_return=episode_return,
    )
    return rollout_data, summary


def load_double_integrator_agent(checkpoint):
    with open(checkpoint, "rb") as f:
        ckpt = pickle.load(f)
    saved = ckpt.get("args", {})
    args = argparse.Namespace(**{
        "seed": ckpt.get("seed", 0),
        "gamma": 0.99,
        "gamma_p": 0.99,
        "lr": 3e-4,
        "alpha_lr": 1e-2,
        "sample_k": 64,
        "lambda_p": 0.0,
        "use_projection_critic": False,
        "fixed_alpha": False,
        "alpha_value": 0.1,
        "init_alpha": 0.1,
        "lambda_p_warmup_steps": 100000,
        "lambda_d": 0.5,
        "use_frpi_score": False,
        "tau_c": 1.0,
        "mu_c": 1.0,
        "lambda_f": 2.0,
        "use_tn_energy": True,
        "tn_coef": 1.0,
        "sigma_n": 0.1,
        "sigma_t": 1.0,
        "tn_r_min": 0.02,
        "tn_r_max": 0.20,
        "tn_clip": 10.0,
        "kappa_tn": 1.0,
        "entropy_reg_mode": "legacy",
        "candidate_temp": 0.1,
        "beta_normal_entropy": 1.0,
        "min_effective_entropy": -20.0,
        "target_effective_entropy": 1.0,
        "normal_energy_coef": 0.2,
        "target_safe_energy": 0.08,
        "safe_iso_coef": 0.2,
        "safe_energy_variant": "normal_tangent",
        "weight_mix": 0.05,
        "diffusion_steps": 10,
        "num_ent_timesteps": 10,
        "policy_noise_scale": 0.3,
        "hidden_sizes": "256,256,256",
        "diffusion_hidden_sizes": "256,256,256",
    })
    for k, v in saved.items():
        if hasattr(args, k):
            setattr(args, k, v)
    agent = make_algo(args)
    agent.state = ckpt["agent_state"]
    return agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=False)
    p.add_argument("--algo", choices=["vanilla_flow", "curvature_flow", "handcrafted"], default="curvature_flow")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--delta", type=float, default=0.0)
    p.add_argument("--outdir", required=True)
    p.add_argument("--use_handcrafted_controller", action="store_true", default=False)
    p.add_argument("--start_y_range", type=float, default=0.45)
    p.add_argument("--save_rollouts", action="store_true", default=False)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--a_max", type=float, default=3.0)
    # algorithm arguments used for agent loading
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--alpha_lr", type=float, default=1e-2)
    p.add_argument("--sample_k", type=int, default=64)
    p.add_argument("--lambda_p", type=float, default=0.0)
    p.add_argument("--use_projection_critic", action="store_true", default=False)
    p.add_argument("--fixed_alpha", action="store_true", default=False)
    p.add_argument("--alpha_value", type=float, default=0.1)
    p.add_argument("--init_alpha", type=float, default=0.1)
    p.add_argument("--diffusion_steps", type=int, default=10)
    p.add_argument("--num_ent_timesteps", type=int, default=10)
    p.add_argument("--policy_noise_scale", type=float, default=0.3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gamma_p", type=float, default=0.99)
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
    p.add_argument("--entropy_reg_mode", choices=["legacy", "likelihood_tn", "flac_tn"], default="legacy")
    p.add_argument("--candidate_temp", type=float, default=0.10)
    p.add_argument("--beta_normal_entropy", type=float, default=1.0)
    p.add_argument("--min_effective_entropy", type=float, default=-20.0)
    p.add_argument("--target_effective_entropy", type=float, default=1.0)
    p.add_argument("--normal_energy_coef", type=float, default=0.2)
    p.add_argument("--target_safe_energy", type=float, default=0.08)
    p.add_argument("--safe_iso_coef", type=float, default=0.2)
    p.add_argument("--safe_energy_variant", choices=["normal_iso", "normal_tangent"], default="normal_tangent")
    p.add_argument("--weight_mix", type=float, default=0.05)
    p.add_argument("--hidden_sizes", type=str, default="256,256,256")
    p.add_argument("--diffusion_hidden_sizes", type=str, default="256,256,256")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.use_handcrafted_controller or args.checkpoint is None:
        agent = None
        args.use_tn_energy = False
    else:
        agent = load_double_integrator_agent(args.checkpoint)

    rollouts, summary = collect_double_integrator_eval_rollouts(
        agent,
        args.algo,
        episodes=args.episodes,
        seed=args.seed,
        delta=args.delta,
        start_y_range=args.start_y_range,
        dt=args.dt,
        a_max=args.a_max,
        use_handcrafted_controller=args.use_handcrafted_controller or args.algo == "handcrafted",
    )

    np.savez(outdir / "rollouts.npz", **rollouts)
    (outdir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    with open(outdir / "eval_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
