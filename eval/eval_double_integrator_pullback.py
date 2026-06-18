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

CHECKPOINT_OBS_DIM_MISMATCH_MSG = (
    "checkpoint observation dimension mismatch; "
    "do not mix checkpoints across different env/map observation layouts"
)


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
    if obs_real.shape == (10,):
        px, py = float(obs_real[0]), float(obs_real[1])
        vx, vy = float(obs_real[2]), float(obs_real[3])
        goal_rel_x, goal_rel_y = float(obs_real[4]), float(obs_real[5])
        rel_obs_x, rel_obs_y = float(obs_real[6]), float(obs_real[7])
        clear = float(obs_real[8])
        d_goal = float(obs_real[9])
        return np.array(
            [
                px,
                py,
                goal_rel_x,
                goal_rel_y,
                rel_obs_x,
                rel_obs_y,
                clear,
                d_goal,
                vx,
                vy,
            ],
            dtype=np.float32,
        )
    if obs_real.shape == (16,):
        px, py = float(obs_real[0]), float(obs_real[1])
        vx, vy = float(obs_real[2]), float(obs_real[3])
        goal_rel_x, goal_rel_y = float(obs_real[4]), float(obs_real[5])
        d_goal = float(obs_real[6])
        obstacle_block = obs_real[7:]
        return np.concatenate(
            [
                np.array([px, py, goal_rel_x, goal_rel_y, d_goal, vx, vy], dtype=np.float32),
                obstacle_block.astype(np.float32),
            ],
            axis=0,
        )
    raise ValueError(f"Unsupported double-integrator observation shape: {obs_real.shape}")


def to_real_action(action_algo):
    a = np.asarray(action_algo, dtype=np.float32).copy()
    a = np.clip(a, -1.0, 1.0)
    return a


def goal_controller(
    obs_real,
    goal=np.array([2.6, 0.0], dtype=np.float32),
    map_id="single_circle",
    start_y=0.0,
    upper_route=None,
    lower_route=None,
):
    p = obs_real[:2]
    v = obs_real[2:4]
    if map_id != "three_circles":
        d = goal - p
        d_norm = float(np.linalg.norm(d) + 1e-6)
        u = d / d_norm
        v_term = -0.12 * v
        return np.clip(u + v_term, -1.0, 1.0).astype(np.float32)

    upper_route = [np.asarray(w, dtype=np.float32) for w in (upper_route or [])]
    lower_route = [np.asarray(w, dtype=np.float32) for w in (lower_route or [])]
    if start_y > 0.20:
        route = upper_route
    elif start_y < -0.20:
        route = lower_route
    else:
        route = upper_route if p[1] >= 0.0 else lower_route
    targets = list(route) + [np.asarray(goal, dtype=np.float32)]

    target = targets[-1]
    for wp in targets:
        if np.linalg.norm(wp - p) > 0.30 and p[0] <= wp[0] + 0.25:
            target = wp
            break

    d = target - p
    d_norm = float(np.linalg.norm(d) + 1e-6)
    u = d / d_norm
    v_term = -0.18 * v
    return np.clip(u + v_term, -1.0, 1.0).astype(np.float32)


def _classify_route(positions, map_id="single_circle"):
    traj = np.asarray(positions, dtype=np.float32)
    if traj.shape[0] == 0:
        return "unknown"
    if map_id == "three_circles":
        mask = (traj[:, 0] > -1.5) & (traj[:, 0] < 1.8)
        corridor = traj[mask] if np.any(mask) else traj
        y_mean = float(np.mean(corridor[:, 1]))
        if y_mean > 0.25:
            return "upper"
        if y_mean < -0.25:
            return "lower"
        return "mixed"
    near = np.where(np.abs(traj[:, 0]) < 0.5)[0]
    if len(near) == 0:
        near = np.arange(traj.shape[0])
    y_mean = float(np.mean(traj[near, 1]))
    return "upper" if y_mean >= 0.0 else "lower"


def _start_group(start_y, map_id="single_circle"):
    if map_id == "three_circles":
        if start_y > 0.25:
            return "upper_start"
        if start_y < -0.25:
            return "lower_start"
        return "middle_start"
    return "all"


def collect_double_integrator_eval_rollouts(
    agent,
    algo,
    episodes=200,
    seed=0,
    delta=0.0,
    start_y_range=0.45,
    dt=0.1,
    a_max=3.0,
    noise_pos=0.0,
    noise_vel=0.0,
    use_handcrafted_controller=False,
    goal=np.array([2.6, 0.0], dtype=np.float32),
    env_kwargs=None,
):
    env_kwargs = dict(env_kwargs or {})
    env = SafeObstacleDoubleIntegrator2DEnv(
        noise_sigma=(noise_pos, noise_vel),
        use_filter=algo != "vanilla_flow" or agent is not None,
        seed=seed,
        start_y_range=start_y_range,
        dt=dt,
        a_max=a_max,
        **env_kwargs,
    )
    env.set_action_gain(1.0 - float(delta))

    t_max = env.episode_len
    obs_dim = int(env.observation_space.shape[0])
    positions = np.zeros((episodes, t_max + 1, 2), dtype=np.float32)
    obs_all = np.zeros((episodes, t_max + 1, obs_dim), dtype=np.float32)
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
    start_ys = np.zeros((episodes,), dtype=np.float32)

    for i in range(episodes):
        obs_real, reset_info = env.reset(seed=seed + i + 1)
        start_ys[i] = float(reset_info.get("start_y", obs_real[1]))
        obs_all[i, 0] = obs_real
        positions[i, 0] = obs_real[:2]
        key = jax.random.PRNGKey(seed + 999 + i)

        for t in range(t_max):
            obs_algo = obs_to_algo_obs(obs_real)
            if use_handcrafted_controller or agent is None:
                raw_algo = goal_controller(
                    obs_real,
                    goal=goal,
                    map_id=env.map_id,
                    start_y=float(start_ys[i]),
                    upper_route=getattr(env, "upper_route", None),
                    lower_route=getattr(env, "lower_route", None),
                )
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
            if bool(info.get("success", False)) and not is_success[i]:
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
    final_distance = np.array(
        [distance_to_goal[i, max(valid_lengths[i] - 2, 0)] for i in range(episodes)],
        dtype=np.float32,
    )
    violation_mean = np.array(
        [
            np.mean((state_violation[i, : valid_lengths[i] - 1] | tight_violation[i, : valid_lengths[i] - 1]).astype(np.float32))
            for i in range(episodes)
        ],
        dtype=np.float32,
    )
    collision_episode = np.array(
        [np.any(collision[i, : valid_lengths[i] - 1]) for i in range(episodes)],
        dtype=bool,
    )

    route_tags = []
    for i in range(episodes):
        route = _classify_route(positions[i, : valid_lengths[i]], map_id=env.map_id)
        route_tags.append(route)

    route_upper_fraction = float(np.mean([tag == "upper" for tag in route_tags]))
    route_lower_fraction = float(np.mean([tag == "lower" for tag in route_tags]))
    route_mixed_fraction = float(np.mean([tag == "mixed" for tag in route_tags]))

    step_mask = np.arange(t_max)[None, :] < (valid_lengths - 1)[:, None]
    mask_count = max(int(np.sum(step_mask)), 1)

    summary = {
        "return_mean": float(np.mean(episode_return)),
        "return_std": float(np.std(episode_return)),
        "success_rate": float(np.mean(is_success)),
        "collision_rate": float(np.mean(collision_episode)),
        "violation_rate": float(np.sum(violation_mean * 1.0) / episodes),
        "h_min_mean": float(np.mean(h_min)),
        "h_min_std": float(np.std(h_min)),
        "J_eval_mean": float(np.mean(-episode_return)),
        "J_eval_std": float(np.std(-episode_return)),
        "filter_activation_rate": float(np.mean(filter_active[step_mask]) if mask_count > 0 else 0.0),
        "route_upper_fraction": float(route_upper_fraction),
        "route_lower_fraction": float(route_lower_fraction),
        "route_mixed_fraction": float(route_mixed_fraction),
        "filter_fallback_rate": float(np.mean(filter_fallback[step_mask]) if mask_count > 0 else 0.0),
        "min_margin": float(np.min(distance_to_obstacle[step_mask])) if mask_count > 0 else 0.0,
        "max_margin": float(np.max(distance_to_obstacle[step_mask])) if mask_count > 0 else 0.0,
        "final_distance_mean": float(np.mean(final_distance)),
        "final_distance_std": float(np.std(final_distance)),
        "final_distance_q25": float(np.quantile(final_distance, 0.25)),
        "final_distance_q50": float(np.quantile(final_distance, 0.50)),
        "final_distance_q75": float(np.quantile(final_distance, 0.75)),
        "final_distance_q90": float(np.quantile(final_distance, 0.90)),
    }

    if env.map_id == "three_circles":
        for group in ["upper_start", "middle_start", "lower_start"]:
            mask = np.array([_start_group(y, env.map_id) == group for y in start_ys], dtype=bool)
            if np.any(mask):
                summary[f"{group}_upper_route_fraction"] = float(np.mean([route_tags[i] == "upper" for i in np.where(mask)[0]]))
                summary[f"{group}_lower_route_fraction"] = float(np.mean([route_tags[i] == "lower" for i in np.where(mask)[0]]))
                summary[f"{group}_mixed_route_fraction"] = float(np.mean([route_tags[i] == "mixed" for i in np.where(mask)[0]]))
                summary[f"{group}_success_rate"] = float(np.mean(is_success[mask]))
                summary[f"{group}_collision_rate"] = float(np.mean(collision_episode[mask]))
                summary[f"{group}_h_min_mean"] = float(np.mean(h_min[mask]))
                summary[f"{group}_J_eval_mean"] = float(np.mean(-episode_return[mask]))
            else:
                summary[f"{group}_upper_route_fraction"] = 0.0
                summary[f"{group}_lower_route_fraction"] = 0.0
                summary[f"{group}_mixed_route_fraction"] = 0.0
                summary[f"{group}_success_rate"] = 0.0
                summary[f"{group}_collision_rate"] = 0.0
                summary[f"{group}_h_min_mean"] = 0.0
                summary[f"{group}_J_eval_mean"] = 0.0

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
        start_y=start_ys,
        time_to_goal=time_to_goal,
        valid_lengths=valid_lengths,
        episode_return=episode_return,
    )
    return rollout_data, summary


def load_double_integrator_agent(checkpoint):
    try:
        with open(checkpoint, "rb") as f:
            ckpt = pickle.load(f)
    except Exception:
        cpu_device = jax.devices("cpu")[0]
        with open(checkpoint, "rb") as f:
            with jax.default_device(cpu_device):
                ckpt = pickle.load(f)
    saved = ckpt.get("args", {})
    saved_obs_dim = saved.get("obs_dim", ckpt.get("obs_dim", None))
    if saved_obs_dim is None or int(saved_obs_dim) <= 0:
        raise ValueError(CHECKPOINT_OBS_DIM_MISMATCH_MSG)
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
        "obs_dim": int(saved_obs_dim),
    })
    for k, v in saved.items():
        if hasattr(args, k):
            setattr(args, k, v)
    args.obs_dim = int(saved.get("obs_dim", getattr(args, "obs_dim", 10)))
    agent = make_algo(args, obs_dim=args.obs_dim)
    try:
        agent.state = ckpt["agent_state"]
    except Exception as exc:
        raise ValueError(CHECKPOINT_OBS_DIM_MISMATCH_MSG) from exc
    return agent, saved


def _float_arg_or_saved(cli_value, saved_args, key, default):
    return float(saved_args.get(key, default) if cli_value is None else cli_value)


def _int_arg_or_saved(cli_value, saved_args, key, default):
    return int(saved_args.get(key, default) if cli_value is None else cli_value)


def build_env_kwargs(args, saved_args):
    args_get = lambda key, default=None: getattr(args, key, default)
    reward_cfg = dict(
        progress_coef=_float_arg_or_saved(args_get("reward_progress_coef"), saved_args, "reward_progress_coef", 8.0),
        success_bonus=_float_arg_or_saved(args_get("reward_success_bonus"), saved_args, "reward_success_bonus", 100.0),
        collision_penalty=_float_arg_or_saved(
            args_get("reward_collision_penalty"), saved_args, "reward_collision_penalty", 100.0
        ),
        near_obs_coef=_float_arg_or_saved(args_get("reward_near_obs_coef"), saved_args, "reward_near_obs_coef", 8.0),
        safety_buffer=_float_arg_or_saved(args_get("reward_safety_buffer"), saved_args, "reward_safety_buffer", 0.20),
        action_coef=_float_arg_or_saved(args_get("reward_action_coef"), saved_args, "reward_action_coef", 0.03),
        speed_coef=_float_arg_or_saved(args_get("reward_speed_coef"), saved_args, "reward_speed_coef", 0.01),
        time_coef=_float_arg_or_saved(args_get("reward_time_coef"), saved_args, "reward_time_coef", 0.01),
        route_softmin_beta=_float_arg_or_saved(
            args_get("reward_route_softmin_beta"), saved_args, "reward_route_softmin_beta", 0.0
        ),
        route_start_bias_scale=_float_arg_or_saved(
            args_get("reward_route_start_bias_scale"), saved_args, "reward_route_start_bias_scale", 0.0
        ),
        goal_progress_mix=_float_arg_or_saved(
            args_get("reward_goal_progress_mix"), saved_args, "reward_goal_progress_mix", 0.0
        ),
        terminal_goal_bonus_radius=_float_arg_or_saved(
            args_get("terminal_goal_bonus_radius"), saved_args, "terminal_goal_bonus_radius", 0.0
        ),
        terminal_goal_bonus_coef=_float_arg_or_saved(
            args_get("terminal_goal_bonus_coef"), saved_args, "terminal_goal_bonus_coef", 0.0
        ),
    )
    return dict(
        map_id=str(saved_args.get("map_id", args_get("map_id", "single_circle"))),
        route_variant=str(saved_args.get("route_variant", args_get("route_variant", "baseline"))),
        obs_mode=saved_args.get("obs_mode", args_get("obs_mode", None)),
        start_y_range=_float_arg_or_saved(args_get("start_y_range"), saved_args, "start_y_range", 0.45),
        dt=_float_arg_or_saved(args_get("dt"), saved_args, "dt", 0.1),
        a_max=_float_arg_or_saved(args_get("a_max"), saved_args, "a_max", 3.0),
        v_max=_float_arg_or_saved(args_get("v_max"), saved_args, "v_max", 2.0),
        damping=_float_arg_or_saved(args_get("damping"), saved_args, "damping", 0.98),
        episode_len=_int_arg_or_saved(args_get("episode_len"), saved_args, "episode_len", 200),
        goal_radius=_float_arg_or_saved(args_get("goal_radius"), saved_args, "goal_radius", 0.18),
        goal_radius_overridden=bool(args_get("goal_radius") is not None),
        eps_obs=_float_arg_or_saved(args_get("eps_obs"), saved_args, "eps_obs", 0.08),
        reward_mode=str(saved_args.get("reward_mode", args_get("reward_mode", "goal_progress"))),
        reward_cfg=reward_cfg,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=False)
    p.add_argument("--algo", choices=["vanilla_flow", "curvature_flow", "handcrafted"], default="curvature_flow")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--delta", type=float, default=0.0)
    p.add_argument("--outdir", required=True)
    p.add_argument("--use_handcrafted_controller", action="store_true", default=False)
    p.add_argument("--start_y_range", type=float, default=None)
    p.add_argument("--save_rollouts", action="store_true", default=False)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--a_max", type=float, default=None)
    p.add_argument("--v_max", type=float, default=None)
    p.add_argument("--damping", type=float, default=None)
    p.add_argument("--episode_len", type=int, default=None)
    p.add_argument("--goal_radius", type=float, default=None)
    p.add_argument("--eps_obs", type=float, default=None)
    p.add_argument("--map_id", choices=["single_circle", "three_circles"], default="single_circle")
    p.add_argument("--route_variant", type=str, default="baseline")
    p.add_argument("--obs_mode", choices=["single_obstacle", "all_obstacles"], default=None)
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
    p.add_argument("--reward_mode", choices=["goal_progress", "symmetric_path_progress", "multi_route_progress"], default="goal_progress")
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
    p.add_argument("--obs_dim", type=int, default=8)
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
        saved_args = {}
        args.use_tn_energy = False
    else:
        agent, saved_args = load_double_integrator_agent(args.checkpoint)

    env_kwargs = build_env_kwargs(args, saved_args)

    rollouts, summary = collect_double_integrator_eval_rollouts(
        agent,
        args.algo,
        episodes=args.episodes,
        seed=args.seed,
        delta=args.delta,
        start_y_range=env_kwargs["start_y_range"],
        dt=env_kwargs["dt"],
        a_max=env_kwargs["a_max"],
        noise_pos=0.0,
        noise_vel=0.0,
        use_handcrafted_controller=args.use_handcrafted_controller or args.algo == "handcrafted",
        env_kwargs={
            "v_max": env_kwargs["v_max"],
            "damping": env_kwargs["damping"],
            "episode_len": env_kwargs["episode_len"],
            "goal_radius": env_kwargs["goal_radius"],
            "eps_obs": env_kwargs["eps_obs"],
            "map_id": env_kwargs["map_id"],
            "route_variant": env_kwargs["route_variant"],
            "obs_mode": env_kwargs["obs_mode"],
            "reward_mode": env_kwargs["reward_mode"],
            "reward_cfg": env_kwargs["reward_cfg"],
        },
    )

    summary["goal_radius"] = float(env_kwargs["goal_radius"])
    summary["goal_radius_overridden"] = bool(env_kwargs["goal_radius_overridden"])
    summary["reward_goal_progress_mix"] = float(env_kwargs["reward_cfg"].get("goal_progress_mix", 0.0))
    summary["reward_near_obs_coef"] = float(env_kwargs["reward_cfg"].get("near_obs_coef", 0.0))
    summary["terminal_goal_bonus_radius"] = float(env_kwargs["reward_cfg"].get("terminal_goal_bonus_radius", 0.0))
    summary["terminal_goal_bonus_coef"] = float(env_kwargs["reward_cfg"].get("terminal_goal_bonus_coef", 0.0))

    np.savez(outdir / "rollouts.npz", **rollouts)
    (outdir / "env_config.json").write_text(json.dumps(env_kwargs, indent=2))
    (outdir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    with open(outdir / "eval_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
