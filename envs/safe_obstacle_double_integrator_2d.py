import importlib.metadata as _metadata

try:
    _orig_entry_points = _metadata.entry_points

    def _no_gymnasium_plugins(*args, **kwargs):
        if kwargs.get("group") == "gymnasium.envs":
            return []
        return _orig_entry_points(*args, **kwargs)

    _metadata.entry_points = _no_gymnasium_plugins
except Exception:
    pass

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from relax.safety.obstacle_double_integrator_filter import (
    DoubleIntegratorObstacleConfig,
    DoubleIntegratorObstacleFilter,
)


def _default_map_spec(map_id):
    if map_id == "single_circle":
        return dict(
            start_center=np.array([-2.6, 0.0], dtype=np.float32),
            goal=np.array([2.6, 0.0], dtype=np.float32),
            start_y_range=0.45,
            goal_radius=0.18,
            obstacles=[
                {"center": np.array([0.0, 0.0], dtype=np.float32), "radius": 0.8},
            ],
            eps_obs=0.08,
            episode_len=200,
            a_max=3.0,
            v_max=2.0,
            dt=0.1,
            damping=0.98,
            workspace_limits=dict(x_min=-3.5, x_max=3.5, y_min=-2.0, y_max=2.0),
        )
    if map_id == "three_circles":
        return dict(
            start_center=np.array([-3.0, 0.0], dtype=np.float32),
            goal=np.array([3.0, 0.0], dtype=np.float32),
            start_y_range=0.90,
            goal_radius=0.25,
            obstacles=[
                {"center": np.array([-0.70, 0.00], dtype=np.float32), "radius": 0.45},
                {"center": np.array([0.70, 0.80], dtype=np.float32), "radius": 0.42},
                {"center": np.array([0.70, -0.80], dtype=np.float32), "radius": 0.42},
            ],
            eps_obs=0.06,
            episode_len=250,
            a_max=3.5,
            v_max=2.5,
            dt=0.08,
            damping=0.98,
            workspace_limits=dict(x_min=-3.8, x_max=3.8, y_min=-2.4, y_max=2.4),
        )
    raise ValueError(f"Unknown map_id: {map_id}")


def _default_reward_cfg(map_id):
    if map_id == "three_circles":
        return dict(
            progress_coef=12.0,
            success_bonus=150.0,
            collision_penalty=120.0,
            near_obs_coef=6.0,
            safety_buffer=0.18,
            action_coef=0.015,
            speed_coef=0.005,
            time_coef=0.02,
            route_softmin_beta=0.0,
            route_start_bias_scale=0.0,
            goal_progress_mix=0.0,
            terminal_goal_bonus_radius=0.0,
            terminal_goal_bonus_coef=0.0,
        )
    return dict(
        progress_coef=8.0,
        success_bonus=100.0,
        collision_penalty=100.0,
        near_obs_coef=8.0,
        safety_buffer=0.20,
        action_coef=0.03,
        speed_coef=0.01,
        time_coef=0.01,
        route_softmin_beta=0.0,
        route_start_bias_scale=0.0,
        goal_progress_mix=0.0,
        terminal_goal_bonus_radius=0.0,
        terminal_goal_bonus_coef=0.0,
    )


def _three_circle_routes(route_variant):
    route_variant = str(route_variant)
    if route_variant == "baseline":
        return (
            [
                np.array([-1.60, 0.90], dtype=np.float32),
                np.array([-0.20, 1.35], dtype=np.float32),
                np.array([1.30, 1.05], dtype=np.float32),
            ],
            [
                np.array([-1.60, -0.90], dtype=np.float32),
                np.array([-0.20, -1.35], dtype=np.float32),
                np.array([1.30, -1.05], dtype=np.float32),
            ],
        )
    if route_variant == "exit_pull_v1":
        return (
            [
                np.array([-1.60, 0.90], dtype=np.float32),
                np.array([-0.20, 1.35], dtype=np.float32),
                np.array([1.20, 1.15], dtype=np.float32),
                np.array([2.10, 0.42], dtype=np.float32),
            ],
            [
                np.array([-1.60, -0.90], dtype=np.float32),
                np.array([-0.20, -1.35], dtype=np.float32),
                np.array([1.20, -1.15], dtype=np.float32),
                np.array([2.10, -0.42], dtype=np.float32),
            ],
        )
    if route_variant == "exit_pull_v2":
        return (
            [
                np.array([-1.60, 0.90], dtype=np.float32),
                np.array([-0.20, 1.35], dtype=np.float32),
                np.array([1.35, 1.20], dtype=np.float32),
                np.array([2.20, 0.25], dtype=np.float32),
            ],
            [
                np.array([-1.60, -0.90], dtype=np.float32),
                np.array([-0.20, -1.35], dtype=np.float32),
                np.array([1.35, -1.20], dtype=np.float32),
                np.array([2.20, -0.25], dtype=np.float32),
            ],
        )
    if route_variant == "exit_pull_v3":
        return (
            [
                np.array([-1.60, 0.90], dtype=np.float32),
                np.array([-0.20, 1.35], dtype=np.float32),
                np.array([1.35, 1.10], dtype=np.float32),
                np.array([2.15, 0.24], dtype=np.float32),
                np.array([2.65, 0.08], dtype=np.float32),
            ],
            [
                np.array([-1.60, -0.90], dtype=np.float32),
                np.array([-0.20, -1.35], dtype=np.float32),
                np.array([1.35, -1.10], dtype=np.float32),
                np.array([2.15, -0.24], dtype=np.float32),
                np.array([2.65, -0.08], dtype=np.float32),
            ],
        )
    if route_variant == "exit_pull_v4":
        return (
            [
                np.array([-1.60, 0.90], dtype=np.float32),
                np.array([-0.20, 1.35], dtype=np.float32),
                np.array([1.35, 1.15], dtype=np.float32),
                np.array([2.15, 0.12], dtype=np.float32),
            ],
            [
                np.array([-1.60, -0.90], dtype=np.float32),
                np.array([-0.20, -1.35], dtype=np.float32),
                np.array([1.35, -1.15], dtype=np.float32),
                np.array([2.15, -0.12], dtype=np.float32),
            ],
        )
    raise ValueError(f"Unknown route_variant: {route_variant}")


def _normalize_obstacles(obstacles):
    out = []
    for obs in obstacles:
        out.append(
            dict(
                center=np.asarray(obs["center"], dtype=np.float32).reshape(2),
                radius=float(obs["radius"]),
            )
        )
    return out


class SafeObstacleDoubleIntegrator2DEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        noise_sigma=(0.01, 0.01),
        use_filter=True,
        seed=0,
        start_y_range=None,
        dt=None,
        a_max=None,
        v_max=None,
        damping=None,
        action_gain=1.0,
        episode_len=None,
        goal=None,
        start_center=None,
        goal_radius=None,
        obstacle_center=None,
        obstacle_radius=None,
        obstacles=None,
        eps_obs=None,
        reward_mode="goal_progress",
        reward_cfg=None,
        map_id="single_circle",
        obs_mode=None,
        route_variant="baseline",
    ):
        self.map_id = str(map_id)
        spec = _default_map_spec(self.map_id)
        reward_cfg_input = reward_cfg
        reward_defaults = _default_reward_cfg(self.map_id)
        reward_cfg = dict(reward_defaults)
        if reward_cfg_input is not None:
            reward_cfg.update(dict(reward_cfg_input))

        if obstacles is None:
            if obstacle_center is not None or obstacle_radius is not None:
                center = np.array([0.0, 0.0], dtype=np.float32) if obstacle_center is None else np.asarray(obstacle_center, dtype=np.float32)
                radius = spec["obstacles"][0]["radius"] if obstacle_radius is None else float(obstacle_radius)
                obstacles = [dict(center=center, radius=radius)]
            else:
                obstacles = spec["obstacles"]
        self.obstacles = _normalize_obstacles(obstacles)
        self.obstacle_centers = np.stack([o["center"] for o in self.obstacles], axis=0).astype(np.float32)
        self.obstacle_radii = np.asarray([o["radius"] for o in self.obstacles], dtype=np.float32)

        self.dt = float(spec["dt"] if dt is None else dt)
        self.a_max = float(spec["a_max"] if a_max is None else a_max)
        self.v_max = float(spec["v_max"] if v_max is None else v_max)
        self.damping = float(spec["damping"] if damping is None else damping)
        self.action_gain = float(action_gain)
        self.episode_len = int(spec["episode_len"] if episode_len is None else episode_len)
        self.goal = np.asarray(spec["goal"] if goal is None else goal, dtype=np.float32)
        self.goal_radius = float(spec["goal_radius"] if goal_radius is None else goal_radius)
        self.start_center = np.asarray(spec["start_center"] if start_center is None else start_center, dtype=np.float32)
        self.eps_obs = float(spec["eps_obs"] if eps_obs is None else eps_obs)
        self.reward_mode = str(reward_mode)
        self.route_variant = str(route_variant)
        self.noise_sigma = np.asarray(noise_sigma, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.start_y_range = float(spec["start_y_range"] if start_y_range is None else start_y_range)
        self.use_filter = bool(use_filter)
        self.obs_mode = str(obs_mode or ("single_obstacle" if self.map_id == "single_circle" else "all_obstacles"))
        if self.obs_mode not in {"single_obstacle", "all_obstacles"}:
            raise ValueError(f"Unsupported obs_mode: {self.obs_mode}")

        obs_dim = 10 if self.obs_mode == "single_obstacle" else int(7 + 3 * len(self.obstacles))
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._workspace_limits = dict(spec["workspace_limits"])
        self.filter_cfg = DoubleIntegratorObstacleConfig(
            dt=self.dt,
            a_max=self.a_max,
            obstacle_centers=self.obstacle_centers,
            obstacle_radii=self.obstacle_radii,
            eps_obs=self.eps_obs,
            x_min=self._workspace_limits["x_min"],
            x_max=self._workspace_limits["x_max"],
            y_min=self._workspace_limits["y_min"],
            y_max=self._workspace_limits["y_max"],
        )
        self.filter = DoubleIntegratorObstacleFilter(self.filter_cfg)
        self.state = np.zeros(4, dtype=np.float32)
        self.prev_exec_action = np.zeros(2, dtype=np.float32)
        self.t = 0
        self.start_y = 0.0

        self.reward_cfg = reward_cfg
        self.obstacle_center = self.obstacle_centers[0].copy()
        self.obstacle_radius = float(self.obstacle_radii[0])

        self.waypoint_up = self.obstacle_center + np.array([0.0, self.obstacle_radius + 0.55], dtype=np.float32)
        self.waypoint_down = self.obstacle_center + np.array([0.0, -(self.obstacle_radius + 0.55)], dtype=np.float32)
        if self.map_id == "three_circles":
            self.upper_route, self.lower_route = _three_circle_routes(self.route_variant)
        else:
            self.upper_route = [
                np.array([-1.60, 0.90], dtype=np.float32),
                np.array([-0.20, 1.35], dtype=np.float32),
                np.array([1.30, 1.05], dtype=np.float32),
            ]
            self.lower_route = [
                np.array([-1.60, -0.90], dtype=np.float32),
                np.array([-0.20, -1.35], dtype=np.float32),
                np.array([1.30, -1.05], dtype=np.float32),
            ]

    def _state_in_workspace(self, pos):
        x, y = float(pos[0]), float(pos[1])
        return (
            self._workspace_limits["x_min"] <= x <= self._workspace_limits["x_max"]
            and self._workspace_limits["y_min"] <= y <= self._workspace_limits["y_max"]
        )

    def _route_polyline(self, waypoints):
        return [self.start_center.astype(np.float32)] + [np.asarray(wp, dtype=np.float32) for wp in waypoints] + [self.goal.astype(np.float32)]

    def _route_potential(self, pos, waypoints):
        pos = np.asarray(pos, dtype=np.float32)
        polyline = self._route_polyline(waypoints)
        seg_lengths = [float(np.linalg.norm(b - a)) for a, b in zip(polyline[:-1], polyline[1:])]
        remaining_after = np.cumsum(seg_lengths[::-1])[::-1]
        best = np.inf
        for i, (a, b) in enumerate(zip(polyline[:-1], polyline[1:])):
            ab = b - a
            denom = float(np.dot(ab, ab))
            if denom <= 1e-8:
                proj = a
                t = 0.0
            else:
                t = float(np.clip(np.dot(pos - a, ab) / denom, 0.0, 1.0))
                proj = a + t * ab
            perp = float(np.linalg.norm(pos - proj))
            rem = (1.0 - t) * seg_lengths[i]
            if i + 1 < len(seg_lengths):
                rem += float(np.sum(seg_lengths[i + 1:]))
            cand = perp + rem
            if cand < best:
                best = cand
        return float(best)

    def _path_potential(self, pos):
        pos = np.asarray(pos, dtype=np.float32)
        d_up = float(np.linalg.norm(pos - self.waypoint_up) + np.linalg.norm(self.waypoint_up - self.goal))
        d_down = float(np.linalg.norm(pos - self.waypoint_down) + np.linalg.norm(self.waypoint_down - self.goal))
        return min(d_up, d_down)

    def _multi_route_potential(self, pos):
        d_upper = self._route_potential(pos, self.upper_route)
        d_lower = self._route_potential(pos, self.lower_route)
        bias_scale = float(self.reward_cfg.get("route_start_bias_scale", 0.0))
        if bias_scale > 0.0:
            y_scale = max(self.start_y_range, 1e-6)
            start_bias = float(np.tanh(self.start_y / y_scale))
            d_upper -= bias_scale * start_bias
            d_lower += bias_scale * start_bias
        beta = float(self.reward_cfg.get("route_softmin_beta", 0.0))
        if beta <= 0.0:
            return min(d_upper, d_lower)
        vals = np.asarray([-beta * d_upper, -beta * d_lower], dtype=np.float64)
        vmax = float(np.max(vals))
        return float(-(vmax + np.log(np.sum(np.exp(vals - vmax)))) / beta)

    def _per_obstacle_clearances(self, pos):
        pos = np.asarray(pos, dtype=np.float32)
        return np.linalg.norm(pos[None, :] - self.obstacle_centers, axis=1) - self.obstacle_radii

    def _per_obstacle_tight_clearances(self, pos):
        pos = np.asarray(pos, dtype=np.float32)
        return np.linalg.norm(pos[None, :] - self.obstacle_centers, axis=1) - (self.obstacle_radii + self.eps_obs)

    def _nearest_obstacle_id(self, pos):
        clearances = self._per_obstacle_clearances(pos)
        return int(np.argmin(clearances))

    def _get_obs_from_state(self, state):
        pos = state[:2].astype(np.float32)
        vel = state[2:].astype(np.float32)
        rel_goal = self.goal - pos
        d_goal = float(np.linalg.norm(rel_goal))
        clearances = self._per_obstacle_clearances(pos)
        nearest_id = int(np.argmin(clearances))

        if self.obs_mode == "single_obstacle":
            rel_obs = pos - self.obstacle_centers[nearest_id]
            clearance = float(clearances[nearest_id])
            return np.array(
                [
                    float(pos[0]),
                    float(pos[1]),
                    float(vel[0]),
                    float(vel[1]),
                    float(rel_goal[0]),
                    float(rel_goal[1]),
                    float(rel_obs[0]),
                    float(rel_obs[1]),
                    clearance,
                    d_goal,
                ],
                dtype=np.float32,
            )

        obs = [
            float(pos[0]),
            float(pos[1]),
            float(vel[0]),
            float(vel[1]),
            float(rel_goal[0]),
            float(rel_goal[1]),
            d_goal,
        ]
        for center, clearance in zip(self.obstacle_centers, clearances):
            rel = center - pos
            obs.extend([float(rel[0]), float(rel[1]), float(clearance)])
        return np.asarray(obs, dtype=np.float32)

    def _compute_reward(
        self,
        state,
        next_state,
        raw_action,
        exec_action,
        success,
        state_violation,
        dist_goal_prev,
    ):
        cfg = self.reward_cfg
        dist_next = float(np.linalg.norm(next_state[:2] - self.goal))
        if self.reward_mode == "goal_progress":
            progress_delta = dist_goal_prev - dist_next
        elif self.reward_mode == "symmetric_path_progress":
            progress_delta = self._path_potential(state[:2]) - self._path_potential(next_state[:2])
        elif self.reward_mode == "multi_route_progress":
            route_delta = self._multi_route_potential(state[:2]) - self._multi_route_potential(next_state[:2])
            goal_delta = dist_goal_prev - dist_next
            goal_mix = float(np.clip(self.reward_cfg.get("goal_progress_mix", 0.0), 0.0, 1.0))
            progress_delta = (1.0 - goal_mix) * route_delta + goal_mix * goal_delta
        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

        progress_reward = cfg["progress_coef"] * progress_delta
        terminal_radius = float(cfg.get("terminal_goal_bonus_radius", 0.0))
        terminal_coef = float(cfg.get("terminal_goal_bonus_coef", 0.0))
        if terminal_radius > 0.0 and terminal_coef != 0.0 and dist_next < terminal_radius:
            progress_reward += terminal_coef * (dist_goal_prev - dist_next)
        action_penalty = cfg["action_coef"] * float(np.sum(np.square(exec_action)))
        speed_penalty = cfg["speed_coef"] * float(np.sum(np.square(next_state[2:])))
        base_reward = progress_reward - action_penalty - speed_penalty - cfg["time_coef"]

        clearance = float(np.min(self._per_obstacle_clearances(next_state[:2])))
        near_obs_penalty = 0.0
        if clearance < cfg["safety_buffer"]:
            near_obs_penalty = cfg["near_obs_coef"] * (cfg["safety_buffer"] - clearance) ** 2

        reward = float(base_reward - near_obs_penalty)
        if success:
            reward += cfg["success_bonus"]
        if state_violation:
            reward -= cfg["collision_penalty"]
        return reward, dist_next

    def set_action_gain(self, action_gain):
        self.action_gain = float(action_gain)
        self.filter.set_action_gain(self.action_gain)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        y0 = float(self.rng.uniform(-self.start_y_range, self.start_y_range))
        self.start_y = y0
        self.state = np.array([self.start_center[0], y0, 0.0, 0.0], dtype=np.float32)
        self.prev_exec_action = np.zeros(2, dtype=np.float32)
        self.t = 0
        clearances = self._per_obstacle_clearances(self.state[:2])
        nearest_id = int(np.argmin(clearances))
        info = {
            "state": self.state.copy(),
            "success": False,
            "distance_to_goal": float(np.linalg.norm(self.state[:2] - self.goal)),
            "clearance": float(np.min(clearances)),
            "clearances": clearances.astype(np.float32),
            "nearest_obstacle_id": nearest_id,
            "h_sq": float(np.min(np.sum((self.state[:2][None, :] - self.obstacle_centers) ** 2, axis=1) - self.obstacle_radii ** 2)),
            "map_id": self.map_id,
            "route_variant": self.route_variant,
            "start_y": float(self.start_y),
        }
        return self._get_obs_from_state(self.state), info

    def _safe_checks(self, pos_next):
        clearances = self._per_obstacle_clearances(pos_next)
        tight_clearances = self._per_obstacle_tight_clearances(pos_next)
        collision = bool(np.min(clearances) < 0.0)
        tightened_violation = bool(np.min(tight_clearances) < 0.0)
        in_box = self._state_in_workspace(pos_next)
        h_sq = float(np.min(np.sum((pos_next[None, :] - self.obstacle_centers) ** 2, axis=1) - self.obstacle_radii ** 2))
        nearest_obstacle_id = int(np.argmin(clearances))
        return (
            collision,
            in_box,
            tightened_violation,
            float(np.min(clearances)),
            clearances.astype(np.float32),
            h_sq,
            nearest_obstacle_id,
        )

    def step(self, raw_action):
        raw_action = np.clip(np.asarray(raw_action, dtype=np.float32), -1.0, 1.0)
        state_before = self.state.copy()
        dist_goal_prev = float(np.linalg.norm(state_before[:2] - self.goal))

        if self.use_filter:
            (
                exec_action_norm,
                filter_active,
                projection_gap,
                safe_violation,
                safe_set_empty,
                filter_fallback,
                filter_details,
            ) = self.filter.project_action_np(state_before, raw_action, return_details=True)
        else:
            exec_action_norm = raw_action.copy()
            filter_active = False
            projection_gap = 0.0
            safe_violation = not self.filter.is_action_feasible_np(state_before, raw_action)
            safe_set_empty = False
            filter_fallback = False
            filter_details = {
                "min_predicted_tight_clearance": float(np.min(self.filter.predicted_tight_clearances(state_before, raw_action))),
                "nearest_obstacle_id": int(np.argmin(self.filter.predicted_tight_clearances(state_before, raw_action))),
            }

        noise_p = self.rng.normal(0.0, self.noise_sigma[0], size=2).astype(np.float32)
        noise_v = self.rng.normal(0.0, self.noise_sigma[1], size=2).astype(np.float32)
        p = state_before[:2]
        v = state_before[2:]
        a = exec_action_norm * (self.a_max * self.action_gain)
        p_next = p + self.dt * v + 0.5 * (self.dt ** 2) * a
        v_next = self.damping * v + self.dt * a
        v_next = np.clip(v_next, -self.v_max, self.v_max)
        p_next = p_next + noise_p
        v_next = v_next + noise_v
        next_state = np.array([p_next[0], p_next[1], v_next[0], v_next[1]], dtype=np.float32)

        collision, in_box, tightened_violation, clearance, clearances, h_sq, nearest_obstacle_id = self._safe_checks(p_next)
        state_violation = bool((not in_box) or tightened_violation)

        success = np.linalg.norm(next_state[:2] - self.goal) <= self.goal_radius
        reward, dist_next = self._compute_reward(
            state_before, next_state, raw_action, exec_action_norm, success, state_violation, dist_goal_prev
        )

        self.state = next_state
        self.prev_exec_action = exec_action_norm.astype(np.float32)
        self.t += 1
        terminated = bool(success or state_violation)
        truncated = bool(self.t >= self.episode_len)

        obs_next = self._get_obs_from_state(self.state)
        projection_residual = float(np.linalg.norm(raw_action - exec_action_norm))
        projection_cost = float(projection_residual ** 2)
        info = {
            "state": state_before.copy(),
            "next_state": next_state.copy(),
            "raw_action": raw_action.copy(),
            "exec_action": exec_action_norm.copy(),
            "raw_u": (self.a_max * self.action_gain * raw_action).copy(),
            "exec_u": (self.a_max * self.action_gain * exec_action_norm).copy(),
            "filter_active": bool(filter_active),
            "filter_activated": bool(filter_active),
            "filter_fallback": bool(filter_fallback),
            "safe_violation": bool(safe_violation),
            "safe_set_empty": bool(safe_set_empty),
            "projection_gap": float(projection_gap),
            "projection_residual": projection_residual,
            "projection_cost": projection_cost,
            "state_violation": bool(state_violation),
            "tightened_violation": bool(tightened_violation),
            "is_success": bool(success),
            "clearance": clearance,
            "clearances": clearances.copy(),
            "nearest_obstacle_id": int(nearest_obstacle_id),
            "h_sq": h_sq,
            "distance_to_goal": dist_next,
            "distance_to_obstacle": clearance,
            "raw_velocity": state_before[2:].copy(),
            "exec_velocity": v_next.copy(),
            "action_gain": float(self.action_gain),
            "success": bool(success),
            "collision": bool(collision),
            "map_id": self.map_id,
            "route_variant": self.route_variant,
            "start_y": float(self.start_y),
            "min_predicted_tight_clearance": float(filter_details["min_predicted_tight_clearance"]),
        }
        return obs_next, reward, terminated, truncated, info
