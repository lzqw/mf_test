import importlib.metadata as _metadata

# Disable gymnasium plugin loading to avoid optional plugin import failures in limited
# environments (e.g. when shimmy or mujoco plugins are present but read-only).
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


class SafeObstacleDoubleIntegrator2DEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        noise_sigma=(0.01, 0.01),
        use_filter=True,
        seed=0,
        start_y_range=0.45,
        dt=0.1,
        a_max=3.0,
        v_max=2.0,
        damping=0.98,
        action_gain=1.0,
        episode_len=200,
        goal=np.array([2.6, 0.0], dtype=np.float32),
        start_center=np.array([-2.6, 0.0], dtype=np.float32),
        goal_radius=0.18,
        obstacle_center=np.array([0.0, 0.0], dtype=np.float32),
        obstacle_radius=0.8,
        eps_obs=0.08,
        reward_cfg=None,
    ):
        if reward_cfg is None:
            reward_cfg = dict(
                progress_coef=8.0,
                success_bonus=100.0,
                collision_penalty=100.0,
                near_obs_coef=8.0,
                safety_buffer=0.20,
                action_coef=0.03,
                speed_coef=0.01,
                time_coef=0.01,
            )

        self.dt = float(dt)
        self.a_max = float(a_max)
        self.v_max = float(v_max)
        self.damping = float(damping)
        self.action_gain = float(action_gain)
        self.episode_len = int(episode_len)
        self.goal = np.asarray(goal, dtype=np.float32)
        self.goal_radius = float(goal_radius)
        self.start_center = np.asarray(start_center, dtype=np.float32)
        self.obstacle_center = np.asarray(obstacle_center, dtype=np.float32)
        self.obstacle_radius = float(obstacle_radius)
        self.eps_obs = float(eps_obs)
        self.noise_sigma = np.asarray(noise_sigma, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.start_y_range = float(start_y_range)
        self.use_filter = bool(use_filter)

        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        self.filter_cfg = DoubleIntegratorObstacleConfig(
            dt=self.dt,
            a_max=self.a_max,
            obstacle_center=self.obstacle_center,
            obstacle_radius=self.obstacle_radius,
            eps_obs=self.eps_obs,
        )
        self.filter = DoubleIntegratorObstacleFilter(self.filter_cfg)
        self.state = np.zeros(4, dtype=np.float32)
        self.prev_exec_action = np.zeros(2, dtype=np.float32)
        self.t = 0

        self.reward_cfg = dict(reward_cfg)

        self._workspace_limits = dict(
            x_min=-3.5,
            x_max=3.5,
            y_min=-2.0,
            y_max=2.0,
        )

    def _state_in_workspace(self, pos):
        x, y = float(pos[0]), float(pos[1])
        return (
            (self._workspace_limits["x_min"] <= x <= self._workspace_limits["x_max"])
            and (self._workspace_limits["y_min"] <= y <= self._workspace_limits["y_max"])
        )

    def _get_obs_from_state(self, state):
        pos = state[:2].astype(np.float32)
        vel = state[2:].astype(np.float32)
        rel_goal = self.goal - pos
        rel_obs = pos - self.obstacle_center
        clearance = float(np.linalg.norm(rel_obs) - self.obstacle_radius)
        d_goal = float(np.linalg.norm(rel_goal))
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
        progress_reward = cfg["progress_coef"] * (dist_goal_prev - dist_next)
        action_penalty = cfg["action_coef"] * float(np.sum(np.square(exec_action)))
        speed_penalty = cfg["speed_coef"] * float(np.sum(np.square(next_state[2:])))
        base_reward = progress_reward - action_penalty - speed_penalty - cfg["time_coef"]

        clearance = float(np.linalg.norm(next_state[:2] - self.obstacle_center) - self.obstacle_radius)
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
        v0 = np.zeros(2, dtype=np.float32)
        y0 = float(self.rng.uniform(-self.start_y_range, self.start_y_range))
        self.state = np.array([self.start_center[0], y0, v0[0], v0[1]], dtype=np.float32)
        self.prev_exec_action = np.zeros(2, dtype=np.float32)
        self.t = 0
        info = {
            "state": self.state.copy(),
            "success": False,
            "distance_to_goal": float(np.linalg.norm(self.state[:2] - self.goal)),
            "clearance": float(np.linalg.norm(self.state[:2] - self.obstacle_center) - self.obstacle_radius),
            "h_sq": float(np.sum((self.state[:2] - self.obstacle_center) ** 2) - self.obstacle_radius ** 2),
        }
        return self._get_obs_from_state(self.state), info

    def _safe_checks(self, pos_next, velocity_next):
        clearance = float(np.linalg.norm(pos_next - self.obstacle_center) - self.obstacle_radius)
        h_sq = float(np.sum((pos_next - self.obstacle_center) ** 2) - self.obstacle_radius ** 2)
        collision = clearance < 0.0
        in_box = self._state_in_workspace(pos_next)
        tightened_clearance = float(np.linalg.norm(pos_next - self.obstacle_center) - (self.obstacle_radius + self.eps_obs))
        tightened_violation = tightened_clearance < 0.0
        return collision, in_box, tightened_violation, clearance, h_sq

    def step(self, raw_action):
        raw_action = np.clip(np.asarray(raw_action, dtype=np.float32), -1.0, 1.0)
        state_before = self.state.copy()
        dist_goal_prev = float(np.linalg.norm(state_before[:2] - self.goal))

        if self.use_filter:
            exec_action_norm, filter_active, projection_gap, safe_violation, safe_set_empty, filter_fallback = self.filter.project_action_np(
                state_before, raw_action
            )
        else:
            exec_action_norm = raw_action.copy()
            filter_active = False
            projection_gap = 0.0
            safe_violation = not self.filter.is_action_feasible_np(state_before, raw_action)
            safe_set_empty = False
            filter_fallback = False

        noise = self.rng.normal(0.0, self.noise_sigma, size=2).astype(np.float32)
        p = state_before[:2]
        v = state_before[2:]
        a = exec_action_norm * (self.a_max * self.action_gain)
        p_next = p + self.dt * v + 0.5 * (self.dt ** 2) * a
        v_next = self.damping * v + self.dt * a
        v_next = np.clip(v_next, -self.v_max, self.v_max)
        p_next = p_next + noise * 0.0

        next_state = np.array([p_next[0], p_next[1], v_next[0], v_next[1]], dtype=np.float32)

        collision, in_box, tightened_violation, clearance, h_sq = self._safe_checks(p_next, v_next)
        state_violation = (not in_box) or collision

        success = np.linalg.norm(next_state[:2] - self.goal) <= self.goal_radius
        reward, dist_next = self._compute_reward(
            state_before, next_state, raw_action, exec_action_norm, success, state_violation, dist_goal_prev
        )

        self.state = next_state
        self.prev_exec_action = exec_action_norm.astype(np.float32)
        self.t += 1
        terminated = bool(success)
        truncated = bool(self.t >= self.episode_len)

        obs_next = self._get_obs_from_state(self.state)
        projection_residual = float(np.linalg.norm(raw_action - exec_action_norm))
        projection_cost = float(projection_residual ** 2)
        info = {
            "state": state_before.copy(),
            "next_state": next_state.copy(),
            "raw_action": raw_action.copy(),
            "exec_action": exec_action_norm.copy(),
            "raw_u": (self.a_max * raw_action).copy(),
            "exec_u": (self.a_max * exec_action_norm).copy(),
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
            "h_sq": h_sq,
            "distance_to_goal": dist_next,
            "distance_to_obstacle": clearance,
            "raw_velocity": state_before[2:].copy(),
            "exec_velocity": v_next.copy(),
            "action_gain": float(self.action_gain),
            "success": bool(success),
            "collision": bool(collision),
        }
        return obs_next, reward, terminated, truncated, info
