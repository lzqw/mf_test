import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from relax.safety.obstacle_navigation_filter import (
    ObstacleNavConfig,
    make_action_grid,
    is_action_feasible_np,
    is_state_safe_original_np,
    is_state_safe_tight_np,
    project_action_np,
)


class SafeObstacleNavigation2DEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, noise_sigma=(0.01, 0.01), use_filter=True, seed=0, start_y_range=0.4):
        self.cfg = ObstacleNavConfig()
        self.use_filter = use_filter
        self.u_max = 1.0
        self.dt = 0.1
        self.episode_len = 200
        self.goal = np.array([-2.6, 0.0], dtype=np.float32)
        self.goal_radius = 0.18
        self.obstacle_center = np.array([0.0, 0.0], dtype=np.float32)
        self.obstacle_radius = 0.8
        self.noise_sigma = np.asarray(noise_sigma, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.start_y_range = float(start_y_range)
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_grid_norm = make_action_grid(61)
        self.state = np.zeros(2, dtype=np.float32)
        self.prev_exec_action = np.zeros(2, dtype=np.float32)
        self.t = 0

    def _get_obs_from_state(self, state):
        pos = state.astype(np.float32)
        rel_goal = pos - self.goal
        rel_obs = pos - self.obstacle_center
        d_obs = np.linalg.norm(rel_obs) - self.obstacle_radius
        d_goal = np.linalg.norm(rel_goal)
        return np.array([pos[0], pos[1], rel_goal[0], rel_goal[1], rel_obs[0], rel_obs[1], d_obs, d_goal], dtype=np.float32)

    def _compute_reward(self, state, next_state, exec_action, prev_exec_action, success, state_violation):
        dist_now = np.linalg.norm(state - self.goal)
        dist_next = np.linalg.norm(next_state - self.goal)
        progress = dist_now - dist_next

        clearance = np.linalg.norm(next_state - self.obstacle_center) - self.obstacle_radius
        action_cost = np.sum(exec_action ** 2)
        delta_action_cost = np.sum((exec_action - prev_exec_action) ** 2)

        reward = 0.0
        reward += 8.0 * progress
        reward -= 0.15 * dist_next
        reward -= 0.02 * action_cost
        reward -= 0.01 * delta_action_cost

        if clearance < 0.15:
            reward -= 2.0 * (0.15 - clearance)

        if success:
            reward += 50.0

        if state_violation:
            reward -= 100.0

        return float(reward)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = np.array([2.6, self.rng.uniform(-self.start_y_range, self.start_y_range)], dtype=np.float32)
        self.prev_exec_action = np.zeros(2, dtype=np.float32)
        self.t = 0
        obs = self._get_obs_from_state(self.state)
        info = {
            "state": self.state.copy(),
            "distance_to_goal": float(np.linalg.norm(self.state - self.goal)),
            "distance_to_obstacle": float(np.linalg.norm(self.state - self.obstacle_center) - self.obstacle_radius),
        }
        return obs, info

    def step(self, raw_action_norm):
        raw_action_norm = np.clip(np.asarray(raw_action_norm, dtype=np.float32), -1.0, 1.0)
        state_before = self.state.copy()
        prev_exec_action = self.prev_exec_action.copy()
        if self.use_filter:
            exec_action_norm, filter_active, projection_gap, safe_violation, safe_set_empty = project_action_np(
                state_before, raw_action_norm, self.action_grid_norm, self.cfg
            )
        else:
            exec_action_norm = raw_action_norm.copy()
            filter_active = False
            projection_gap = 0.0
            safe_violation = not is_action_feasible_np(state_before, raw_action_norm, self.cfg)
            safe_set_empty = False
        noise = self.rng.normal(0.0, self.noise_sigma, size=2).astype(np.float32)
        next_state = state_before + self.dt * self.u_max * exec_action_norm + noise
        success = np.linalg.norm(next_state - self.goal) <= self.goal_radius
        state_violation = not is_state_safe_original_np(next_state, self.cfg)
        tightened_violation = not is_state_safe_tight_np(next_state, self.cfg)
        reward = self._compute_reward(state_before, next_state, exec_action_norm, prev_exec_action, success, state_violation)
        self.state = next_state.astype(np.float32)
        self.prev_exec_action = exec_action_norm.astype(np.float32)
        self.t += 1
        terminated = bool(success)
        truncated = bool(self.t >= self.episode_len)
        obs_next = self._get_obs_from_state(self.state)
        projection_residual = float(np.linalg.norm(raw_action_norm - exec_action_norm))
        projection_cost = float(projection_residual ** 2)
        info = {
            "state": state_before.copy(), "next_state": next_state.copy(), "raw_action": raw_action_norm.copy(),
            "exec_action": exec_action_norm.copy(), "raw_u": (self.u_max * raw_action_norm).copy(),
            "exec_u": (self.u_max * exec_action_norm).copy(), "filter_active": bool(filter_active),
            "safe_violation": bool(safe_violation), "safe_set_empty": bool(safe_set_empty),
            "projection_gap": float(projection_gap), "projection_residual": projection_residual,
            "projection_cost": projection_cost, "state_violation": bool(state_violation),
            "tightened_violation": bool(tightened_violation), "is_success": bool(success),
            "distance_to_goal": float(np.linalg.norm(next_state - self.goal)),
            "distance_to_obstacle": float(np.linalg.norm(next_state - self.obstacle_center) - self.obstacle_radius),
            "noise": noise.copy(),
        }
        return obs_next, reward, terminated, truncated, info
