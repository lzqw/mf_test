import gymnasium as gym
import numpy as np
import safety_gymnasium

from relax.safety.safety_gym_filter import SafetyGymFilterConfig, SafetyGymHardFilter, _extract_goal_from_env


def _default_filter_info(raw_action: np.ndarray, exec_action: np.ndarray):
    residual = float(np.linalg.norm(exec_action - raw_action))
    return {
        "projection_residual": residual,
        "projection_cost": float(residual ** 2),
        "filter_active": float(residual > 1e-6),
        "raw_action_norm": float(np.linalg.norm(raw_action)),
        "exec_action_norm": float(np.linalg.norm(exec_action)),
        "action_clip_active": float(np.any(np.abs(exec_action - raw_action) > 1e-6)),
        "smooth_active": 0.0,
        "control_limit_active": 0.0,
        "cost_aware_active": 0.0,
        "min_h": np.nan,
        "cbf_violation": 0.0,
        "num_candidates": 0.0,
        "num_safe_candidates": 0.0,
        "safe_candidate_ratio": 0.0,
        "emergency_active": 0.0,
        "global_min_h": np.nan,
        "front_h": np.nan,
        "left_h": np.nan,
        "right_h": np.nan,
        "filter_active_005": float(residual > 0.05),
        "filter_active_010": float(residual > 0.10),
    }


class SafeSafetyGymWrapper(gym.Wrapper):
    def __init__(self, env_id="SafetyPointGoal1-v0", use_filter=True, filter_type="hybrid", filter_cfg=None,
                 render_mode=None, terminate_on_safety_violation=False, cost_limit_per_step=0.0, goal_threshold=None, **env_kwargs):
        env = safety_gymnasium.make(env_id, render_mode=render_mode, **env_kwargs)
        super().__init__(env)
        self.env_id = env_id
        self.use_filter = use_filter
        self.terminate_on_safety_violation = terminate_on_safety_violation
        self.cost_limit_per_step = float(cost_limit_per_step)
        cfg = filter_cfg or SafetyGymFilterConfig()
        self.safe_filter = SafetyGymHardFilter(cfg, filter_type=filter_type)
        self.goal_threshold = float(cfg.goal_threshold if goal_threshold is None else goal_threshold)
        self.last_obs = None
        self.last_info = {}
        self.prev_exec_action = np.zeros(self.action_space.shape, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_obs = obs
        self.last_info = dict(info)
        self.prev_exec_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.safe_filter.reset()
        return obs, info

    def step(self, raw_action):
        raw_action = np.asarray(raw_action, dtype=np.float32)
        if self.use_filter:
            exec_action, filter_info = self.safe_filter.project(
                raw_action=raw_action, obs=self.last_obs, info=self.last_info,
                prev_exec_action=self.prev_exec_action, action_space=self.action_space, env_id=self.env_id,
                env=self.env,
            )
        else:
            exec_action = np.clip(raw_action, self.action_space.low, self.action_space.high)
            filter_info = _default_filter_info(raw_action, exec_action)

        out = self.env.step(exec_action)
        if len(out) == 6:
            next_obs, reward, cost, terminated, truncated, info = out
        elif len(out) == 5:
            next_obs, reward, terminated, truncated, info = out
            cost = float(dict(info).get("cost", 0.0))
        else:
            raise RuntimeError(f"Unexpected env.step return length={len(out)} for env_id={self.env_id}")
        info = dict(info)
        info["cost"] = float(cost)
        info["cost_step"] = float(cost)
        info["safety_violation"] = float(cost > self.cost_limit_per_step)
        info["constraint_violation"] = float(cost > self.cost_limit_per_step)
        info["safe_violation"] = float(cost > self.cost_limit_per_step)
        info["raw_action"] = raw_action
        info["exec_action"] = exec_action
        info.setdefault("state_violation", info["safety_violation"])

        ego = self.safe_filter._extract_ego_state_from_env(self.env)
        goal = _extract_goal_from_env(self.env)
        goal_known = bool(ego.get("known", False) and goal.get("known", False))
        goal_dist = np.nan
        if goal_known:
            goal_dist = float(np.linalg.norm(np.asarray(ego["pos"], dtype=np.float32) - np.asarray(goal["pos"], dtype=np.float32)))

        has_goal_flag = any(k in info for k in ("goal_met", "success", "task_success"))
        goal_met = float(info.get("goal_met", info.get("success", info.get("task_success", 0.0))))
        goal_reached_by_dist = float(goal_known and np.isfinite(goal_dist) and goal_dist < self.goal_threshold) if not has_goal_flag else float(info.get("goal_reached_by_dist", 0.0))

        info["goal_known"] = float(goal_known)
        info["goal_x"] = float(goal["pos"][0]) if goal.get("known", False) else np.nan
        info["goal_y"] = float(goal["pos"][1]) if goal.get("known", False) else np.nan
        info["goal_dist"] = float(goal_dist) if np.isfinite(goal_dist) else np.nan
        info["goal_met"] = goal_met
        info["goal_reached_by_dist"] = goal_reached_by_dist
        info["is_success"] = max(float(info.get("is_success", 0.0)), goal_met, goal_reached_by_dist)

        info.update(filter_info)

        if self.terminate_on_safety_violation and cost > self.cost_limit_per_step:
            terminated = True
            info["terminated_by_safety"] = 1.0
        else:
            info["terminated_by_safety"] = 0.0

        self.last_obs = next_obs
        self.last_info = info
        self.prev_exec_action = exec_action
        return next_obs, reward, terminated, truncated, info
