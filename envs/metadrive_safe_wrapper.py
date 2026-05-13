from typing import Any

import gymnasium as gym
import numpy as np
import relax.env  # noqa: F401

from relax.safety.metadrive_sample_filter import SampleBasedVehicleSafetyFilter, SampleVehicleFilterConfig


class SafeMetaDriveSampleWrapper(gym.Wrapper):
    def __init__(self, env_name="FlatThreeLaneStraight", use_filter=True, filter_type="sample_vo", filter_cfg=None, render_mode=None, **env_kwargs):
        env = gym.make(env_name, render_mode=render_mode, **env_kwargs)
        super().__init__(env)
        assert isinstance(self.action_space, gym.spaces.Box) and self.action_space.shape == (2,)
        self.filter_type = filter_type
        self.use_filter = bool(use_filter and filter_type != "none")
        self.safe_filter = SampleBasedVehicleSafetyFilter(filter_cfg or SampleVehicleFilterConfig())
        self.prev_exec_action = np.zeros(2, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.safe_filter.reset()
        self.prev_exec_action = np.zeros(2, dtype=np.float32)
        return self.env.reset(seed=seed, options=options)

    def _simple_rate(self, raw):
        exec_action = np.asarray(raw, dtype=np.float32).copy()
        exec_action[0] = np.clip(exec_action[0], -0.7, 0.7)
        exec_action[1] = np.clip(exec_action[1], -0.8, 0.8)
        d = np.clip(exec_action - self.prev_exec_action, [-0.12, -0.2], [0.12, 0.2])
        return np.clip(self.prev_exec_action + d, -1.0, 1.0)

    def _default_filter_info(self, raw_action, exec_action, projection_cost, projection_residual):
        return dict(
            raw_action=raw_action,
            exec_action=exec_action,
            projection_residual=float(projection_residual),
            projection_cost=float(projection_cost),
            filter_active=float(projection_residual > 1e-6),
            raw_action_norm=float(np.linalg.norm(raw_action)),
            exec_action_norm=float(np.linalg.norm(exec_action)),
            raw_steer=float(raw_action[0]),
            exec_steer=float(exec_action[0]),
            raw_accel=float(raw_action[1]),
            exec_accel=float(exec_action[1]),
            sample_filter_active=0.0,
            num_candidates=0.0,
            num_valid_candidates=0.0,
            valid_candidate_ratio=0.0,
            no_safe_candidate=0.0,
            fallback=0.0,
            min_pred_dist=float("inf"),
            min_pred_ttc=float("inf"),
            min_pred_h_vo=float("inf"),
            min_lane_margin=float("inf"),
            vo_active=0.0,
            ttc_violation=0.0,
            vo_violation=0.0,
            lane_violation=0.0,
            predicted_collision=0.0,
            predicted_offroad=0.0,
            filter_time_ms=0.0,
        )

    def step(self, raw_action):
        raw_action = np.asarray(raw_action, dtype=np.float32)
        if self.use_filter and self.filter_type == "sample_vo":
            exec_action, filter_info = self.safe_filter.project(raw_action, env=self.env, prev_exec_action=self.prev_exec_action)
        elif self.use_filter and self.filter_type == "rate":
            exec_action = self._simple_rate(raw_action)
            diff = exec_action - raw_action
            filter_info = self._default_filter_info(raw_action, exec_action, np.sum(diff**2), np.linalg.norm(diff))
        else:
            exec_action = np.clip(raw_action, -1.0, 1.0)
            diff = exec_action - raw_action
            filter_info = self._default_filter_info(raw_action, exec_action, np.sum(diff**2), np.linalg.norm(diff))
        self.prev_exec_action = np.asarray(exec_action, dtype=np.float32)

        obs, reward, terminated, truncated, info = self.env.step(exec_action)
        info = dict(info)
        info.update(filter_info)
        crash = float(info.get("crash", 0.0))
        out_of_road = float(info.get("out_of_road", 0.0))
        cost = float(info.get("cost", crash + out_of_road))
        success = float(info.get("arrive_dest", info.get("success", 0.0)))
        info["safe_violation"] = float(max(crash, out_of_road, float(cost > 0.0)))
        info["state_violation"] = info["safe_violation"]
        info["is_success"] = float(success > 0.0)
        return obs, reward, terminated, truncated, info
