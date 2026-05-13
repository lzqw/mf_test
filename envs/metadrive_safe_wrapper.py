from typing import Any

import gymnasium as gym
import numpy as np

from relax.safety.metadrive_sample_filter import SampleBasedVehicleSafetyFilter, SampleVehicleFilterConfig


class SafeMetaDriveSampleWrapper(gym.Wrapper):
    def __init__(self, env_name="FlatThreeLaneStraight", use_filter=True, filter_type="sample_vo", filter_cfg=None, render_mode=None, **env_kwargs):
        env = gym.make(env_name, render_mode=render_mode, **env_kwargs)
        super().__init__(env)
        assert isinstance(self.action_space, gym.spaces.Box) and self.action_space.shape == (2,)
        self.use_filter = use_filter
        self.filter_type = filter_type
        self.safe_filter = SampleBasedVehicleSafetyFilter(filter_cfg or SampleVehicleFilterConfig())
        self.prev_exec_action = np.zeros(2, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.safe_filter.reset()
        self.prev_exec_action = np.zeros(2, dtype=np.float32)
        return self.env.reset(seed=seed, options=options)

    def _simple_rate(self, raw):
        exec_action = np.asarray(raw, dtype=np.float32).copy()
        exec_action = np.clip(exec_action, -1.0, 1.0)
        d = exec_action - self.prev_exec_action
        d = np.clip(d, [-0.12, -0.2], [0.12, 0.2])
        return np.clip(self.prev_exec_action + d, -1.0, 1.0)

    def step(self, raw_action):
        raw_action = np.asarray(raw_action, dtype=np.float32)
        if self.use_filter and self.filter_type == "sample_vo":
            exec_action, filter_info = self.safe_filter.project(raw_action, env=self.env, prev_exec_action=self.prev_exec_action)
        elif self.use_filter and self.filter_type == "rate":
            exec_action = self._simple_rate(raw_action)
            diff = exec_action - raw_action
            filter_info = {"projection_residual": float(np.linalg.norm(diff)), "projection_cost": float(np.sum(diff**2)), "filter_active": float(np.linalg.norm(diff) > 1e-6)}
        else:
            exec_action = np.clip(raw_action, -1.0, 1.0)
            diff = exec_action - raw_action
            filter_info = {"projection_residual": float(np.linalg.norm(diff)), "projection_cost": float(np.sum(diff**2)), "filter_active": float(np.linalg.norm(diff) > 1e-6)}
        self.prev_exec_action = np.asarray(exec_action, dtype=np.float32)

        obs, reward, terminated, truncated, info = self.env.step(exec_action)
        info = dict(info)
        info.update(filter_info)
        info["raw_action"] = raw_action
        info["exec_action"] = exec_action
        crash = float(info.get("crash", 0.0))
        out_of_road = float(info.get("out_of_road", 0.0))
        cost = float(info.get("cost", crash + out_of_road))
        success = float(info.get("arrive_dest", info.get("success", 0.0)))
        info["safe_violation"] = float(max(crash, out_of_road, float(cost > 0.0)))
        info["state_violation"] = info["safe_violation"]
        info["is_success"] = float(success > 0.0)
        return obs, reward, terminated, truncated, info
