from typing import Any, Dict

import gymnasium as gym
import humanoid_bench  # noqa: F401
import numpy as np

from relax.safety.humanoidbench_filter import HumanoidSafeFilter, HumanoidSafeFilterConfig


class SafeHumanoidBenchWrapper(gym.Wrapper):
    def __init__(
        self,
        env_name: str = "h1hand-reach-v0",
        use_filter: bool = True,
        render_mode=None,
        filter_cfg: HumanoidSafeFilterConfig | None = None,
        policy_path: str | None = None,
        mean_path: str | None = None,
        var_path: str | None = None,
        policy_type: str | None = None,
        **env_kwargs,
    ):
        if policy_type is not None:
            env_kwargs.update(
                policy_path=policy_path,
                mean_path=mean_path,
                var_path=var_path,
                policy_type=policy_type,
            )
        env = gym.make(env_name, render_mode=render_mode, **env_kwargs)
        super().__init__(env)
        self.use_filter = use_filter
        self.safe_filter = HumanoidSafeFilter(filter_cfg or HumanoidSafeFilterConfig())

    def reset(self, *, seed=None, options=None):
        self.safe_filter.reset()
        return self.env.reset(seed=seed, options=options)

    def _get_safety_metrics(self) -> Dict[str, float]:
        out = dict(head_height=0.0, torso_upright=1.0, fall=0.0,
                   joint_angle_abs_mean=0.0, joint_angle_abs_max=0.0,
                   joint_vel_abs_mean=0.0, joint_vel_abs_max=0.0)
        try:
            robot = self.env.unwrapped.robot
            head_height = float(robot.head_height())
            torso_upright = float(robot.torso_upright())
            try:
                joint_angles = np.asarray(robot.joint_angles(), dtype=np.float32)
            except Exception:
                joint_angles = np.zeros((1,), dtype=np.float32)
            try:
                joint_vel = np.asarray(robot.joint_velocities(), dtype=np.float32)
            except Exception:
                joint_vel = np.zeros((1,), dtype=np.float32)
            out.update(
                head_height=head_height,
                torso_upright=torso_upright,
                fall=float(head_height < 0.65 or torso_upright < 0.4),
                joint_angle_abs_mean=float(np.mean(np.abs(joint_angles))),
                joint_angle_abs_max=float(np.max(np.abs(joint_angles))),
                joint_vel_abs_mean=float(np.mean(np.abs(joint_vel))),
                joint_vel_abs_max=float(np.max(np.abs(joint_vel))),
            )
        except Exception:
            pass
        return out

    def step(self, raw_action):
        raw_action = np.asarray(raw_action, dtype=np.float32)
        if self.use_filter:
            exec_action, filter_info = self.safe_filter.project(raw_action)
        else:
            exec_action = np.clip(raw_action, -1.0, 1.0)
            projection = exec_action - raw_action
            projection_residual = float(np.linalg.norm(projection))
            filter_info = {
                "exec_action": exec_action,
                "projection_residual": projection_residual,
                "projection_cost": float(np.sum(projection ** 2)),
                "filter_active": float(projection_residual > 1e-6),
                "raw_action_norm": float(np.linalg.norm(raw_action)),
                "exec_action_norm": float(np.linalg.norm(exec_action)),
                "prior_deviation": float(np.linalg.norm(exec_action)),
            }
        next_obs, reward, terminated, truncated, info = self.env.step(exec_action)
        info = dict(info)
        info.update(filter_info)
        info.update(self._get_safety_metrics())
        info["raw_action"] = raw_action
        info["action"] = exec_action
        info.setdefault("safe_violation", info.get("fall", 0.0))
        info.setdefault("state_violation", info.get("fall", 0.0))
        info.setdefault("is_success", float(info.get("success", info.get("reward_success", 0.0))))
        return next_obs, reward, terminated, truncated, info
