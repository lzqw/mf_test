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
        augment_reach_obs: bool = False,
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
        self.augment_reach_obs = augment_reach_obs
        self.safe_filter = HumanoidSafeFilter(filter_cfg or HumanoidSafeFilterConfig())
        if self.augment_reach_obs and self.env.action_space.shape[-1] == 3:
            base_obs_space = self.env.observation_space
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(base_obs_space.shape[0] + 9,),
                dtype=np.float32,
            )

    def reset(self, *, seed=None, options=None):
        self.safe_filter.reset()
        obs, info = self.env.reset(seed=seed, options=options)
        return self._augment_obs(obs), info

    def _get_hier_task(self):
        return self.env.unwrapped.task

    def _get_base_task(self):
        task = self.env.unwrapped.task
        if hasattr(task, "task"):
            return task.task
        return task

    def _augment_obs(self, obs):
        if not self.augment_reach_obs:
            return obs
        try:
            if self.env.action_space.shape[-1] != 3:
                return obs
            hier_task = self._get_hier_task()
            base_task = self._get_base_task()
            last_target = np.asarray(hier_task.last_target, dtype=np.float32).reshape(3)
            goal = np.asarray(base_task.goal, dtype=np.float32).reshape(3)
            hand = np.asarray(self.env.unwrapped.robot.left_hand_position(), dtype=np.float32).reshape(3)
            extra = np.concatenate([last_target, goal - last_target, hand - goal]).astype(np.float32)
            return np.concatenate([np.asarray(obs, dtype=np.float32), extra], axis=0)
        except Exception:
            return obs

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
            exec_action, filter_info = self._project_action(raw_action)
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
        next_obs = self._augment_obs(next_obs)
        info = dict(info)
        success_flag = bool(info.get("success", False)) or float(info.get("reward_success", 0.0) > 0.0)
        terminated_failure = float(bool(terminated) and not success_flag)
        info.update(filter_info)
        info.update(self._get_safety_metrics())
        info["terminated_failure"] = terminated_failure
        if terminated_failure > 0:
            info["fall"] = max(float(info.get("fall", 0.0)), 1.0)
        info["raw_action"] = raw_action
        info["action"] = exec_action
        info["safe_violation"] = max(float(info.get("fall", 0.0)), terminated_failure)
        info["state_violation"] = max(float(info.get("fall", 0.0)), terminated_failure)
        success_value = info.get("success", None)
        if success_value is None:
            success_value = float(info.get("reward_success", 0.0) > 0.0)
        else:
            success_value = float(success_value)

        info["is_success"] = float(success_value > 0.0)
        return next_obs, reward, terminated, truncated, info

    def _project_action(self, raw_action: np.ndarray):
        if raw_action.shape[-1] != 3:
            return self.safe_filter.project(raw_action)
        try:
            task = self.env.unwrapped.task
            last_target = np.asarray(task.last_target, dtype=np.float32)
            target_low = np.asarray(task.target_low, dtype=np.float32)
            target_high = np.asarray(task.target_high, dtype=np.float32)
            hand_pos = np.asarray(self.env.unwrapped.robot.left_hand_position(), dtype=np.float32)
            return self.safe_filter.project_high_level_reach(
                raw_action=raw_action,
                last_target=last_target,
                target_low=target_low,
                target_high=target_high,
                hand_pos=hand_pos,
            )
        except Exception:
            return self.safe_filter.project(raw_action)
