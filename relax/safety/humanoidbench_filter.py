from dataclasses import dataclass
from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np


@dataclass
class HumanoidSafeFilterConfig:
    action_limit: float = 1.0
    residual_radius: float = 0.35
    smooth_radius: float = 0.25
    mode: str = "action"
    max_delta: float = 0.1
    target_step_radius: float = 0.08
    reachable_radius: float = 0.45
    z_min_safe: float = 0.4
    z_max_safe: float = 1.8
    eps: float = 1e-6


class HumanoidSafeFilter:
    def __init__(self, cfg: HumanoidSafeFilterConfig):
        self.cfg = cfg
        self.prev_exec_action = None

    def reset(self):
        self.prev_exec_action = None

    def _project_ball(self, x: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
        d = x - center
        n = np.linalg.norm(d)
        if n <= radius:
            return x
        return center + d * (radius / (n + self.cfg.eps))

    def project(self, raw_action: np.ndarray, prior_action: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        raw_action = np.asarray(raw_action, dtype=np.float32)
        if prior_action is None:
            prior_action = np.zeros_like(raw_action)
        else:
            prior_action = np.asarray(prior_action, dtype=np.float32)

        residual = raw_action - prior_action
        residual_norm = np.linalg.norm(residual)
        if residual_norm > self.cfg.residual_radius:
            exec_action = prior_action + residual * (self.cfg.residual_radius / (residual_norm + self.cfg.eps))
        else:
            exec_action = raw_action.copy()

        if self.prev_exec_action is not None:
            exec_action = self._project_ball(exec_action, self.prev_exec_action, self.cfg.smooth_radius)

        exec_action = np.clip(exec_action, -self.cfg.action_limit, self.cfg.action_limit)
        self.prev_exec_action = exec_action.copy()

        projection = exec_action - raw_action
        projection_residual = float(np.linalg.norm(projection))
        info = {
            "exec_action": exec_action,
            "projection_residual": projection_residual,
            "projection_cost": float(np.sum(projection ** 2)),
            "filter_active": float(projection_residual > 1e-6),
            "raw_action_norm": float(np.linalg.norm(raw_action)),
            "exec_action_norm": float(np.linalg.norm(exec_action)),
            "prior_deviation": float(np.linalg.norm(exec_action - prior_action)),
        }
        return exec_action, info

    def project_high_level_reach(
        self,
        raw_action: np.ndarray,
        last_target: np.ndarray,
        target_low: np.ndarray,
        target_high: np.ndarray,
        hand_pos: np.ndarray,
        prior_action: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, dict]:
        del prior_action
        raw_action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if raw_action.shape[0] != 3:
            raise ValueError(f"Expected raw_action shape (3,), got {raw_action.shape}")
        last_target = np.asarray(last_target, dtype=np.float32).reshape(3)
        target_low = np.asarray(target_low, dtype=np.float32).reshape(3)
        target_high = np.asarray(target_high, dtype=np.float32).reshape(3)
        hand_pos = np.asarray(hand_pos, dtype=np.float32).reshape(3)

        clipped_raw_action = np.clip(raw_action, -1.0, 1.0)
        raw_target = last_target + self.cfg.max_delta * clipped_raw_action
        filtered_target = raw_target.copy()

        step_delta = filtered_target - last_target
        step_norm = np.linalg.norm(step_delta)
        if step_norm > self.cfg.target_step_radius:
            filtered_target = last_target + step_delta * (self.cfg.target_step_radius / (step_norm + self.cfg.eps))

        filtered_target = np.clip(filtered_target, target_low, target_high)
        filtered_target[2] = np.clip(filtered_target[2], self.cfg.z_min_safe, self.cfg.z_max_safe)

        reach_delta = filtered_target - hand_pos
        reach_norm = np.linalg.norm(reach_delta)
        if reach_norm > self.cfg.reachable_radius:
            filtered_target = hand_pos + reach_delta * (self.cfg.reachable_radius / (reach_norm + self.cfg.eps))

        filtered_action = (filtered_target - last_target) / self.cfg.max_delta
        filtered_action = np.clip(filtered_action, -1.0, 1.0)

        action_projection = filtered_action - raw_action
        target_projection = filtered_target - raw_target
        projection_residual = float(np.linalg.norm(action_projection))
        info = {
            "exec_action": filtered_action,
            "projection_residual": projection_residual,
            "projection_cost": float(np.sum(action_projection ** 2)),
            "target_projection_residual": float(np.linalg.norm(target_projection)),
            "filter_active": float(projection_residual > 1e-6 or np.linalg.norm(target_projection) > 1e-6),
            "raw_target": raw_target,
            "exec_target": filtered_target,
        }
        return filtered_action, info


def project_action_jax_humanoid(raw_action: jnp.ndarray, prior_action: Optional[jnp.ndarray] = None,
                                residual_radius: float = 0.35, action_limit: float = 1.0, eps: float = 1e-6):
    if prior_action is None:
        prior_action = jnp.zeros_like(raw_action)
    residual = raw_action - prior_action
    residual_norm = jnp.linalg.norm(residual, axis=-1, keepdims=True)
    scale = jnp.minimum(1.0, residual_radius / (residual_norm + eps))
    exec_action = prior_action + residual * scale
    exec_action = jnp.clip(exec_action, -action_limit, action_limit)
    diff = exec_action - raw_action
    projection_cost = jnp.sum(diff ** 2, axis=-1)
    projection_residual = jnp.linalg.norm(diff, axis=-1)
    return exec_action, projection_cost, projection_residual
