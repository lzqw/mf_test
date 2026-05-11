from dataclasses import dataclass
from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np


@dataclass
class HumanoidSafeFilterConfig:
    action_limit: float = 1.0
    residual_radius: float = 0.35
    smooth_radius: float = 0.25
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
