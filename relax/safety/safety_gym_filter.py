from dataclasses import dataclass
import numpy as np


@dataclass
class SafetyGymFilterConfig:
    max_action_norm_point: float = 0.80
    max_action_norm_car: float = 0.70
    max_delta_point: float = 0.20
    steer_limit: float = 0.60
    speed_limit: float = 0.70
    max_dsteer: float = 0.15
    max_dspeed: float = 0.20
    hazard_warning: float = 0.35
    hazard_stop: float = 0.15
    eps: float = 1e-6


def extract_min_lidar_or_hazard(obs, info):
    keys = ["hazards_lidar", "hazard_lidar", "obstacles_lidar", "lidar"]
    if isinstance(info, dict):
        for k in keys:
            if k in info:
                arr = np.asarray(info[k], dtype=np.float32).reshape(-1)
                if arr.size > 0:
                    return float(np.nanmin(arr))
        for k in ["min_hazard_dist", "hazard_dist", "obstacle_dist"]:
            if k in info and np.isfinite(float(info[k])):
                return float(info[k])
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    if arr.size > 8:
        cands = arr[(arr >= 0.0) & (arr <= 1.5)]
        if cands.size > 0:
            return float(np.nanmin(cands))
    return np.nan


class SafetyGymHardFilter:
    def __init__(self, cfg: SafetyGymFilterConfig, filter_type="hybrid"):
        self.cfg = cfg
        self.filter_type = filter_type

    def reset(self):
        return None

    def project(self, raw_action, obs, info, prev_exec_action, action_space, env_id):
        a0 = np.asarray(raw_action, dtype=np.float32)
        a = a0.copy()
        prev = np.asarray(prev_exec_action, dtype=np.float32)
        action_clip_active = smooth_active = control_limit_active = cost_aware_active = 0.0

        if self.filter_type in ["action", "hybrid", "control"]:
            before = a.copy()
            a = np.clip(a, action_space.low, action_space.high)
            max_norm = self.cfg.max_action_norm_car if "Car" in env_id else self.cfg.max_action_norm_point
            norm = np.linalg.norm(a)
            if norm > max_norm:
                a = a / max(norm, self.cfg.eps) * max_norm
            action_clip_active = float(np.linalg.norm(a - before) > self.cfg.eps)

        if self.filter_type in ["smooth", "hybrid", "control"]:
            before = a.copy()
            if "Car" in env_id and a.shape[0] == 2 and prev.shape[0] == 2:
                a[0] = np.clip(a[0], -self.cfg.steer_limit, self.cfg.steer_limit)
                a[1] = np.clip(a[1], -self.cfg.speed_limit, self.cfg.speed_limit)
                a[0] = prev[0] + np.clip(a[0] - prev[0], -self.cfg.max_dsteer, self.cfg.max_dsteer)
                a[1] = prev[1] + np.clip(a[1] - prev[1], -self.cfg.max_dspeed, self.cfg.max_dspeed)
                control_limit_active = float(True)
            else:
                delta = np.clip(a - prev, -self.cfg.max_delta_point, self.cfg.max_delta_point)
                a = prev + delta
            smooth_active = float(np.linalg.norm(a - before) > self.cfg.eps)

        min_dist = extract_min_lidar_or_hazard(obs, info)
        if self.filter_type in ["control", "hybrid"] and np.isfinite(min_dist):
            h = float(min_dist - self.cfg.hazard_stop)
            if min_dist < self.cfg.hazard_warning:
                ss = np.clip((min_dist - self.cfg.hazard_stop) / max(self.cfg.hazard_warning - self.cfg.hazard_stop, self.cfg.eps), 0.0, 1.0)
                if "Car" in env_id and a.shape[0] >= 2:
                    a[1] = min(a[1], a[1] * ss)
                else:
                    a = a * ss
                cost_aware_active = 1.0
            if min_dist < self.cfg.hazard_stop:
                if "Car" in env_id and a.shape[0] >= 2:
                    a[1] = min(a[1], 0.0)
                else:
                    a = a * 0.2
                cost_aware_active = 1.0
            min_h = h
            cbf_violation = float(min_h < 0.0)
        else:
            min_h = np.nan
            cbf_violation = 0.0

        a = np.clip(a, action_space.low, action_space.high)
        residual = float(np.linalg.norm(a - a0))
        filter_info = {
            "raw_action": a0,
            "exec_action": a,
            "projection_residual": residual,
            "projection_cost": float(residual ** 2),
            "filter_active": float(residual > self.cfg.eps),
            "raw_action_norm": float(np.linalg.norm(a0)),
            "exec_action_norm": float(np.linalg.norm(a)),
            "action_clip_active": action_clip_active,
            "smooth_active": smooth_active,
            "control_limit_active": control_limit_active,
            "cost_aware_active": cost_aware_active,
            "min_h": min_h,
            "cbf_violation": cbf_violation,
        }
        return a, filter_info
