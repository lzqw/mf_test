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

    def parse_lidar_danger(self, obs, info):
        danger = dict(global_min_dist=np.nan, front_min_dist=np.nan, left_min_dist=np.nan, right_min_dist=np.nan)
        arr = None
        if isinstance(info, dict):
            for k in ["hazards_lidar", "hazard_lidar", "obstacles_lidar", "lidar"]:
                if k in info:
                    cand = np.asarray(info[k], dtype=np.float32).reshape(-1)
                    if cand.size > 0:
                        arr = cand
                        break
        if arr is None:
            arr = np.asarray(obs, dtype=np.float32).reshape(-1)
            if arr.size > 8:
                arr = arr[(arr >= 0.0) & (arr <= 1.5)]
            else:
                arr = np.asarray([], dtype=np.float32)
        if arr.size == 0:
            return danger
        danger["global_min_dist"] = float(np.nanmin(arr))
        n = int(arr.size)
        if n >= 4:
            q = max(n // 4, 1)
            front_idx = np.concatenate([np.arange(0, q), np.arange(n - q, n)])
            left_idx = np.arange(n // 4, n // 2)
            right_idx = np.arange(n // 2, 3 * n // 4)
            danger["front_min_dist"] = float(np.nanmin(arr[front_idx])) if front_idx.size > 0 else np.nan
            danger["left_min_dist"] = float(np.nanmin(arr[left_idx])) if left_idx.size > 0 else np.nan
            danger["right_min_dist"] = float(np.nanmin(arr[right_idx])) if right_idx.size > 0 else np.nan
        return danger

    def is_safe_candidate(self, a, danger, env_id):
        d_stop = self.cfg.hazard_stop
        g = danger.get("global_min_dist", np.nan)
        if np.isfinite(g) and g <= d_stop:
            return False
        front = danger.get("front_min_dist", np.nan)
        left = danger.get("left_min_dist", np.nan)
        right = danger.get("right_min_dist", np.nan)
        if "Car" in env_id and a.shape[0] >= 2:
            steer, speed = float(a[0]), float(a[1])
            if np.isfinite(front) and front <= self.cfg.hazard_warning and speed > 0.3:
                return False
            if np.isfinite(left) and left <= self.cfg.hazard_warning and steer < -0.25:
                return False
            if np.isfinite(right) and right <= self.cfg.hazard_warning and steer > 0.25:
                return False
        else:
            if np.isfinite(front) and front <= self.cfg.hazard_warning and np.linalg.norm(a) > 0.3:
                return False
            if a.shape[0] >= 2:
                if np.isfinite(left) and left <= self.cfg.hazard_warning and a[0] < -0.2:
                    return False
                if np.isfinite(right) and right <= self.cfg.hazard_warning and a[0] > 0.2:
                    return False
        return True

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

        num_candidates = 1
        num_safe_candidates = 0
        emergency_active = 0.0
        safe_candidate_ratio = 0.0
        global_min_h = front_h = left_h = right_h = np.nan
        if self.filter_type == "sample_shield":
            projected = a.copy()
            danger = self.parse_lidar_danger(obs, info)
            global_min_h = float(danger["global_min_dist"] - self.cfg.hazard_stop) if np.isfinite(danger["global_min_dist"]) else np.nan
            front_h = float(danger["front_min_dist"] - self.cfg.hazard_stop) if np.isfinite(danger["front_min_dist"]) else np.nan
            left_h = float(danger["left_min_dist"] - self.cfg.hazard_stop) if np.isfinite(danger["left_min_dist"]) else np.nan
            right_h = float(danger["right_min_dist"] - self.cfg.hazard_stop) if np.isfinite(danger["right_min_dist"]) else np.nan
            cands = [projected, prev.copy(), np.zeros_like(projected)]
            for s in [0.25, 0.5, 0.75]:
                cands.append(projected * s)
            if "Car" in env_id and projected.shape[0] >= 2:
                cands += [
                    np.array([projected[0], min(projected[1], 0.0)], dtype=np.float32),
                    np.array([-0.4, -0.5], dtype=np.float32),
                    np.array([0.4, -0.5], dtype=np.float32),
                    np.array([-0.35, 0.2], dtype=np.float32),
                    np.array([0.35, 0.2], dtype=np.float32),
                ]
            else:
                if projected.shape[0] >= 2:
                    perp = np.array([-projected[1], projected[0]], dtype=np.float32)
                    nrm = np.linalg.norm(perp)
                    if nrm > self.cfg.eps:
                        perp = 0.4 * perp / nrm
                        cands += [perp, -perp]
                    for gx in [-0.2, 0.0, 0.2]:
                        for gy in [-0.2, 0.0, 0.2]:
                            cands.append(np.array([gx, gy], dtype=np.float32))
                cands.append(-0.5 * projected)
            cands = [np.clip(ci, action_space.low, action_space.high) for ci in cands]
            safe = [ci for ci in cands if self.is_safe_candidate(ci, danger, env_id)]
            num_candidates = len(cands)
            num_safe_candidates = len(safe)
            safe_candidate_ratio = float(num_safe_candidates / max(num_candidates, 1))
            if safe:
                costs = [float(np.sum((ci - a0) ** 2) + 0.1 * np.sum((ci - prev) ** 2)) for ci in safe]
                a = safe[int(np.argmin(costs))]
            else:
                emergency_active = 1.0
                if "Car" in env_id and projected.shape[0] >= 2:
                    a = np.array([projected[0] * 0.2, min(projected[1], 0.0)], dtype=np.float32)
                else:
                    a = np.zeros_like(projected) if projected.shape[0] < 2 else np.array([0.0, -0.15], dtype=np.float32)

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
            "num_candidates": float(num_candidates),
            "num_safe_candidates": float(num_safe_candidates),
            "safe_candidate_ratio": float(safe_candidate_ratio),
            "emergency_active": float(emergency_active),
            "global_min_h": global_min_h,
            "front_h": front_h,
            "left_h": left_h,
            "right_h": right_h,
            "filter_active_005": float(residual > 0.05),
            "filter_active_010": float(residual > 0.10),
        }
        return a, filter_info
