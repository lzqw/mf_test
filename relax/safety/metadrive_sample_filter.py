from dataclasses import dataclass
import time
import numpy as np


@dataclass
class SampleVehicleFilterConfig:
    action_limit: float = 1.0
    steer_limit: float = 0.7
    throttle_limit: float = 0.8
    brake_limit: float = -0.8
    max_dsteer: float = 0.12
    max_daccel: float = 0.20
    dt: float = 0.1
    horizon: int = 8
    v_max: float = 20.0
    k_steer: float = 0.35
    k_accel: float = 3.0
    num_local_samples: int = 64
    num_prev_samples: int = 32
    local_sigma_steer: float = 0.20
    local_sigma_accel: float = 0.25
    prev_sigma: float = 0.15
    num_obstacles: int = 4
    safe_radius: float = 4.0
    ego_radius: float = 1.5
    obs_radius: float = 1.5
    ttc_min: float = 1.5
    h_vo_margin: float = 0.2
    lane_margin_min: float = 0.3
    w_raw: float = 1.0
    w_rate: float = 0.3
    w_lane: float = 0.2
    w_progress: float = 0.1
    w_margin: float = 0.05
    eps: float = 1e-6
    seed: int = 0


class SampleBasedVehicleSafetyFilter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.prev_exec_action = np.zeros(2, dtype=np.float32)

    def reset(self):
        self.prev_exec_action = np.zeros(2, dtype=np.float32)

    def _box_rate(self, a, prev):
        a = np.asarray(a, dtype=np.float32).copy()
        a[0] = np.clip(a[0], -self.cfg.steer_limit, self.cfg.steer_limit)
        a[1] = np.clip(a[1], self.cfg.brake_limit, self.cfg.throttle_limit)
        d = a - prev
        d[0] = np.clip(d[0], -self.cfg.max_dsteer, self.cfg.max_dsteer)
        d[1] = np.clip(d[1], -self.cfg.max_daccel, self.cfg.max_daccel)
        return np.clip(prev + d, -1.0, 1.0)

    def _extract_ego_state(self, env):
        try:
            u = getattr(env, "unwrapped", env)
            v = getattr(u, "vehicle", None) or getattr(u, "agent", None)
            pos = np.asarray(getattr(v, "position", [0.0, 0.0]), dtype=np.float32)
            heading = float(getattr(v, "heading_theta", getattr(v, "heading", 0.0)))
            speed = float(getattr(v, "speed", np.linalg.norm(getattr(v, "velocity", [0.0, 0.0]))))
            vel = np.asarray(getattr(v, "velocity", [speed * np.cos(heading), speed * np.sin(heading)]), dtype=np.float32)
            return dict(x=float(pos[0]), y=float(pos[1]), heading=heading, speed=speed, v=vel)
        except Exception:
            return dict(x=0.0, y=0.0, heading=0.0, speed=0.0, v=np.zeros(2, dtype=np.float32))

    def _extract_obstacles(self, env):
        obs = []
        try:
            u = getattr(env, "unwrapped", env)
            mgr = getattr(u, "traffic_manager", None)
            vehicles = []
            if mgr is not None:
                vehicles = list(getattr(mgr, "vehicles", {}).values())
            vehicles = vehicles[: self.cfg.num_obstacles]
            for v in vehicles:
                pos = np.asarray(getattr(v, "position", [0.0, 0.0]), dtype=np.float32)
                heading = float(getattr(v, "heading_theta", getattr(v, "heading", 0.0)))
                speed = float(getattr(v, "speed", np.linalg.norm(getattr(v, "velocity", [0.0, 0.0]))))
                vel = np.asarray(getattr(v, "velocity", [speed * np.cos(heading), speed * np.sin(heading)]), dtype=np.float32)
                obs.append(dict(x=float(pos[0]), y=float(pos[1]), heading=heading, speed=speed, v=vel))
        except Exception:
            return []
        return obs

    def _extract_lane_info(self, env):
        try:
            u = getattr(env, "unwrapped", env)
            v = getattr(u, "vehicle", None)
            lane = getattr(v, "lane", None)
            width = float(getattr(lane, "width", 4.0))
            lateral = float(getattr(v, "lateral", 0.0)) if hasattr(v, "lateral") else 0.0
            return dict(lane_available=True, lane_margin=max(0.0, width / 2.0 - abs(lateral)), lane_center_error=abs(lateral))
        except Exception:
            return dict(lane_available=False, lane_margin=np.inf, lane_center_error=0.0)

    def project(self, raw_action, env=None, prev_exec_action=None):
        t0 = time.perf_counter()
        raw = np.asarray(raw_action, dtype=np.float32).reshape(2)
        prev = self.prev_exec_action if prev_exec_action is None else np.asarray(prev_exec_action, dtype=np.float32).reshape(2)
        clipped = self._box_rate(raw, prev)
        ego = self._extract_ego_state(env)
        obstacles = self._extract_obstacles(env)
        lane = self._extract_lane_info(env)

        cands = [raw, np.clip(raw, -1, 1), prev, np.zeros(2), np.array([0.0, -0.3]), np.array([0.0, self.cfg.brake_limit]), np.array([raw[0], -0.4]), np.array([0.3, -0.4]), np.array([-0.3, -0.4])]
        cands += [raw + self.rng.normal([0, 0], [self.cfg.local_sigma_steer, self.cfg.local_sigma_accel]) for _ in range(self.cfg.num_local_samples)]
        cands += [prev + self.rng.normal(0, self.cfg.prev_sigma, size=2) for _ in range(self.cfg.num_prev_samples)]
        for s in [-0.6, 0.0, 0.6]:
            for a in [self.cfg.brake_limit, -0.2, 0.2, self.cfg.throttle_limit]:
                cands.append(np.array([s, a], dtype=np.float32))
        cands = [self._box_rate(c, prev) for c in cands]

        def eval_cand(a):
            speed = ego["speed"]; hdg = ego["heading"]; x = ego["x"]; y = ego["y"]
            min_d, min_ttc, min_h = np.inf, np.inf, np.inf
            approaching = False
            for _ in range(self.cfg.horizon):
                hdg += self.cfg.k_steer * float(a[0]) * self.cfg.dt
                speed = np.clip(speed + self.cfg.k_accel * float(a[1]) * self.cfg.dt, 0.0, self.cfg.v_max)
                x += speed * np.cos(hdg) * self.cfg.dt; y += speed * np.sin(hdg) * self.cfg.dt
                ego_v = np.array([speed * np.cos(hdg), speed * np.sin(hdg)])
                for ob in obstacles:
                    p_rel = np.array([ob["x"] - x, ob["y"] - y]); v_rel = ego_v - ob["v"]
                    d = np.linalg.norm(p_rel); min_d = min(min_d, d)
                    appr = float(np.dot(p_rel, v_rel)) > 0.0; approaching = approaching or appr
                    if appr:
                        ttc = d / (np.linalg.norm(v_rel) + self.cfg.eps); min_ttc = min(min_ttc, ttc)
                        h = abs(p_rel[0] * v_rel[1] - p_rel[1] * v_rel[0]) - (self.cfg.ego_radius + self.cfg.obs_radius) * np.linalg.norm(v_rel)
                        min_h = min(min_h, h)
            if len(obstacles) == 0:
                min_d, min_ttc, min_h = np.inf, np.inf, np.inf
            lane_margin = float(lane["lane_margin"])
            lane_ok = (not lane["lane_available"]) or (lane_margin >= self.cfg.lane_margin_min)
            valid = (min_d >= self.cfg.safe_radius) and ((not approaching) or (min_ttc >= self.cfg.ttc_min and min_h >= self.cfg.h_vo_margin)) and lane_ok
            return dict(a=a, valid=valid, approaching=approaching, min_d=min_d, min_ttc=min_ttc, min_h=min_h, lane_margin=lane_margin,
                        collision=float(min_d < self.cfg.safe_radius), offroad=float(lane["lane_available"] and lane_margin < self.cfg.lane_margin_min),
                        progress=float(speed), lane_center_error=float(lane.get("lane_center_error", 0.0)), margin_bonus=float(min_d + min_ttc + min_h if np.isfinite(min_ttc) else min_d))

        evals = [eval_cand(a) for a in cands]
        raw_eval = eval_cand(clipped)
        valids = [e for e in evals if e["valid"]]
        fallback = len(valids) == 0
        hard_brake = np.array([0.0, self.cfg.brake_limit], dtype=np.float32)
        if raw_eval["valid"]:
            best = raw_eval
        elif valids:
            best = min(valids, key=lambda e: self.cfg.w_raw*np.sum((e["a"]-raw)**2)+self.cfg.w_rate*np.sum((e["a"]-prev)**2)+self.cfg.w_lane*e["lane_center_error"]-self.cfg.w_progress*e["progress"]-self.cfg.w_margin*e["margin_bonus"])
        else:
            best = min(evals, key=lambda e: 1000*e["collision"]+500*e["offroad"]+50*max(0,self.cfg.ttc_min-e["min_ttc"])+20*max(0,self.cfg.h_vo_margin-e["min_h"])+10*max(0,self.cfg.safe_radius-e["min_d"])+np.sum((e["a"]-hard_brake)**2))
        exec_action = np.clip(best["a"], -1.0, 1.0).astype(np.float32)
        self.prev_exec_action = exec_action.copy()
        diff = exec_action - raw
        info = dict(raw_action=raw, exec_action=exec_action, projection_residual=float(np.linalg.norm(diff)), projection_cost=float(np.sum(diff**2)), filter_active=float(np.linalg.norm(diff)>1e-6),
                    raw_action_norm=float(np.linalg.norm(raw)), exec_action_norm=float(np.linalg.norm(exec_action)), raw_steer=float(raw[0]), exec_steer=float(exec_action[0]), raw_accel=float(raw[1]), exec_accel=float(exec_action[1]),
                    sample_filter_active=1.0, num_candidates=len(cands), num_valid_candidates=len(valids), valid_candidate_ratio=float(len(valids)/max(len(cands),1)), no_safe_candidate=float(fallback), fallback=float(fallback), least_risk_score=float(0.0 if not fallback else 1.0),
                    min_pred_dist=float(best['min_d']), min_pred_ttc=float(best['min_ttc']), min_pred_h_vo=float(best['min_h']), min_lane_margin=float(best['lane_margin']), vo_active=float(best['approaching']),
                    ttc_violation=float(best['approaching'] and best['min_ttc']<self.cfg.ttc_min), vo_violation=float(best['approaching'] and best['min_h']<self.cfg.h_vo_margin), lane_violation=float(lane['lane_available'] and best['lane_margin']<self.cfg.lane_margin_min), predicted_collision=float(best['collision']), predicted_offroad=float(best['offroad']),
                    filter_time_ms=float((time.perf_counter()-t0)*1000.0))
        return exec_action, info
