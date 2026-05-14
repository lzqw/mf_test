from dataclasses import dataclass
import numpy as np


def _to_xy(value):
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size < 2:
        return None
    return arr[:2].astype(np.float32)


def _get_attr_path(obj, path):
    cur = obj
    for name in path.split('.'):
        if cur is None or not hasattr(cur, name):
            return None
        cur = getattr(cur, name)
    return cur


def _iter_collection(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        if value.ndim == 2 and value.shape[1] >= 2:
            return [row for row in value]
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


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
    robot_radius: float = 0.20
    hazard_radius: float = 0.20
    safety_margin: float = 0.05
    point_dt: float = 0.1
    point_action_scale: float = 1.0
    car_dt: float = 0.1
    car_k_steer: float = 0.6
    car_k_accel: float = 1.0
    car_v_max: float = 2.0
    shield_horizon: int = 1
    gt_action_grid_size: int = 31
    gt_eps: float = 1e-6
    eps: float = 1e-6


def make_action_grid(grid_size, action_space):
    adim = int(np.prod(action_space.shape))
    if adim != 2:
        return np.zeros((0, adim), dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, int(grid_size), dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, int(grid_size), dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    grid = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
    return np.clip(grid, action_space.low, action_space.high).astype(np.float32)


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

    def _project_feasible(self, action, prev_exec_action, action_space, env_id):
        a = np.asarray(action, dtype=np.float32).copy()
        prev = np.asarray(prev_exec_action, dtype=np.float32).copy()
        a = np.clip(a, action_space.low, action_space.high)
        if "Car" in env_id and a.shape[0] >= 2 and prev.shape[0] >= 2:
            a[0] = np.clip(a[0], -self.cfg.steer_limit, self.cfg.steer_limit)
            a[1] = np.clip(a[1], -self.cfg.speed_limit, self.cfg.speed_limit)
            a[0] = prev[0] + np.clip(a[0] - prev[0], -self.cfg.max_dsteer, self.cfg.max_dsteer)
            a[1] = prev[1] + np.clip(a[1] - prev[1], -self.cfg.max_dspeed, self.cfg.max_dspeed)
        else:
            max_norm = self.cfg.max_action_norm_car if "Car" in env_id else self.cfg.max_action_norm_point
            nrm = float(np.linalg.norm(a))
            if nrm > max_norm:
                a = a / max(nrm, self.cfg.eps) * max_norm
            delta = np.clip(a - prev, -self.cfg.max_delta_point, self.cfg.max_delta_point)
            a = prev + delta
        return np.clip(a, action_space.low, action_space.high).astype(np.float32)

    def _compute_min_h(self, pos, hazards):
        if not hazards:
            return {"min_h": np.inf, "nearest_dist": np.inf, "nearest_hazard_pos": None, "nearest_hazard_radius": np.inf}
        dists = np.array([np.linalg.norm(pos - h["pos"]) for h in hazards], dtype=np.float32)
        i = int(np.argmin(dists))
        hs = [float(np.linalg.norm(pos - h["pos"]) - (self.cfg.robot_radius + h["radius"] + self.cfg.safety_margin)) for h in hazards]
        return {"min_h": float(np.min(hs)), "nearest_dist": float(dists[i]), "nearest_hazard_pos": hazards[i]["pos"], "nearest_hazard_radius": float(hazards[i]["radius"])}

    def _rollout_point(self, pos, action):
        p = np.asarray(pos, dtype=np.float32).copy()
        out = []
        for _ in range(int(self.cfg.shield_horizon)):
            p = p + self.cfg.point_dt * self.cfg.point_action_scale * np.asarray(action[:2], dtype=np.float32)
            out.append(p.copy())
        return out

    def _rollout_car(self, pos, heading, speed, action):
        p = np.asarray(pos, dtype=np.float32).copy()
        hd = float(heading)
        v = float(speed)
        steer, throttle = float(action[0]), float(action[1])
        out = []
        for _ in range(int(self.cfg.shield_horizon)):
            hd = hd + self.cfg.car_dt * self.cfg.car_k_steer * steer
            v = float(np.clip(v + self.cfg.car_dt * self.cfg.car_k_accel * throttle, 0.0, self.cfg.car_v_max))
            p = p + self.cfg.car_dt * v * np.array([np.cos(hd), np.sin(hd)], dtype=np.float32)
            out.append(p.copy())
        return out

    def _predict_candidate_min_h(self, ego, hazards, action, env_id):
        traj = self._rollout_car(ego["pos"], ego["heading"], ego["speed"], action) if "Car" in env_id else self._rollout_point(ego["pos"], action)
        if not traj:
            return np.inf
        return float(np.min([self._compute_min_h(p, hazards)["min_h"] for p in traj]))

    def _generate_gt_candidates(self, projected_raw, raw_action, prev_exec_action, action_space, env_id):
        cands = [projected_raw, prev_exec_action, np.zeros_like(projected_raw), 0.25 * projected_raw, 0.5 * projected_raw, 0.75 * projected_raw, -0.3 * projected_raw]
        adim = projected_raw.shape[0]
        if adim == 2:
            grid = make_action_grid(self.cfg.gt_action_grid_size, action_space)
            cands.extend([g for g in grid])
            if "Car" not in env_id:
                nrm = float(np.linalg.norm(projected_raw))
                if nrm > self.cfg.gt_eps:
                    t = np.array([-projected_raw[1], projected_raw[0]], dtype=np.float32) / nrm
                    cands += [0.4 * t, -0.4 * t]
        if "Car" in env_id and adim >= 2:
            prev_steer = float(prev_exec_action[0]) if prev_exec_action.shape[0] >= 2 else 0.0
            prev_speed = float(prev_exec_action[1]) if prev_exec_action.shape[0] >= 2 else 0.0
            cands += [np.array([projected_raw[0], min(projected_raw[1], 0.0)], np.float32), np.array([0.0, -0.5], np.float32), np.array([-0.4, -0.5], np.float32), np.array([0.4, -0.5], np.float32), np.array([-0.35, 0.2], np.float32), np.array([0.35, 0.2], np.float32), np.array([prev_steer, min(prev_speed, 0.0)], np.float32)]
        proj = [self._project_feasible(ci, prev_exec_action, action_space, env_id) for ci in cands]
        uniq = {}
        for ci in proj:
            uniq[tuple(np.round(ci, 4).tolist())] = ci
        return list(uniq.values())

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

    def _extract_ego_state_from_env(self, env):
        unknown = {"known": False, "pos": np.zeros(2, dtype=np.float32), "vel": np.zeros(2, dtype=np.float32),
                   "heading": 0.0, "speed": 0.0, "source": "unknown"}
        if env is None:
            return unknown
        u = getattr(env, "unwrapped", env)
        obj_paths = ["agent", "robot", "ego", "_agent", "_robot", "task.agent", "task.robot", "task.ego", "unwrapped.agent"]
        pos_attrs = ["pos", "position", "body_pos", "xy"]
        pos_methods = ["get_position", "get_xy_position", "get_pos"]
        vel_attrs = ["vel", "velocity", "linear_velocity"]
        vel_methods = ["get_velocity", "get_linear_velocity"]
        heading_attrs = ["heading", "heading_theta", "angle", "theta"]

        for p in obj_paths:
            obj = _get_attr_path(u, p)
            if obj is None:
                continue
            pos = None
            src = p
            for a in pos_attrs:
                pos = _to_xy(getattr(obj, a, None))
                if pos is not None:
                    src = f"{p}.{a}"
                    break
            if pos is None:
                for m in pos_methods:
                    fn = getattr(obj, m, None)
                    if callable(fn):
                        pos = _to_xy(fn())
                        if pos is not None:
                            src = f"{p}.{m}()"
                            break
            if pos is None:
                continue

            vel = None
            for a in vel_attrs:
                vel = _to_xy(getattr(obj, a, None))
                if vel is not None:
                    break
            if vel is None:
                for m in vel_methods:
                    fn = getattr(obj, m, None)
                    if callable(fn):
                        vel = _to_xy(fn())
                        if vel is not None:
                            break
            if vel is None:
                vel = np.zeros(2, dtype=np.float32)

            heading = 0.0
            for a in heading_attrs:
                v = getattr(obj, a, None)
                if v is not None:
                    try:
                        heading = float(v)
                        break
                    except Exception:
                        pass
            fn = getattr(obj, "get_heading", None)
            if heading == 0.0 and callable(fn):
                try:
                    heading = float(fn())
                except Exception:
                    pass

            speed = None
            if hasattr(obj, "speed"):
                try:
                    speed = float(getattr(obj, "speed"))
                except Exception:
                    speed = None
            if speed is None:
                speed = float(np.linalg.norm(vel))
            return {"known": True, "pos": pos, "vel": vel, "heading": heading, "speed": speed, "source": src}

        for p in ["task.agent_pos", "task.robot_pos", "task.robot_position", "agent_pos", "robot_pos"]:
            pos = _to_xy(_get_attr_path(u, p))
            if pos is not None:
                return {"known": True, "pos": pos, "vel": np.zeros(2, dtype=np.float32), "heading": 0.0, "speed": 0.0, "source": p}
        return unknown

    def _extract_hazards_from_env(self, env):
        if env is None:
            return []
        u = getattr(env, "unwrapped", env)
        default_radius = float(getattr(self.cfg, "hazard_radius", self.cfg.hazard_stop))
        hazards = []
        cands = ["task.hazards", "task._hazards", "task.hazard", "task.hazards_pos", "task.hazards_position",
                 "task.hazards_locations", "task.hazards_locations_list", "hazards", "_hazards", "world.hazards",
                 "world._hazards", "world.geoms", "world.objects", "engine.world.hazards"]
        scalar_radius_paths = ["task.hazards_size", "task.hazards_radius", "task.hazards_keepout"]
        scalar_radius = None
        for rp in scalar_radius_paths:
            rv = _get_attr_path(u, rp)
            try:
                arr = np.asarray(rv, dtype=np.float32).reshape(-1)
                if arr.size > 0:
                    scalar_radius = float(arr[0])
                    break
            except Exception:
                pass

        def is_hazard_like(item, src):
            t = str(type(item)).lower() + ' ' + src.lower() + ' ' + str(getattr(item, 'name', '')).lower()
            return any(k in t for k in ["hazard", "obstacle", "pillar", "circle"]) and not any(k in t for k in ["goal", "object", "box", "button"])

        for src in cands:
            coll = _get_attr_path(u, src)
            if coll is None:
                continue
            for h in _iter_collection(coll):
                pos = None
                if isinstance(h, (list, tuple, np.ndarray)):
                    pos = _to_xy(h)
                    if pos is not None and ("hazards_pos" in src or "hazards_location" in src):
                        hazards.append({"pos": pos, "radius": scalar_radius or default_radius, "source": src})
                        continue
                if pos is None and not is_hazard_like(h, src) and not ("hazard" in src.lower()):
                    continue
                for a in ["pos", "position", "body_pos", "xy", "center"]:
                    pos = _to_xy(getattr(h, a, None))
                    if pos is not None:
                        break
                if pos is None:
                    for m in ["get_position", "get_xy_position", "get_pos"]:
                        fn = getattr(h, m, None)
                        if callable(fn):
                            pos = _to_xy(fn())
                            if pos is not None:
                                break
                if pos is None:
                    continue
                r = None
                for ra in ["radius", "size", "keepout", "keepout_radius", "geom_radius"]:
                    try:
                        v = getattr(h, ra, None)
                        if v is not None:
                            r = float(np.asarray(v).reshape(-1)[0])
                            break
                    except Exception:
                        pass
                if r is None:
                    r = scalar_radius or default_radius
                hazards.append({"pos": pos, "radius": float(r), "source": src})
        return hazards

    def _extract_objects_from_env(self, env):
        if env is None:
            return []
        u = getattr(env, "unwrapped", env)
        objs = []
        for src in ["task.objects", "task.object", "task.box", "task.boxes", "task.push_object", "task.obj", "objects", "world.objects"]:
            coll = _get_attr_path(u, src)
            if coll is None:
                continue
            for o in _iter_collection(coll):
                pos = _to_xy(o if isinstance(o, (list, tuple, np.ndarray)) else None)
                if pos is None:
                    for a in ["pos", "position", "body_pos", "xy", "center"]:
                        pos = _to_xy(getattr(o, a, None))
                        if pos is not None:
                            break
                if pos is None:
                    for m in ["get_position", "get_xy_position", "get_pos"]:
                        fn = getattr(o, m, None)
                        if callable(fn):
                            pos = _to_xy(fn())
                            if pos is not None:
                                break
                if pos is None:
                    continue
                radius = 0.0
                for ra in ["radius", "size", "keepout", "geom_radius"]:
                    try:
                        v = getattr(o, ra, None)
                        if v is not None:
                            radius = float(np.asarray(v).reshape(-1)[0])
                            break
                    except Exception:
                        pass
                objs.append({"pos": pos, "radius": radius, "source": src})
        return objs

    def project(self, raw_action, obs, info, prev_exec_action, action_space, env_id, env=None):
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
        if self.filter_type in ["sample_shield", "gt_shield"]:
            projected = a.copy()
            ego = self._extract_ego_state_from_env(env)
            hazards = self._extract_hazards_from_env(env)
            objects = self._extract_objects_from_env(env)
            selected_candidate_type = "none"
            predicted_min_h = np.nan
            current_min_h = np.nan
            nearest_hazard_dist = np.nan
            gt_known = 0.0
            if self.filter_type == "gt_shield":
                projected_raw = self._project_feasible(a0, prev, action_space, env_id)
                if (not ego["known"]) or (len(hazards) == 0):
                    a = projected_raw
                    num_candidates = num_safe_candidates = 1
                    safe_candidate_ratio = 1.0
                    emergency_active = 0.0
                    selected_candidate_type = "projected_raw_no_gt"
                else:
                    gt_known = 1.0
                    cur = self._compute_min_h(ego["pos"], hazards)
                    current_min_h = float(cur["min_h"])
                    nearest_hazard_dist = float(cur["nearest_dist"])
                    cands = self._generate_gt_candidates(projected_raw, a0, prev, action_space, env_id)
                    scores = [self._predict_candidate_min_h(ego, hazards, ci, env_id) for ci in cands]
                    num_candidates = len(cands)
                    safe_idx = [i for i, s in enumerate(scores) if s >= 0.0]
                    num_safe_candidates = len(safe_idx)
                    safe_candidate_ratio = float(num_safe_candidates / max(1, num_candidates))
                    if safe_idx:
                        costs = [float(np.sum((cands[i] - a0) ** 2) + 0.1 * np.sum((cands[i] - prev) ** 2)) for i in safe_idx]
                        bi = safe_idx[int(np.argmin(costs))]
                        a = cands[bi]; predicted_min_h = float(scores[bi]); selected_candidate_type = "safe_candidate"
                    else:
                        emergency_active = 1.0
                        bi = int(np.argmax(scores))
                        best = cands[bi]; best_h = float(scores[bi]); selected_candidate_type = "emergency_best_pred"
                        em = []
                        if current_min_h < 0.0 or best_h < 0.0:
                            if "Car" in env_id and cur["nearest_hazard_pos"] is not None:
                                rel = cur["nearest_hazard_pos"] - ego["pos"]
                                left = np.array([-np.sin(ego["heading"]), np.cos(ego["heading"])], dtype=np.float32)
                                side = np.sign(float(np.dot(rel, left)))
                                em.append(np.array([-0.5 * side, -0.6], dtype=np.float32))
                            elif cur["nearest_hazard_pos"] is not None:
                                d = ego["pos"] - cur["nearest_hazard_pos"]; n = np.linalg.norm(d)
                                if n > self.cfg.gt_eps:
                                    em.append(d / n)
                        for e in em:
                            ce = self._project_feasible(e, prev, action_space, env_id)
                            sh = self._predict_candidate_min_h(ego, hazards, ce, env_id)
                            if sh > best_h:
                                best, best_h, selected_candidate_type = ce, float(sh), "emergency_backup"
                        a = best; predicted_min_h = best_h

            if hazards and ego["known"]:
                dists = [float(np.linalg.norm(h["pos"] - ego["pos"]) - h["radius"]) for h in hazards]
                nearest = float(np.min(dists))
                danger = dict(global_min_dist=nearest, front_min_dist=nearest, left_min_dist=nearest, right_min_dist=nearest)
            else:
                danger = dict(global_min_dist=np.nan, front_min_dist=np.nan, left_min_dist=np.nan, right_min_dist=np.nan)
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
            "gt_known": gt_known if self.filter_type == "gt_shield" else (float(ego["known"]) if self.filter_type == "sample_shield" else 0.0),
            "ego_x": float(ego["pos"][0]) if self.filter_type in ["sample_shield", "gt_shield"] and ego["known"] else np.nan,
            "ego_y": float(ego["pos"][1]) if self.filter_type in ["sample_shield", "gt_shield"] and ego["known"] else np.nan,
            "num_hazards": float(len(hazards)) if self.filter_type in ["sample_shield", "gt_shield"] else 0.0,
            "nearest_hazard_dist": nearest_hazard_dist if self.filter_type == "gt_shield" else (danger.get("global_min_dist", np.nan) if self.filter_type == "sample_shield" else np.nan),
            "predicted_min_h": predicted_min_h if self.filter_type == "gt_shield" else global_min_h,
            "current_min_h": current_min_h if self.filter_type == "gt_shield" else np.nan,
            "selected_candidate_type": selected_candidate_type if self.filter_type == "gt_shield" else ("safe_candidate" if self.filter_type == "sample_shield" and num_safe_candidates > 0 else ("emergency" if self.filter_type == "sample_shield" else "none")),
            "num_objects": float(len(objects)) if self.filter_type in ["sample_shield", "gt_shield"] else 0.0,
            "cbf_violation": float(current_min_h < 0.0) if self.filter_type == "gt_shield" and np.isfinite(current_min_h) else cbf_violation,
            "min_h": current_min_h if self.filter_type == "gt_shield" else min_h,
            "filter_active_005": float(residual > 0.05),
            "filter_active_010": float(residual > 0.10),
        }
        return a, filter_info
