from dataclasses import dataclass
import inspect
import time
import numpy as np


@dataclass
class CBFBuiltinFilterConfig:
    dt: float = 0.1
    horizon: int = 10
    wheelbase: float = 2.5
    obstacle_radius: float = 1.35
    safe_distance: float = 1.35
    steer_limit: float = 0.7
    throttle_limit: float = 0.8
    brake_limit: float = -0.8
    max_steer_angle: float = 0.25
    max_dsteer: float = 0.20
    max_daccel: float = 0.30
    cbf_activation_distance: float = 20.0
    ttc_activation_threshold: float = 5.0
    min_closing_speed: float = 0.05
    enable_ttc_lateral_gate: bool = True
    ttc_lateral_gate: float = 1.8
    ttc_front_min_longitudinal: float = 0.0
    ttc_front_max_longitudinal: float = 30.0
    cbf_h_margin: float = 0.0
    cbf_min_margin: float = 0.0
    enable_cbf_lateral_gate: bool = True
    cbf_lateral_gate: float = 2.0
    cbf_front_min_longitudinal: float = -1.0
    cbf_front_max_longitudinal: float = 30.0
    cbf_close_distance_margin: float = 0.3
    require_approaching: bool = True
    builtin_policy_name: str = "idm"
    builtin_action_scale: float = 1.0
    blend_with_raw: float = 0.0
    correction_min_throttle: float = 0.0
    preserve_raw_accel_if_positive: bool = False
    enable_rate_limit: bool = True
    debug: bool = False
    fallback_pass_side: float = -1.0
    fallback_avoid_distance: float = 18.0
    fallback_max_steer: float = 0.45
    fallback_min_steer: float = 0.12
    fallback_target_speed: float = 3.0
    fallback_speed_kp: float = 0.25
    fallback_obstacle_lateral_gate: float = 4.5
    fallback_obstacle_front_min: float = -1.0
    fallback_obstacle_front_max: float = 20.0
    enable_road_edge_guard: bool = True
    road_edge_max_abs_lateral: float = 5.0
    road_edge_margin: float = 0.35
    road_edge_soft_margin: float = 0.80
    road_edge_cost_weight: float = 10.0
    road_edge_use_start_lane: bool = True
    road_edge_lane_width_fallback: float = 3.7
    road_edge_num_allowed_lane_changes: int = 1
    road_edge_block_raw: bool = True


class MetaDriveBuiltinPolicyAdapter:
    def __init__(self, policy_name: str = "idm", cfg: CBFBuiltinFilterConfig = None):
        self.policy_name = policy_name
        self.policy = None
        self.policy_status = "not_initialized"
        self.builtin_policy_class = None
        self.constructor_signature = None
        self.build_exception = None
        self.cfg = cfg or CBFBuiltinFilterConfig()
        self.last_info = {}

    def reset(self):
        self.policy = None
        self.policy_status = "reset"
        self.builtin_policy_class = None
        self.constructor_signature = None
        self.build_exception = None
        self.last_info = {}

    def _get_vehicle(self, env):
        u = getattr(env, "unwrapped", env)
        return getattr(u, "vehicle", None) or getattr(u, "agent", None)

    def _get_agent_id(self, env):
        u = getattr(env, "unwrapped", env)
        agent = getattr(u, "agent", None)
        return getattr(agent, "id", None) or getattr(u, "agent_id", None) or "default_agent"

    def _build_policy(self, env):
        self.build_exception = None
        if self.policy_name in ["fallback_lane", "fallback_obstacle_avoid"]:
            self.policy_status = "fallback_lane_following"
            self.builtin_policy_class = self.policy_name
            self.constructor_signature = "fallback"
            return
        modules = {
            "idm": ("metadrive.policy.idm_policy", "IDMPolicy"),
            "lane_change": ("metadrive.policy.lane_change_policy", "LaneChangePolicy"),
            "trajectory_idm": ("metadrive.policy.trajectory_idm_policy", "TrajectoryIDMPolicy"),
            "expert": ("metadrive.policy.expert_policy", "ExpertPolicy"),
        }
        mod_name, cls_name = modules.get(self.policy_name, modules["idm"])
        try:
            mod = __import__(mod_name, fromlist=[cls_name, "PPOExpertPolicy"])
            cls = getattr(mod, cls_name, None) or getattr(mod, "PPOExpertPolicy", None)
            if cls is None:
                raise RuntimeError("policy class missing")
            self.builtin_policy_class = getattr(cls, '__name__', str(cls))
            sig = inspect.signature(cls)
            self.constructor_signature = str(sig)
            veh = self._get_vehicle(env)
            kwargs = {}
            if "control_object" in sig.parameters:
                kwargs["control_object"] = veh
            elif "vehicle" in sig.parameters:
                kwargs["vehicle"] = veh
            elif "obj" in sig.parameters:
                kwargs["obj"] = veh
            try:
                self.policy = cls(**kwargs) if kwargs else cls(veh)
            except Exception:
                self.policy = cls()
            self.policy_status = "builtin_constructed"
        except Exception as e:
            self.policy = None
            self.build_exception = repr(e)
            self.policy_status = "build_failed"

    def _fallback_lane_follow(self, env):
        veh = self._get_vehicle(env)
        steer, throttle = 0.0, 0.25
        try:
            lane = getattr(veh, "lane", None) or getattr(getattr(veh, "navigation", None), "current_lane", None)
            pos = np.asarray(getattr(veh, "position", [0.0, 0.0]), dtype=np.float32)
            yaw = float(getattr(veh, "heading_theta", 0.0))
            speed = float(getattr(veh, "speed", 0.0))
            if lane is not None and hasattr(lane, "local_coordinates") and hasattr(lane, "position"):
                lon, lat = lane.local_coordinates(pos)
                target = lane.position(lon + 6.0, 0.0)
                desired = np.arctan2(target[1] - pos[1], target[0] - pos[0])
                he = np.arctan2(np.sin(desired - yaw), np.cos(desired - yaw))
                steer = np.clip(0.9 * he - 0.25 * lat, -0.7, 0.7)
            throttle = np.clip(0.35 * (4.0 - speed), -0.2, 0.45)
        except Exception:
            pass
        return np.array([steer, throttle], dtype=np.float32)

    def _fallback_obstacle_avoid(self, env):
        lane_action = self._fallback_lane_follow(env)
        veh = self._get_vehicle(env)
        ego = np.asarray(getattr(veh, "position", [0.0, 0.0]), dtype=np.float32)
        yaw = float(getattr(veh, "heading_theta", 0.0))
        speed = float(getattr(veh, "speed", 0.0))
        forward = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32)
        left = np.array([-np.sin(yaw), np.cos(yaw)], dtype=np.float32)
        u = getattr(env, "unwrapped", env)
        tm = getattr(u, "traffic_manager", None)
        objs = list(getattr(tm, "vehicles", {}).values()) if tm is not None and hasattr(tm, 'vehicles') else []
        pk = getattr(u, "_parked_obj", None)
        if pk is not None:
            objs.append(pk)
        nearest = None
        best_lon = np.inf
        best_lat = np.nan
        for o in objs:
            if o is None or o is veh:
                continue
            p = np.asarray(getattr(o, 'position', [0.0, 0.0]), dtype=np.float32)
            rel = p - ego
            lon = float(np.dot(rel, forward))
            lat = float(np.dot(rel, left))
            if lon < self.cfg.fallback_obstacle_front_min or lon > self.cfg.fallback_obstacle_front_max or abs(lat) > self.cfg.fallback_obstacle_lateral_gate:
                continue
            if lon < best_lon:
                best_lon, best_lat, nearest = lon, lat, p
        if nearest is None:
            self.last_info = dict(fallback_obstacle_seen=0.0, fallback_obstacle_longitudinal=np.nan, fallback_obstacle_lateral=np.nan, fallback_avoid_bias=0.0, fallback_pass_side=float(self.cfg.fallback_pass_side), fallback_status="fallback_lane_following")
            return lane_action, "fallback_lane_following"
        strength = float(np.clip((self.cfg.fallback_avoid_distance - best_lon) / max(self.cfg.fallback_avoid_distance, 1e-6), 0.0, 1.0))
        avoid_bias = float(self.cfg.fallback_pass_side * (self.cfg.fallback_min_steer + strength * (self.cfg.fallback_max_steer - self.cfg.fallback_min_steer)))
        steer = float(np.clip(lane_action[0] + avoid_bias, -0.7, 0.7))
        throttle = float(np.clip(self.cfg.fallback_speed_kp * (self.cfg.fallback_target_speed - speed), -0.3, 0.45))
        self.last_info = dict(fallback_obstacle_seen=1.0, fallback_obstacle_longitudinal=float(best_lon), fallback_obstacle_lateral=float(best_lat), fallback_avoid_bias=avoid_bias, fallback_pass_side=float(self.cfg.fallback_pass_side), fallback_status="fallback_obstacle_avoid")
        return np.array([steer, throttle], dtype=np.float32), "fallback_obstacle_avoid"

    def act(self, env):
        if self.policy is None and self.policy_name not in ["fallback_lane", "fallback_obstacle_avoid"]:
            self._build_policy(env)
        builtin_exception = None
        if self.policy is not None:
            aid = self._get_agent_id(env)
            vehicle = self._get_vehicle(env)
            for args in [(aid,), (), (env,), (vehicle,)]:
                try:
                    act = self.policy.act(*args)
                    self.policy_status = "builtin_constructed_and_called"
                    return np.clip(np.asarray(act, dtype=np.float32).reshape(2), -1.0, 1.0), 1.0, self.policy_status, None
                except TypeError:
                    continue
                except Exception as e:
                    builtin_exception = repr(e)
            self.policy_status = "builtin_constructed_but_act_failed"
        if self.policy_name == "fallback_obstacle_avoid":
            fb_action, fb_status = self._fallback_obstacle_avoid(env)
            return np.clip(fb_action, -1.0, 1.0), 0.0, fb_status, builtin_exception or self.build_exception
        if self.policy_status == "build_failed":
            return np.clip(self._fallback_lane_follow(env), -1.0, 1.0), 0.0, "build_failed", builtin_exception or self.build_exception
        return np.clip(self._fallback_lane_follow(env), -1.0, 1.0), 0.0, (self.policy_status or "fallback_lane_following"), builtin_exception or self.build_exception


class CBFBuiltinSafetyFilter:
    def __init__(self, cfg: CBFBuiltinFilterConfig):
        self.cfg = cfg
        self.prev_exec_action = np.zeros(2, dtype=np.float32)
        self.adapter = MetaDriveBuiltinPolicyAdapter(cfg.builtin_policy_name, cfg=cfg)
        self.road_ref_lane = None
        self.road_ref_pos = None
        self.road_ref_yaw = None
        self.road_ref_known = False

    def reset(self):
        self.prev_exec_action[:] = 0.0
        self.adapter.reset()
        self.road_ref_lane = None
        self.road_ref_pos = None
        self.road_ref_yaw = None
        self.road_ref_known = False

    def _box_rate(self, action, prev):
        a = np.asarray(action, dtype=np.float32).copy()
        a[0] = np.clip(a[0], -self.cfg.steer_limit, self.cfg.steer_limit)
        a[1] = np.clip(a[1], self.cfg.brake_limit, self.cfg.throttle_limit)
        d = np.clip(a - prev, [-self.cfg.max_dsteer, -self.cfg.max_daccel], [self.cfg.max_dsteer, self.cfg.max_daccel])
        return np.clip(prev + d, -1.0, 1.0)

    def _extract_ego_state(self, env):
        u = getattr(env, "unwrapped", env)
        v = getattr(u, "vehicle", None) or getattr(u, "agent", None)
        pos = np.asarray(getattr(v, "position", [0.0, 0.0]), dtype=np.float32)
        yaw = float(getattr(v, "heading_theta", getattr(v, "heading", 0.0)))
        spd = float(getattr(v, "speed", np.linalg.norm(getattr(v, "velocity", [0.0, 0.0]))))
        return np.array([pos[0], pos[1], yaw, spd], dtype=np.float32), v

    def _extract_obstacles(self, env, ego_obj, ego_xy):
        obs = []
        u = getattr(env, "unwrapped", env)
        cands = []
        pk = getattr(u, "_parked_obj", None)
        if pk is not None:
            cands.append(pk)
        tm = getattr(u, "traffic_manager", None)
        if tm is not None and hasattr(tm, "vehicles"):
            cands += list(getattr(tm, "vehicles", {}).values())
        for o in cands:
            if o is None or o is ego_obj:
                continue
            pos = np.asarray(getattr(o, "position", [0.0, 0.0]), dtype=np.float32)
            yaw = float(getattr(o, "heading_theta", getattr(o, "heading", 0.0)))
            speed = float(getattr(o, "speed", np.linalg.norm(getattr(o, "velocity", [0.0, 0.0]))))
            d = float(np.linalg.norm(pos - ego_xy))
            obs.append((d, np.array([pos[0], pos[1], yaw, speed], dtype=np.float32)))
        if not obs:
            return None, np.inf
        obs.sort(key=lambda t: t[0])
        return obs[0][1], obs[0][0]

    def evaluate_raw_cbf(self, raw_action, ego, obstacle):
        H, dt = self.cfg.horizon, self.cfg.dt
        R = self.cfg.obstacle_radius + self.cfg.safe_distance
        x, y, yaw, v = map(float, ego)
        steer = float(np.clip(raw_action[0] / max(self.cfg.steer_limit, 1e-6) * self.cfg.max_steer_angle, -self.cfg.max_steer_angle, self.cfg.max_steer_angle))
        accel = float(raw_action[1])
        ox, oy, oyaw, ov = map(float, obstacle)
        vox, voy = ov * np.cos(oyaw), ov * np.sin(oyaw)
        ex0, ey0 = ox - x, oy - y
        forward = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32)
        left = np.array([-np.sin(yaw), np.cos(yaw)], dtype=np.float32)
        obstacle_longitudinal = ex0 * forward[0] + ey0 * forward[1]
        obstacle_lateral = ex0 * left[0] + ey0 * left[1]
        in_ttc_corridor = (obstacle_longitudinal >= self.cfg.ttc_front_min_longitudinal and obstacle_longitudinal <= self.cfg.ttc_front_max_longitudinal and abs(obstacle_lateral) <= self.cfg.ttc_lateral_gate)
        if not self.cfg.enable_ttc_lateral_gate:
            in_ttc_corridor = True
        in_cbf_corridor = (obstacle_longitudinal >= self.cfg.cbf_front_min_longitudinal and obstacle_longitudinal <= self.cfg.cbf_front_max_longitudinal and abs(obstacle_lateral) <= self.cfg.cbf_lateral_gate)
        if not self.cfg.enable_cbf_lateral_gate:
            in_cbf_corridor = True
        evx0, evy0 = v * np.cos(yaw) - vox, v * np.sin(yaw) - voy
        sign_s = 1.0 if ex0 * evy0 - ey0 * evx0 > 0 else -1.0
        min_d = min_ttc = h_min = cbf_min = np.inf
        pred_col = 0.0
        hk_prev = None
        closing_speed = 0.0
        for k in range(H + 1):
            ex, ey = ox - x, oy - y
            evx, evy = v * np.cos(yaw) - vox, v * np.sin(yaw) - voy
            dist = np.sqrt(ex * ex + ey * ey)
            dot = ex * evx + ey * evy
            min_d = min(min_d, dist)
            closing_speed = max(closing_speed, max(dot / max(dist, 1e-6), 0.0))
            if closing_speed > 1e-6:
                min_ttc = min(min_ttc, max(dist - R, 0.0) / max(closing_speed, 1e-6))
            h = sign_s * (ex * evy - ey * evx) - R * np.sqrt(evx * evx + evy * evy)
            h_min = min(h_min, h)
            if hk_prev is not None:
                cbf_min = min(cbf_min, h - hk_prev + 0.5 * hk_prev)
            hk_prev = h
            if dist < R:
                pred_col = 1.0
            if k < H:
                yaw += dt * v * np.tan(steer) / max(self.cfg.wheelbase, 1e-6)
                v = max(0.0, v + 3.0 * accel * dt)
                x += dt * v * np.cos(yaw)
                y += dt * v * np.sin(yaw)
        cbf_min = float(cbf_min if np.isfinite(cbf_min) else h_min)
        close_distance_relevant = float(min_d <= (R + self.cfg.cbf_close_distance_margin))
        cbf_relevant = float(in_cbf_corridor or close_distance_relevant)
        return dict(min_dist=float(min_d), min_ttc=float(min_ttc), h_min=float(h_min), cbf_min=cbf_min, predicted_collision=pred_col, cbf_violation=float((h_min < self.cfg.cbf_h_margin) or (cbf_min < self.cfg.cbf_min_margin)), closing_speed=float(closing_speed), obstacle_longitudinal=float(obstacle_longitudinal), obstacle_lateral=float(obstacle_lateral), in_ttc_corridor=float(in_ttc_corridor), ttc_lateral_gate=float(self.cfg.ttc_lateral_gate), ttc_front_min_longitudinal=float(self.cfg.ttc_front_min_longitudinal), ttc_front_max_longitudinal=float(self.cfg.ttc_front_max_longitudinal), in_cbf_corridor=float(in_cbf_corridor), cbf_lateral_gate=float(self.cfg.cbf_lateral_gate), close_distance_relevant=close_distance_relevant, cbf_relevant=cbf_relevant)


    def _ensure_road_reference(self, env, ego_obj):
        if self.road_ref_known:
            return
        veh = ego_obj
        lane = getattr(veh, "lane", None) or getattr(getattr(veh, "navigation", None), "current_lane", None)
        pos = np.asarray(getattr(veh, "position", [0, 0]), dtype=np.float32)
        yaw = float(getattr(veh, "heading_theta", 0.0))
        self.road_ref_lane = lane
        self.road_ref_pos = pos
        self.road_ref_yaw = yaw
        self.road_ref_known = True

    def _road_lateral_from_reference(self, xy):
        xy = np.asarray(xy, dtype=np.float32)
        if self.cfg.road_edge_use_start_lane and self.road_ref_lane is not None and hasattr(self.road_ref_lane, "local_coordinates"):
            try:
                _, lat = self.road_ref_lane.local_coordinates(xy)
                return float(lat), 1.0
            except Exception:
                pass

        if self.road_ref_pos is not None:
            left = np.array([-np.sin(self.road_ref_yaw), np.cos(self.road_ref_yaw)], dtype=np.float32)
            lat = float(np.dot(xy - self.road_ref_pos, left))
            return lat, 0.0

        return 0.0, 0.0

    def evaluate_raw_road_edge(self, raw_action, ego):
        H, dt = self.cfg.horizon, self.cfg.dt
        x, y, yaw, v = map(float, ego)
        steer = float(np.clip(raw_action[0] / max(self.cfg.steer_limit, 1e-6) * self.cfg.max_steer_angle, -self.cfg.max_steer_angle, self.cfg.max_steer_angle))
        accel = float(raw_action[1])
        min_margin = np.inf
        max_abs_lateral = 0.0
        for k in range(H + 1):
            lat, _ = self._road_lateral_from_reference(np.array([x, y], dtype=np.float32))
            abs_lat = abs(lat)
            max_abs_lateral = max(max_abs_lateral, abs_lat)
            margin = self.cfg.road_edge_max_abs_lateral - abs_lat
            min_margin = min(min_margin, margin)
            if k < H:
                yaw += dt * v * np.tan(steer) / max(self.cfg.wheelbase, 1e-6)
                v = max(0.0, v + 3.0 * accel * dt)
                x += dt * v * np.cos(yaw)
                y += dt * v * np.sin(yaw)

        hard_violation = float(min_margin < self.cfg.road_edge_margin)
        soft_violation = max(self.cfg.road_edge_soft_margin - min_margin, 0.0)
        road_edge_projection_cost = self.cfg.road_edge_cost_weight * float(soft_violation ** 2)
        return dict(min_road_edge_margin=float(min_margin), max_abs_lateral_from_road_ref=float(max_abs_lateral), predicted_offroad=float(hard_violation), road_edge_violation=float(hard_violation), road_edge_projection_cost=float(road_edge_projection_cost), road_edge_ref_known=float(self.road_ref_known), road_edge_max_abs_lateral=float(self.cfg.road_edge_max_abs_lateral), road_edge_margin=float(self.cfg.road_edge_margin), road_edge_soft_margin=float(self.cfg.road_edge_soft_margin))

    def project(self, raw_action, env=None, prev_exec_action=None):
        t0 = time.perf_counter()
        raw = np.asarray(raw_action, dtype=np.float32).reshape(2)
        prev = self.prev_exec_action if prev_exec_action is None else np.asarray(prev_exec_action, dtype=np.float32)
        ego, ego_obj = self._extract_ego_state(env)
        self._ensure_road_reference(env, ego_obj)
        obs, obs_d = self._extract_obstacles(env, ego_obj, ego[:2])
        safe, cbf_active = True, False
        eval_info = dict(min_dist=np.inf, min_ttc=np.inf, h_min=np.inf, cbf_min=np.inf, cbf_violation=0.0, predicted_collision=0.0, closing_speed=0.0, obstacle_longitudinal=np.nan, obstacle_lateral=np.nan, in_ttc_corridor=1.0, ttc_lateral_gate=float(self.cfg.ttc_lateral_gate), in_cbf_corridor=1.0, cbf_lateral_gate=float(self.cfg.cbf_lateral_gate), close_distance_relevant=0.0, cbf_relevant=1.0)
        if self.cfg.enable_road_edge_guard:
            road_info = self.evaluate_raw_road_edge(raw, ego)
        else:
            road_info = dict(min_road_edge_margin=float("inf"), max_abs_lateral_from_road_ref=0.0, predicted_offroad=0.0, road_edge_violation=0.0, road_edge_projection_cost=0.0, road_edge_ref_known=float(self.road_ref_known), road_edge_max_abs_lateral=float(self.cfg.road_edge_max_abs_lateral), road_edge_margin=float(self.cfg.road_edge_margin), road_edge_soft_margin=float(self.cfg.road_edge_soft_margin))
        obstacle_safe = True
        if obs is not None:
            eval_info = self.evaluate_raw_cbf(raw, ego, obs)
            cbf_active = (obs_d <= self.cfg.cbf_activation_distance)
            approaching = eval_info["closing_speed"] > self.cfg.min_closing_speed
            ttc_relevant = bool(eval_info.get("in_ttc_corridor", 1.0))
            if not ttc_relevant:
                ttc_ok = True
            else:
                ttc_ok = (eval_info["min_ttc"] > self.cfg.ttc_activation_threshold) or (not self.cfg.require_approaching) or (not approaching)
            cbf_relevant = bool(eval_info.get("cbf_relevant", 1.0))
            if not cbf_active:
                cbf_ok = True
            elif not cbf_relevant:
                cbf_ok = True
            else:
                cbf_ok = (eval_info["cbf_violation"] == 0.0)
            obstacle_safe = (eval_info["predicted_collision"] == 0.0 and eval_info["min_dist"] > (self.cfg.obstacle_radius + self.cfg.safe_distance) and cbf_ok and ttc_ok)
        road_edge_ok = True
        if self.cfg.enable_road_edge_guard and self.cfg.road_edge_block_raw:
            road_edge_ok = (road_info["road_edge_violation"] == 0.0)
        safe = obstacle_safe and road_edge_ok
        builtin_action, success, status, builtin_exception = self.adapter.act(env)
        builtin_action = np.clip(self.cfg.builtin_action_scale * builtin_action, -1.0, 1.0)
        if self.cfg.preserve_raw_accel_if_positive and raw[1] > 0:
            builtin_action[1] = max(builtin_action[1], raw[1])
        builtin_action[1] = max(builtin_action[1], self.cfg.correction_min_throttle)
        if self.cfg.blend_with_raw > 0:
            builtin_action = (1 - self.cfg.blend_with_raw) * builtin_action + self.cfg.blend_with_raw * raw
        exec_action = raw if safe else builtin_action
        if self.cfg.enable_rate_limit:
            exec_action = self._box_rate(exec_action, prev)
        self.prev_exec_action = np.asarray(exec_action, dtype=np.float32).copy()
        diff = exec_action - raw
        base_projection_cost = float(np.sum(diff ** 2))
        projection_cost = float(base_projection_cost + road_info["road_edge_projection_cost"])
        info = dict(
            cbf_safe=float(safe), cbf_active=float(cbf_active), cbf_violation=float(eval_info["cbf_violation"]), predicted_collision=float(eval_info["predicted_collision"]),
            min_pred_dist=float(eval_info["min_dist"]), min_pred_ttc=float(eval_info["min_ttc"]), min_pred_h_cbf=float(eval_info["h_min"]), min_pred_cbf=float(eval_info["cbf_min"]),
            obstacle_distance=float(obs_d), closing_speed=float(eval_info["closing_speed"]), ttc_relevant=float(bool(eval_info.get("in_ttc_corridor", 1.0))), in_ttc_corridor=float(eval_info.get("in_ttc_corridor", 1.0)), obstacle_longitudinal=float(eval_info.get("obstacle_longitudinal", np.nan)), obstacle_lateral=float(eval_info.get("obstacle_lateral", np.nan)), ttc_lateral_gate=float(eval_info.get("ttc_lateral_gate", self.cfg.ttc_lateral_gate)), cbf_relevant=float(eval_info.get("cbf_relevant", 1.0)), in_cbf_corridor=float(eval_info.get("in_cbf_corridor", 1.0)), cbf_lateral_gate=float(eval_info.get("cbf_lateral_gate", self.cfg.cbf_lateral_gate)), close_distance_relevant=float(eval_info.get("close_distance_relevant", 0.0)), raw_action=raw.copy(), builtin_action=np.asarray(builtin_action, dtype=np.float32).copy(), exec_action=np.asarray(exec_action, dtype=np.float32).copy(),
            projection_residual=float(np.linalg.norm(diff)), projection_cost=projection_cost, base_projection_cost=base_projection_cost, min_road_edge_margin=float(road_info["min_road_edge_margin"]), max_abs_lateral_from_road_ref=float(road_info["max_abs_lateral_from_road_ref"]), predicted_offroad=float(road_info["predicted_offroad"]), road_edge_violation=float(road_info["road_edge_violation"]), road_edge_projection_cost=float(road_info["road_edge_projection_cost"]), road_edge_ref_known=float(road_info["road_edge_ref_known"]), road_edge_max_abs_lateral=float(road_info["road_edge_max_abs_lateral"]), road_edge_margin=float(road_info["road_edge_margin"]), road_edge_soft_margin=float(road_info["road_edge_soft_margin"]), filter_active=float(not safe), selected_candidate_type=("raw_safe" if safe else "builtin_correction"), builtin_policy_name=self.cfg.builtin_policy_name,
            builtin_policy_success=float(success), builtin_policy_status=status, builtin_policy_class=self.adapter.builtin_policy_class, builtin_policy_exception=builtin_exception or self.adapter.build_exception, fallback=float(success < 0.5), filter_time_ms=float((time.perf_counter() - t0) * 1000.0)
        )
        info.update(getattr(self.adapter, "last_info", {}) or {})
        return exec_action, info
