from dataclasses import dataclass
import time
import numpy as np

try:
    import casadi as ca
except Exception:
    ca = None


@dataclass
class MPCVehicleCBFConfig:
    wheelbase: float = 2.5
    dt: float = 0.1
    horizon: int = 10
    obstacle_radius: float = 2.0
    safe_distance: float = 2.0
    max_steer_angle: float = 0.5
    steer_limit: float = 0.7
    throttle_limit: float = 0.8
    brake_limit: float = -0.8
    max_dsteer: float = 0.22
    max_daccel: float = 0.30
    target_speed: float = 6.0
    lookahead_distance: float = 20.0
    enable_cbf: bool = True
    cbf_activation_distance: float = 20.0
    ttc_activation_threshold: float = 3.0
    min_closing_speed: float = 0.5
    fallback_brake: float = -0.4
    solver_max_iter: int = 100
    solver_print_level: int = 0
    warm_start: bool = True


class MPCVehicleCBFSafetyFilter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.prev_exec_action = np.zeros(2, dtype=np.float32)

    def reset(self):
        self.prev_exec_action = np.zeros(2, dtype=np.float32)

    def _box_rate(self, a, prev):
        a = np.asarray(a, dtype=np.float32).copy()
        a[0] = np.clip(a[0], -self.cfg.steer_limit, self.cfg.steer_limit)
        a[1] = np.clip(a[1], self.cfg.brake_limit, self.cfg.throttle_limit)
        d = np.clip(a - prev, [-self.cfg.max_dsteer, -self.cfg.max_daccel], [self.cfg.max_dsteer, self.cfg.max_daccel])
        return np.clip(prev + d, -1.0, 1.0)

    def _extract_ego_state(self, env):
        try:
            u = getattr(env, "unwrapped", env)
            v = getattr(u, "vehicle", None) or getattr(u, "agent", None)
            pos = np.asarray(getattr(v, "position", [0.0, 0.0]), dtype=np.float32)
            yaw = float(getattr(v, "heading_theta", getattr(v, "heading", 0.0)))
            speed = float(getattr(v, "speed", np.linalg.norm(getattr(v, "velocity", [0.0, 0.0]))))
            return np.array([float(pos[0]), float(pos[1]), yaw, speed], dtype=np.float32), v
        except Exception:
            return np.zeros(4, dtype=np.float32), None

    def _extract_obstacles(self, env, ego_obj, ego_xy):
        obs = []
        try:
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
        except Exception:
            pass
        if not obs:
            return None
        obs.sort(key=lambda x: x[0])
        return obs[0][1]

    def _extract_lane_reference(self, env, ego):
        ref = np.array([ego[0] + self.cfg.lookahead_distance * np.cos(ego[2]), ego[1] + self.cfg.lookahead_distance * np.sin(ego[2]), ego[2], self.cfg.target_speed], dtype=np.float32)
        try:
            u = getattr(env, "unwrapped", env)
            veh = getattr(u, "vehicle", None) or getattr(u, "agent", None)
            lane = getattr(veh, "lane", None) or getattr(getattr(veh, "navigation", None), "current_lane", None)
            if lane is not None and hasattr(lane, "local_coordinates") and hasattr(lane, "position"):
                lon, _ = lane.local_coordinates(np.asarray([ego[0], ego[1]]))
                lon2 = lon + self.cfg.lookahead_distance
                p = lane.position(lon2, 0.0)
                yaw = float(lane.heading_theta(lon2)) if hasattr(lane, "heading_theta") else ego[2]
                ref = np.array([p[0], p[1], yaw, self.cfg.target_speed], dtype=np.float32)
        except Exception:
            pass
        return ref

    def evaluate_raw_cbf(self, raw_action, ego, obstacle):
        H = self.cfg.horizon
        dt = self.cfg.dt
        R = self.cfg.obstacle_radius + self.cfg.safe_distance
        x, y, yaw, v = map(float, ego)
        steer = float(np.clip(raw_action[0] / max(self.cfg.steer_limit, 1e-6) * self.cfg.max_steer_angle, -self.cfg.max_steer_angle, self.cfg.max_steer_angle))
        accel = float(raw_action[1])
        ox, oy, oyaw, ov = map(float, obstacle)
        vox, voy = ov * np.cos(oyaw), ov * np.sin(oyaw)
        ex0, ey0 = ox - x, oy - y
        evx0, evy0 = vox - v * np.cos(yaw), voy - v * np.sin(yaw)
        cross0 = ex0 * evy0 - ey0 * evx0
        sign_s = 1.0 if cross0 > 0 else -1.0
        min_d, min_ttc, h_min, cbf_min = np.inf, np.inf, np.inf, np.inf
        predicted_collision = 0.0
        hk_prev = None
        for k in range(H + 1):
            ex, ey = ox - x, oy - y
            evx, evy = vox - v * np.cos(yaw), voy - v * np.sin(yaw)
            dot = ex * evx + ey * evy
            dist = np.sqrt(ex * ex + ey * ey)
            min_d = min(min_d, dist)
            closing = max(dot / max(dist, 1e-6), 0.0)
            if closing > 1e-6:
                min_ttc = min(min_ttc, dist / closing)
            h = sign_s * (ex * evy - ey * evx) - R * np.sqrt(evx * evx + evy * evy)
            h_min = min(h_min, h)
            if hk_prev is not None:
                cbf_min = min(cbf_min, h - hk_prev + 0.5 * hk_prev)
            hk_prev = h
            if dist < R:
                predicted_collision = 1.0
            if k < H:
                yaw += dt * v * np.tan(steer) / max(self.cfg.wheelbase, 1e-6)
                v = max(0.0, v + 3.0 * accel * dt)
                x += dt * v * np.cos(yaw)
                y += dt * v * np.sin(yaw)
        return dict(h_min=float(h_min), cbf_min=float(cbf_min if np.isfinite(cbf_min) else h_min), min_dist=float(min_d), min_ttc=float(min_ttc), cbf_violation=float((h_min < 0.0) or (cbf_min < 0.0)), predicted_collision=float(predicted_collision), sign_s=float(sign_s), vox=float(vox), voy=float(voy))

    def _mpc_solve(self, ref, ego, obstacle_points, vox, voy, v_ego, sign_s, enable_cbf):
        if ca is None:
            return False, "casadi_unavailable", 0.0, 1.0
        N = self.cfg.horizon
        dt = self.cfg.dt
        R = self.cfg.obstacle_radius + self.cfg.safe_distance
        x = ca.SX.sym("x", N + 1); y = ca.SX.sym("y", N + 1); yaw = ca.SX.sym("yaw", N + 1)
        u_th = ca.SX.sym("uth", N); u_al = ca.SX.sym("ual", N)
        w = ca.vertcat(x, y, yaw, u_th, u_al)
        g = [x[0] - ego[0], y[0] - ego[1], yaw[0] - ego[2]]
        cost = 0
        for k in range(N):
            g += [x[k + 1] - (x[k] + dt * v_ego * ca.cos(yaw[k]))]
            g += [y[k + 1] - (y[k] + dt * v_ego * ca.sin(yaw[k]))]
            g += [yaw[k + 1] - (yaw[k] + dt * v_ego * ca.tan(u_th[k]) / self.cfg.wheelbase)]
            lat_err = y[k] - ref[1]
            head_err = ca.atan2(ca.sin(yaw[k] - ref[2]), ca.cos(yaw[k] - ref[2]))
            cost += 4.0 * lat_err * lat_err + 1.0 * head_err * head_err + 0.2 * u_th[k] * u_th[k] + 0.05 * (1 - u_al[k]) * (1 - u_al[k])
            if k > 0:
                cost += 0.2 * (u_th[k] - u_th[k - 1]) ** 2
            ex = obstacle_points[k, 0] - x[k]; ey = obstacle_points[k, 1] - y[k]
            exn = obstacle_points[k + 1, 0] - x[k + 1]; eyn = obstacle_points[k + 1, 1] - y[k + 1]
            evx = vox - v_ego * ca.cos(yaw[k]); evy = voy - v_ego * ca.sin(yaw[k])
            evxn = vox - v_ego * ca.cos(yaw[k + 1]); evyn = voy - v_ego * ca.sin(yaw[k + 1])
            hk = sign_s * (ex * evy - ey * evx) - R * ca.sqrt(evx * evx + evy * evy + 1e-6)
            hkn = sign_s * (exn * evyn - eyn * evxn) - R * ca.sqrt(evxn * evxn + evyn * evyn + 1e-6)
            dot = ex * evx + ey * evy
            cbf = ca.if_else(dot > 0, hkn - hk + u_al[k] * hk, 1.0)
            g += [ca.if_else(enable_cbf, cbf, 1.0)]
        nxyz = 3 * (N + 1)
        lbw = [-ca.inf] * nxyz + [-self.cfg.max_steer_angle] * N + [0.0] * N
        ubw = [ca.inf] * nxyz + [self.cfg.max_steer_angle] * N + [1.0] * N
        lbg = [0.0] * (3 + 3 * N) + [0.0] * N
        ubg = [0.0] * (3 + 3 * N) + [ca.inf] * N
        nlp = {"x": w, "f": cost, "g": ca.vertcat(*g)}
        solver = ca.nlpsol("solver", "ipopt", nlp, {"ipopt.print_level": self.cfg.solver_print_level, "print_time": False, "ipopt.max_iter": self.cfg.solver_max_iter})
        x0 = np.zeros(int(w.shape[0]))
        x0[0] = ego[0]; x0[N + 1] = ego[1]; x0[2 * (N + 1)] = ego[2]
        try:
            sol = solver(x0=x0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
            wv = np.array(sol["x"]).reshape(-1)
            u0 = float(wv[3 * (N + 1)])
            al = wv[3 * (N + 1) + N:3 * (N + 1) + 2 * N]
            return True, solver.stats().get("return_status", "success"), u0, float(np.min(al))
        except Exception as exc:
            return False, str(exc), 0.0, 1.0

    def project(self, raw_action, env=None, prev_exec_action=None):
        t0 = time.perf_counter()
        raw = np.asarray(raw_action, dtype=np.float32).reshape(2)
        prev = self.prev_exec_action if prev_exec_action is None else np.asarray(prev_exec_action, dtype=np.float32).reshape(2)
        ego, ego_obj = self._extract_ego_state(env)
        obs = self._extract_obstacles(env, ego_obj, ego[:2])
        ref = self._extract_lane_reference(env, ego)
        if obs is None:
            exec_action = self._box_rate(raw, prev)
            info = dict(selected_candidate_type="raw", filter_active=0.0, mpc_success=1.0, mpc_status="no_obstacle")
        else:
            raw_eval = self.evaluate_raw_cbf(raw, ego, obs)
            d0 = float(np.linalg.norm(obs[:2] - ego[:2]))
            closing_speed = max(0.0, float(np.dot(obs[:2] - ego[:2], np.array([raw_eval['vox'], raw_eval['voy']]) - ego[3] * np.array([np.cos(ego[2]), np.sin(ego[2])])) / max(d0, 1e-6)))
            cbf_active = bool((d0 < self.cfg.cbf_activation_distance and closing_speed > self.cfg.min_closing_speed) or (raw_eval['min_ttc'] < self.cfg.ttc_activation_threshold)) and self.cfg.enable_cbf
            if (not cbf_active) and (raw_eval["predicted_collision"] < 0.5):
                exec_action = self._box_rate(raw, prev)
                info = dict(selected_candidate_type="raw", filter_active=0.0, mpc_success=1.0, mpc_status="raw_safe")
            else:
                pts = np.stack([obs[:2] + i * self.cfg.dt * np.array([raw_eval["vox"], raw_eval["voy"]]) for i in range(self.cfg.horizon + 1)], axis=0)
                ok, status, steer, alpha_min = self._mpc_solve(ref, ego, pts, raw_eval["vox"], raw_eval["voy"], ego[3], raw_eval["sign_s"], cbf_active)
                if ok:
                    nsteer = np.clip(steer / max(self.cfg.max_steer_angle, 1e-6) * self.cfg.steer_limit, -self.cfg.steer_limit, self.cfg.steer_limit)
                    if raw_eval["predicted_collision"] > 0.5 or raw_eval["min_ttc"] < 1.0:
                        accel = self.cfg.fallback_brake
                    elif raw_eval["cbf_violation"] > 0.5:
                        accel = min(float(raw[1]), 0.2)
                    else:
                        accel = float(raw[1])
                    exec_action = self._box_rate(np.array([nsteer, accel], dtype=np.float32), prev)
                    info = dict(selected_candidate_type="mpc_cbf", filter_active=1.0, mpc_success=1.0, mpc_status=status, mpc_steer=float(nsteer), mpc_alpha_min=float(alpha_min), fallback=0.0, no_safe_candidate=0.0)
                else:
                    steer_fb = np.clip(0.2 * np.sin(ref[2] - ego[2]), -self.cfg.steer_limit, self.cfg.steer_limit)
                    exec_action = self._box_rate(np.array([steer_fb, self.cfg.fallback_brake], dtype=np.float32), prev)
                    info = dict(selected_candidate_type="mpc_failed_brake", filter_active=1.0, mpc_success=0.0, mpc_status=status, mpc_steer=float(steer_fb), mpc_alpha_min=0.0, fallback=1.0, no_safe_candidate=1.0)
            info.update(raw_eval)
        self.prev_exec_action = np.asarray(exec_action, dtype=np.float32).copy()
        diff = exec_action - raw
        min_ttc = float(info.get("min_ttc", np.inf))
        info.update(dict(raw_action=raw, exec_action=exec_action, projection_residual=float(np.linalg.norm(diff)), projection_cost=float(np.sum(diff ** 2)), sample_filter_active=0.0, mpc_cbf_active=1.0, num_candidates=1, num_valid_candidates=1 if info.get("selected_candidate_type") == "raw" else 0, valid_candidate_ratio=float(1.0 if info.get("selected_candidate_type") == "raw" else 0.0), fallback=float(info.get("fallback", 0.0)), no_safe_candidate=float(info.get("no_safe_candidate", 0.0)), min_pred_dist=float(info.get("min_dist", np.inf)), min_pred_ttc=min_ttc, min_pred_h_cbf=float(info.get("h_min", np.inf)), min_pred_cbf=float(info.get("cbf_min", np.inf)), cbf_violation=float(info.get("cbf_violation", 0.0)), predicted_collision=float(info.get("predicted_collision", 0.0)), mpc_steer=float(info.get("mpc_steer", 0.0)), mpc_alpha_min=float(info.get("mpc_alpha_min", 1.0)), sign_s=float(info.get("sign_s", 1.0)), min_pred_h_vo=float(info.get("h_min", np.inf)), vo_violation=float(info.get("cbf_violation", 0.0)), ttc_violation=float(min_ttc < self.cfg.ttc_activation_threshold), lane_violation=float(1.0 if info.get("selected_candidate_type") == "mpc_failed_brake" else 0.0), filter_time_ms=float((time.perf_counter() - t0) * 1000.0)))
        return exec_action, info
