import numpy as np


def make_action_grid(grid_size=41):
    vals = np.linspace(-1.0, 1.0, int(grid_size), dtype=np.float32)
    gx, gy = np.meshgrid(vals, vals)
    return np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1).astype(np.float32)


class DoubleIntegratorObstacleConfig:
    """Configuration for the double-integrator obstacle filter."""

    def __init__(
        self,
        dt=0.1,
        a_max=3.0,
        obstacle_center=None,
        obstacle_radius=0.8,
        obstacle_centers=None,
        obstacle_radii=None,
        eps_obs=0.08,
        eps_box=0.05,
        x_min=-3.5,
        x_max=3.5,
        y_min=-2.0,
        y_max=2.0,
        grid_size=61,
    ):
        self.dt = float(dt)
        self.a_max = float(a_max)
        if obstacle_centers is None:
            obstacle_centers = [np.zeros(2, dtype=np.float32) if obstacle_center is None else obstacle_center]
        if obstacle_radii is None:
            obstacle_radii = [obstacle_radius]
        self.obstacle_centers = np.asarray(obstacle_centers, dtype=np.float32).reshape(-1, 2)
        self.obstacle_radii = np.asarray(obstacle_radii, dtype=np.float32).reshape(-1)
        if self.obstacle_centers.shape[0] != self.obstacle_radii.shape[0]:
            raise ValueError("obstacle_centers and obstacle_radii must have the same length")
        # Backward-compatible single-obstacle aliases.
        self.obstacle_center = self.obstacle_centers[0].copy()
        self.obstacle_radius = float(self.obstacle_radii[0])
        self.eps_obs = float(eps_obs)
        self.eps_box = float(eps_box)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.grid_size = int(grid_size)
        self.action_grid = make_action_grid(self.grid_size)


def _clip_action(action):
    return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)


class DoubleIntegratorObstacleFilter:
    def __init__(self, cfg: DoubleIntegratorObstacleConfig):
        self.cfg = cfg
        self.action_grid = np.asarray(cfg.action_grid, dtype=np.float32)
        self._obs_centers = np.asarray(self.cfg.obstacle_centers, dtype=np.float32)
        self._obs_radii = np.asarray(self.cfg.obstacle_radii, dtype=np.float32)
        self._action_gain = 1.0

    def set_action_gain(self, action_gain: float):
        self._action_gain = float(action_gain)

    def predict_pos(self, state, action, action_gain=None):
        if action_gain is None:
            action_gain = self._action_gain
        state = np.asarray(state, dtype=np.float32)
        action = _clip_action(action)
        pos = state[:2]
        vel = state[2:]
        a = action * self.cfg.a_max * float(action_gain)
        return pos + self.cfg.dt * vel + 0.5 * (self.cfg.dt ** 2) * a

    def predicted_tight_clearances(self, state, action, action_gain=None):
        p_next = self.predict_pos(state, action, action_gain=action_gain)
        return np.linalg.norm(p_next[None, :] - self._obs_centers, axis=1) - (self._obs_radii + self.cfg.eps_obs)

    def _workspace_ok(self, p_next):
        return bool(
            (self.cfg.x_min <= p_next[0] <= self.cfg.x_max)
            and (self.cfg.y_min <= p_next[1] <= self.cfg.y_max)
        )

    def _feasibility_details(self, state, action):
        p_next = self.predict_pos(state, action)
        tight_clearances = self.predicted_tight_clearances(state, action)
        min_clearance = float(np.min(tight_clearances))
        nearest_obstacle_id = int(np.argmin(tight_clearances))
        feasible = self._workspace_ok(p_next) and (min_clearance >= 0.0)
        return feasible, min_clearance, nearest_obstacle_id, p_next

    def _is_action_feasible(self, state, action):
        feasible, _, _, _ = self._feasibility_details(state, action)
        return feasible

    def is_action_feasible_np(self, state, action):
        return self._is_action_feasible(state, action)

    def project_action_np(self, state, raw_action, return_details=False):
        raw_action = _clip_action(raw_action)
        state = np.asarray(state, dtype=np.float32)

        feasible, min_clearance_raw, nearest_raw, _ = self._feasibility_details(state, raw_action)
        if feasible:
            details = {
                "min_predicted_tight_clearance": float(min_clearance_raw),
                "nearest_obstacle_id": int(nearest_raw),
            }
            if return_details:
                return raw_action.copy(), False, 0.0, False, False, False, details
            return raw_action.copy(), False, 0.0, False, False, False

        actions = self.action_grid
        p = state[:2]
        vel = state[2:]
        a = actions * self.cfg.a_max * self._action_gain
        p_next = p[None, :] + self.cfg.dt * vel[None, :] + 0.5 * (self.cfg.dt ** 2) * a
        deltas = p_next[:, None, :] - self._obs_centers[None, :, :]
        tight_clearances = np.linalg.norm(deltas, axis=2) - (self._obs_radii[None, :] + self.cfg.eps_obs)
        min_clearance = np.min(tight_clearances, axis=1)
        nearest_ids = np.argmin(tight_clearances, axis=1)
        in_box = (
            (p_next[:, 0] >= self.cfg.x_min)
            & (p_next[:, 0] <= self.cfg.x_max)
            & (p_next[:, 1] >= self.cfg.y_min)
            & (p_next[:, 1] <= self.cfg.y_max)
        )
        feasible_mask = in_box & (min_clearance >= 0.0)

        if np.any(feasible_mask):
            safe_actions = actions[feasible_mask]
            safe_clearance = min_clearance[feasible_mask]
            safe_nearest = nearest_ids[feasible_mask]
            dist2 = np.sum((safe_actions - raw_action[None, :]) ** 2, axis=1)
            idx = int(np.argmin(dist2))
            projected = np.asarray(safe_actions[idx], dtype=np.float32)
            gap = float(np.linalg.norm(raw_action - projected))
            details = {
                "min_predicted_tight_clearance": float(safe_clearance[idx]),
                "nearest_obstacle_id": int(safe_nearest[idx]),
            }
            if return_details:
                return projected, True, gap, True, False, False, details
            return projected, True, gap, True, False, False

        idx = int(np.argmax(min_clearance))
        fallback = np.asarray(actions[idx], dtype=np.float32)
        gap = float(np.linalg.norm(raw_action - fallback))
        details = {
            "min_predicted_tight_clearance": float(min_clearance[idx]),
            "nearest_obstacle_id": int(nearest_ids[idx]),
        }
        if return_details:
            return fallback, True, gap, True, True, True, details
        return fallback, True, gap, True, True, True
