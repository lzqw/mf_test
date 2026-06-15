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
        self.obstacle_center = np.asarray(
            np.zeros(2, dtype=np.float32) if obstacle_center is None else obstacle_center,
            dtype=np.float32,
        )
        self.obstacle_radius = float(obstacle_radius)
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
        self._obs_center = np.asarray(self.cfg.obstacle_center, dtype=np.float32)
        self._action_gain = 1.0

    def set_action_gain(self, action_gain: float):
        self._action_gain = float(action_gain)

    def _is_action_feasible(self, state, action):
        state = np.asarray(state, dtype=np.float32)
        action = _clip_action(action)
        pos = state[:2]
        vel = state[2:]
        a = action * self.cfg.a_max * self._action_gain
        # double-integrator predicted position for one step
        p_next = pos + self.cfg.dt * vel + 0.5 * (self.cfg.dt ** 2) * a
        safety_radius = self.cfg.obstacle_radius + self.cfg.eps_obs
        clearance = np.linalg.norm(p_next - self._obs_center) - safety_radius
        return bool(
            (self.cfg.x_min <= p_next[0] <= self.cfg.x_max)
            and (self.cfg.y_min <= p_next[1] <= self.cfg.y_max)
            and (clearance >= 0.0)
        )

    def is_action_feasible_np(self, state, action):
        return self._is_action_feasible(state, action)

    def predict_pos(self, state, action, action_gain=None):
        if action_gain is None:
            action_gain = self._action_gain
        state = np.asarray(state, dtype=np.float32)
        action = _clip_action(action)
        pos = state[:2]
        vel = state[2:]
        a = action * self.cfg.a_max * float(action_gain)
        return pos + self.cfg.dt * vel + 0.5 * (self.cfg.dt ** 2) * a

    def project_action_np(self, state, raw_action):
        raw_action = _clip_action(raw_action)
        state = np.asarray(state, dtype=np.float32)

        if self._is_action_feasible(state, raw_action):
            return raw_action.copy(), False, 0.0, False, False, False

        # Evaluate every grid candidate in parallel.
        actions = self.action_grid
        p = state[:2]
        vel = state[2:]
        a = actions * self.cfg.a_max * self._action_gain
        p_next = p[None, :] + self.cfg.dt * vel[None, :] + 0.5 * (self.cfg.dt ** 2) * a
        center = self._obs_center[None, :]
        clearance = np.linalg.norm(p_next - center, axis=1) - (self.cfg.obstacle_radius + self.cfg.eps_obs)
        in_box = (
            (p_next[:, 0] >= self.cfg.x_min)
            & (p_next[:, 0] <= self.cfg.x_max)
            & (p_next[:, 1] >= self.cfg.y_min)
            & (p_next[:, 1] <= self.cfg.y_max)
            & (clearance >= 0.0)
        )

        safe_actions = actions[in_box]
        if safe_actions.shape[0] > 0:
            deltas = safe_actions - raw_action[None, :]
            dist2 = np.sum(deltas * deltas, axis=1)
            idx = int(np.argmin(dist2))
            projected = np.asarray(safe_actions[idx], dtype=np.float32)
            gap = float(np.linalg.norm(raw_action - projected))
            return projected, True, gap, True, False, False

        # No safe candidate found; choose the action giving the largest clearance.
        clear = np.maximum(clearance, -np.inf)
        idx = int(np.argmax(clear))
        fallback = np.asarray(actions[idx], dtype=np.float32)
        gap = float(np.linalg.norm(raw_action - fallback))
        return fallback, True, gap, True, True, True
