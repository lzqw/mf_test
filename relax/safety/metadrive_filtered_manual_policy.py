import numpy as np

from metadrive.policy.manual_control_policy import ManualControlPolicy

from relax.safety.metadrive_sample_filter import SampleBasedVehicleSafetyFilter, SampleVehicleFilterConfig


def build_default_filter_info(raw_action, exec_action):
    raw_action = np.asarray(raw_action, dtype=np.float32).reshape(2)
    exec_action = np.asarray(exec_action, dtype=np.float32).reshape(2)
    diff = exec_action - raw_action
    return {
        "raw_action": raw_action.copy(),
        "exec_action": exec_action.copy(),
        "filter_active": float(np.linalg.norm(diff) > 1e-6),
        "projection_residual": float(np.linalg.norm(diff)),
        "projection_cost": float(np.sum(diff ** 2)),
        "num_candidates": 0,
        "num_valid_candidates": 0,
        "valid_candidate_ratio": 0.0,
        "fallback": 0.0,
        "no_safe_candidate": 0.0,
        "min_pred_dist": float(np.inf),
        "min_pred_ttc": float(np.inf),
        "min_pred_h_vo": float(np.inf),
        "filter_time_ms": 0.0,
        "selected_candidate_type": "none",
        "selected_is_maneuver": 0.0,
        "predicted_opposite_lane": 0.0,
        "min_corridor_margin": float(np.inf),
        "max_abs_lateral": 0.0,
        "longitudinal_progress": 0.0,
        "pass_obstacle_bonus": 0.0,
    }


def rate_filter(raw_action, prev_exec_action):
    raw_action = np.asarray(raw_action, dtype=np.float32).reshape(2)
    prev_exec_action = np.asarray(prev_exec_action, dtype=np.float32).reshape(2)

    clipped = raw_action.copy()
    clipped[0] = np.clip(clipped[0], -0.7, 0.7)
    clipped[1] = np.clip(clipped[1], -0.8, 0.8)
    delta = np.clip(clipped - prev_exec_action, [-0.12, -0.2], [0.12, 0.2])
    exec_action = np.clip(prev_exec_action + delta, -1.0, 1.0).astype(np.float32)

    return exec_action, build_default_filter_info(raw_action, exec_action)


class FilteredManualControlPolicy(ManualControlPolicy):
    filter_cfg = SampleVehicleFilterConfig()
    filter_type = "sample_vo"
    use_filter = True
    env_ref = None

    @classmethod
    def configure(cls, filter_cfg=None, filter_type="sample_vo", use_filter=True, env_ref=None):
        cls.filter_cfg = filter_cfg or SampleVehicleFilterConfig()
        cls.filter_type = filter_type
        cls.use_filter = use_filter
        cls.env_ref = env_ref

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.safe_filter = SampleBasedVehicleSafetyFilter(self.__class__.filter_cfg)
        self.prev_exec_action = np.zeros(2, dtype=np.float32)
        self.last_filter_info = {}
        self.last_raw_action = np.zeros(2, dtype=np.float32)
        self.last_exec_action = np.zeros(2, dtype=np.float32)

    def reset_filter_state(self):
        self.safe_filter.reset()
        self.prev_exec_action = np.zeros(2, dtype=np.float32)
        self.last_filter_info = {}
        self.last_raw_action = np.zeros(2, dtype=np.float32)
        self.last_exec_action = np.zeros(2, dtype=np.float32)

    def act(self, *args, **kwargs):
        raw_action = np.asarray(super().act(*args, **kwargs), dtype=np.float32).reshape(2)

        if (not self.__class__.use_filter) or self.__class__.filter_type == "none":
            exec_action = np.clip(raw_action, -1.0, 1.0).astype(np.float32)
            filter_info = build_default_filter_info(raw_action, exec_action)
        elif self.__class__.filter_type == "rate":
            exec_action, filter_info = rate_filter(raw_action, self.prev_exec_action)
        else:
            env = self.__class__.env_ref
            if env is not None:
                env = getattr(env, "unwrapped", env)
            exec_action, filter_info = self.safe_filter.project(
                raw_action,
                env=env,
                prev_exec_action=self.prev_exec_action,
            )

        self.prev_exec_action = np.asarray(exec_action, dtype=np.float32).copy()
        self.last_raw_action = raw_action.copy()
        self.last_exec_action = self.prev_exec_action.copy()
        self.last_filter_info = dict(filter_info)
        return exec_action
