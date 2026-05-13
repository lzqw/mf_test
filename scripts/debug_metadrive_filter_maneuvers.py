from pathlib import Path
import sys
import numpy as np
import gymnasium as gym

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import relax.env.drive.lane_change  # noqa: F401
from relax.safety.metadrive_sample_filter import SampleBasedVehicleSafetyFilter, SampleVehicleFilterConfig


def main():
    cfg = SampleVehicleFilterConfig()
    filt = SampleBasedVehicleSafetyFilter(cfg)
    env = gym.make("FlatThreeLaneStraight", use_render=False, manual_control=False)
    obs, info = env.reset(seed=0)
    del obs, info
    filt.reset()

    test_actions = [
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.5], dtype=np.float32),
        np.array([0.3, 0.5], dtype=np.float32),
        np.array([-0.3, 0.5], dtype=np.float32),
        np.array([0.0, -0.5], dtype=np.float32),
    ]

    for i, raw in enumerate(test_actions):
        exec_action, info = filt.project(raw, env=env.unwrapped)
        print(f"[{i}] raw_action={raw} exec_action={exec_action}")
        for k in [
            "selected_candidate_type", "num_valid_candidates", "valid_candidate_ratio", "fallback",
            "predicted_opposite_lane", "min_corridor_margin", "max_abs_lateral", "min_pred_dist",
            "min_pred_ttc", "min_pred_h_vo", "longitudinal_progress", "pass_obstacle_bonus"
        ]:
            print(f"    {k}={info.get(k)}")
        env.step(exec_action)

    env.close()


if __name__ == "__main__":
    main()
