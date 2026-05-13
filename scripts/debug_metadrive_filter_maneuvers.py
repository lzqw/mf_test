from pathlib import Path
import sys
import numpy as np
import gymnasium as gym

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import relax.env.drive.lane_change  # noqa: F401
from relax.safety.metadrive_sample_filter import SampleBasedVehicleSafetyFilter, SampleVehicleFilterConfig


def print_case(case_name, raw, exec_action, info):
    print(f"[{case_name}] raw_action={raw} exec_action={exec_action}")
    for k in [
        "selected_candidate_type",
        "filter_active",
        "projection_residual",
        "num_valid_candidates",
        "valid_candidate_ratio",
        "fallback",
        "predicted_opposite_lane",
        "min_corridor_margin",
        "max_abs_lateral",
        "longitudinal_progress",
        "pass_obstacle_bonus",
    ]:
        print(f"    {k}={info.get(k)}")


def main():
    cfg = SampleVehicleFilterConfig()
    filt = SampleBasedVehicleSafetyFilter(cfg)
    env = gym.make("FlatThreeLaneStraight", use_render=False, manual_control=False)

    obs, info = env.reset(seed=0)
    del obs, info
    filt.reset()

    raw_reset = np.array([0.0, 0.0], dtype=np.float32)
    exec_action, info = filt.project(raw_reset, env=env.unwrapped)
    print_case("reset", raw_reset, exec_action, info)

    # drive forward to create a near-obstacle scenario before evaluating again
    for _ in range(40):
        env.step(np.array([0.0, 0.6], dtype=np.float32))

    raw_near = np.array([0.3, 0.5], dtype=np.float32)
    exec_action, info = filt.project(raw_near, env=env.unwrapped)
    print_case("near_obstacle", raw_near, exec_action, info)

    env.close()


if __name__ == "__main__":
    main()
