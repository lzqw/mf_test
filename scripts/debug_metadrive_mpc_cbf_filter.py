from pathlib import Path
import sys
import numpy as np
import gymnasium as gym

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import relax.env.drive.lane_change  # noqa: F401
from relax.safety.metadrive_mpc_cbf_filter import MPCVehicleCBFSafetyFilter, MPCVehicleCBFConfig


def main():
    env = gym.make("FlatThreeLaneStraight")
    filt = MPCVehicleCBFSafetyFilter(MPCVehicleCBFConfig())
    obs, _ = env.reset(seed=0)
    filt.reset()
    exec_action, info = filt.project([0.0, 0.0], env=env, prev_exec_action=np.zeros(2, dtype=np.float32))
    print("reset raw/exec/info:", [0.0, 0.0], exec_action, info.get("selected_candidate_type"), info.get("filter_active"))
    for _ in range(20):
        obs, reward, terminated, truncated, env_info = env.step(np.array([0.0, 0.5], dtype=np.float32))
        if terminated or truncated:
            obs, _ = env.reset()
            filt.reset()
            break
    exec_action, info = filt.project([0.3, 0.5], env=env, prev_exec_action=exec_action)
    keys = ["selected_candidate_type", "mpc_success", "mpc_status", "raw_action", "exec_action", "filter_active", "projection_residual", "min_pred_dist", "min_pred_ttc", "min_pred_h_cbf", "min_pred_cbf", "cbf_violation", "fallback", "sign_s"]
    for k in keys:
        print(f"{k}: {info.get(k)}")
    env.close()


if __name__ == "__main__":
    main()
