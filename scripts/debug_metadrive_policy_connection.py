from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import numpy as np

import relax.env.drive.lane_change  # noqa: F401
from relax.safety.metadrive_sample_filter import SampleBasedVehicleSafetyFilter, SampleVehicleFilterConfig
from scripts.interactive_metadrive_manual_filter_demo import attach_filter_to_active_policy, get_current_policy, parse_args


def main():
    args = parse_args()
    cfg = SampleVehicleFilterConfig()
    filt = SampleBasedVehicleSafetyFilter(cfg)

    env = gym.make(args.env_name, use_render=True, manual_control=True, controller="keyboard")
    try:
        env.reset(seed=args.seed)
        policy = get_current_policy(env)
        print(f"active policy type before patch: {type(policy).__name__}")
        policy = attach_filter_to_active_policy(env, filt, args)

        for i in range(10):
            _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
            env.render()
            policy = get_current_policy(env)
            info = getattr(policy, "_safe_filter_last_info", {})
            print(
                f"step={i + 1} active_policy_type={type(policy).__name__} "
                f"is_patched={getattr(policy, '_safe_filter_is_patched', False)} "
                f"act_call_count={getattr(policy, '_safe_filter_act_call_count', 0)} "
                f"num_candidates={info.get('num_candidates', 0)} "
                f"selected_candidate_type={info.get('selected_candidate_type', 'n/a')} "
                f"filter_time_ms={info.get('filter_time_ms', 0):.2f} "
                f"speed={float(getattr(env.unwrapped.agent, 'speed', 0.0)):.3f}"
            )
            if terminated or truncated:
                env.reset()
                filt.reset()
                policy = attach_filter_to_active_policy(env, filt, args)
    finally:
        env.close()


if __name__ == "__main__":
    main()
