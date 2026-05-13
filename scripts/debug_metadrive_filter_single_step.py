import numpy as np

from envs.metadrive_safe_wrapper import SafeMetaDriveSampleWrapper


KEYS = [
    "projection_residual",
    "num_valid_candidates",
    "valid_candidate_ratio",
    "fallback",
    "no_safe_candidate",
    "min_pred_dist",
    "min_pred_ttc",
    "min_pred_h_vo",
    "vo_active",
    "ttc_violation",
    "vo_violation",
]


def main():
    env = SafeMetaDriveSampleWrapper(env_name="FlatThreeLaneStraight", use_filter=True, filter_type="sample_vo")
    try:
        env.reset(seed=0)
        raw_action = np.array([0.0, 0.0], dtype=np.float32)
        _, _, terminated, truncated, info = env.step(raw_action)
        print(f"raw_action: {raw_action.tolist()}")
        print(f"exec_action: {np.asarray(info.get('exec_action'), dtype=np.float32).tolist()}")
        for key in KEYS:
            print(f"{key}: {info.get(key)}")
        print(f"terminated: {terminated}, truncated: {truncated}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
