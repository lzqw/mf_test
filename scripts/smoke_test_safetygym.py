import argparse
import sys
from pathlib import Path

import numpy as np
import safety_gymnasium

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from envs.safety_gym_safe_wrapper import SafeSafetyGymWrapper

DEFAULT_ENVS = ["SafetyPointGoal1-v0", "SafetyPointPush1-v0", "SafetyCarGoal1-v0", "SafetyCarPush1-v0"]
REQ_KEYS = ["cost", "raw_action", "exec_action", "projection_residual", "projection_cost", "filter_active", "raw_action_norm", "exec_action_norm", "safety_violation", "constraint_violation"]


def candidate_env_ids():
    ids = list(getattr(safety_gymnasium, "registry", {}).keys())
    if not ids:
        import gymnasium as gym
        ids = list(gym.envs.registry.keys())
    return sorted([x for x in ids if any(k in x for k in ["Safety", "Point", "Car", "Goal", "Push"])])


def test_env(env_id, episodes, use_filter, filter_type, render_mode):
    print(f"\n=== Testing {env_id} ===")
    try:
        env = safety_gymnasium.make(env_id, render_mode=render_mode)
    except Exception as e:
        print(f"[FAIL CREATE] {env_id}: {e}")
        print("Candidates:", candidate_env_ids())
        return False

    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)
    print("obs_dim:", int(np.prod(env.observation_space.shape)))
    print("act_dim:", int(np.prod(env.action_space.shape)))

    for _ in range(episodes):
        obs, info = env.reset()
        for i in range(5):
            out = env.step(env.action_space.sample())
            if len(out) not in (5, 6):
                print(f"[FAIL API] expected 5/6 returns got {len(out)}")
                return False
            if len(out) == 6:
                obs, reward, cost, term, trunc, info = out
            else:
                obs, reward, term, trunc, info = out
                cost = float(info.get("cost", 0.0))
            if term or trunc:
                break

    wenv = SafeSafetyGymWrapper(env_id=env_id, use_filter=use_filter, filter_type=filter_type, render_mode=render_mode)
    for _ in range(episodes):
        obs, info = wenv.reset()
        for i in range(5):
            obs, reward, term, trunc, info = wenv.step(wenv.action_space.sample())
            if filter_type in ["gt_shield", "sample_shield"] and i < 3:
                print("[shield]", {k: info.get(k, np.nan) for k in ["gt_known", "num_hazards", "nearest_hazard_dist", "current_min_h", "predicted_min_h", "num_safe_candidates", "safe_candidate_ratio", "emergency_active", "selected_candidate_type", "projection_residual", "filter_active_005", "filter_active_010"]})
                if not np.isfinite(float(info.get("num_hazards", np.nan))) or float(info.get("num_hazards", 0.0)) <= 0:
                    print(f"[WARN] {env_id} has no hazard ground-truth extracted at step {i}")
            miss = [k for k in REQ_KEYS if k not in info]
            if miss:
                print("[FAIL KEYS] missing:", miss)
                return False
            if term or trunc:
                break
    print("[PASS]", env_id)
    return True


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", type=str, default=None)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--use_filter", action="store_true")
    p.add_argument("--filter_type", default="hybrid", choices=["none", "action", "smooth", "control", "hybrid", "sample_shield", "gt_shield"])
    p.add_argument("--render_mode", default=None)
    args = p.parse_args()

    envs = [args.env_id] if args.env_id else DEFAULT_ENVS
    ok = []
    for env_id in envs:
        ok.append(test_env(env_id, args.episodes, args.use_filter, args.filter_type, args.render_mode))
    print("\nSummary:", {e: s for e, s in zip(envs, ok)})
