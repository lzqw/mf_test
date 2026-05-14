import argparse
import sys
from pathlib import Path

import numpy as np
import safety_gymnasium

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from relax.safety.safety_gym_filter import SafetyGymFilterConfig, SafetyGymHardFilter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", type=str, default="SafetyPointGoal1-v0")
    p.add_argument("--steps", type=int, default=3)
    args = p.parse_args()

    env = safety_gymnasium.make(args.env_id)
    obs, info = env.reset()
    u = getattr(env, "unwrapped", env)
    print("env type:", type(env))
    print("env.unwrapped type:", type(u))
    print("env.unwrapped keys:", sorted(list(getattr(u, "__dict__", {}).keys())))
    task = getattr(u, "task", None)
    if task is not None:
        print("task type:", type(task))
        print("task keys:", sorted(list(getattr(task, "__dict__", {}).keys())))

    f = SafetyGymHardFilter(SafetyGymFilterConfig(), filter_type="sample_shield")

    def dump(tag):
        ego = f._extract_ego_state_from_env(env)
        hazards = f._extract_hazards_from_env(env)
        objects = f._extract_objects_from_env(env)
        print(f"\n[{tag}] ego known={ego['known']} source={ego['source']} pos={ego['pos']} vel={ego['vel']} heading={ego['heading']:.4f} speed={ego['speed']:.4f}")
        print("num_hazards:", len(hazards))
        for i, h in enumerate(hazards[:10]):
            print(f"  hazard[{i}] pos={h['pos']} radius={h['radius']:.4f} source={h['source']}")
        print("num_objects:", len(objects))
        for i, o in enumerate(objects[:10]):
            print(f"  object[{i}] pos={o['pos']} radius={o['radius']:.4f} source={o['source']}")

    dump('reset')
    for t in range(args.steps):
        out = env.step(env.action_space.sample())
        if len(out) == 6:
            obs, rew, cost, term, trunc, info = out
        elif len(out) == 5:
            obs, rew, term, trunc, info = out
            cost = info.get("cost", 0.0)
        else:
            raise RuntimeError(f"Unexpected env.step return length: {len(out)}")
        ego = f._extract_ego_state_from_env(env)
        hazards = f._extract_hazards_from_env(env)
        d = np.nan
        if ego['known'] and hazards:
            d = min(float(np.linalg.norm(h['pos'] - ego['pos']) - h['radius']) for h in hazards)
        print(f"step={t} ego_known={ego['known']} ego_pos={ego['pos']} num_hazards={len(hazards)} nearest={d}")
        if term or trunc:
            obs, info = env.reset()
    env.close()


if __name__ == '__main__':
    main()
