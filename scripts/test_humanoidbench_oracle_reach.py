import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.humanoidbench_safe_wrapper import SafeHumanoidBenchWrapper
from relax.safety.humanoidbench_filter import HumanoidSafeFilterConfig


def get_hier_task(env):
    return env.env.unwrapped.task


def get_task(env):
    task = env.env.unwrapped.task
    if hasattr(task, "task"):
        return task.task
    return task


def get_goal(env):
    task = get_task(env)
    goal = getattr(task, "goal", None)
    if goal is None:
        raise AttributeError("Failed to read goal from task or task.task")
    return np.asarray(goal, dtype=np.float32)


def get_last_target(env):
    hier_task = get_hier_task(env)
    if not hasattr(hier_task, "last_target"):
        raise AttributeError("Failed to read last_target from hierarchical task")
    return np.asarray(hier_task.last_target, dtype=np.float32)


def get_left_hand_pos(env):
    return np.asarray(env.env.unwrapped.robot.left_hand_position(), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="h1hand-reach-v0")
    parser.add_argument("--policy_type", type=str, default="reach_single")
    parser.add_argument("--policy_path", type=str, required=True)
    parser.add_argument("--mean_path", type=str, required=True)
    parser.add_argument("--var_path", type=str, required=True)
    parser.add_argument("--use_filter", action="store_true")
    parser.add_argument("--render_mode", type=str, default="human", choices=["human", "rgb_array"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--max_delta", type=float, default=0.1)
    parser.add_argument("--residual_radius", type=float, default=0.35)
    parser.add_argument("--smooth_radius", type=float, default=0.25)
    parser.add_argument("--target_step_radius", type=float, default=0.08)
    parser.add_argument("--reachable_radius", type=float, default=0.45)
    parser.add_argument("--z_min_safe", type=float, default=0.4)
    parser.add_argument("--z_max_safe", type=float, default=1.8)
    args = parser.parse_args()

    env = SafeHumanoidBenchWrapper(
        args.env_name,
        use_filter=args.use_filter,
        render_mode=args.render_mode,
        filter_cfg=HumanoidSafeFilterConfig(
            residual_radius=args.residual_radius,
            smooth_radius=args.smooth_radius,
            max_delta=args.max_delta,
            target_step_radius=args.target_step_radius,
            reachable_radius=args.reachable_radius,
            z_min_safe=args.z_min_safe,
            z_max_safe=args.z_max_safe,
        ),
        policy_path=args.policy_path,
        mean_path=args.mean_path,
        var_path=args.var_path,
        policy_type=args.policy_type,
    )

    obs, _ = env.reset(seed=args.seed)
    ep_ret = 0.0
    min_hand_dist = float("inf")
    final_hand_dist = float("nan")
    last_info = {}

    for step in range(args.max_steps):
        goal = get_goal(env)
        last_target = get_last_target(env)
        left_hand = get_left_hand_pos(env)

        oracle_action = np.clip((goal - last_target) / args.max_delta, -1.0, 1.0).astype(np.float32)

        obs, reward, terminated, truncated, info = env.step(oracle_action)
        ep_ret += float(reward)
        done = bool(terminated or truncated)

        hand_dist = float(np.linalg.norm(left_hand - goal))
        target_to_goal_dist = float(np.linalg.norm(last_target - goal))
        min_hand_dist = min(min_hand_dist, hand_dist)
        final_hand_dist = hand_dist
        last_info = info

        if step % 50 == 0:
            print(
                f"step={step:04d}, reward={float(reward):.4f}, "
                f"hand_dist={hand_dist:.4f}, "
                f"target_to_goal_dist={target_to_goal_dist:.4f}, "
                f"FAR={float(info.get('filter_active', 0.0)):.3f}, "
                f"APR={float(info.get('projection_residual', 0.0)):.4f}, "
                f"success={float(info.get('is_success', 0.0)):.3f}"
            )

        if done:
            print(f"Episode finished at step={step}")
            break

    print("\n=== Episode summary ===")
    print(f"min_hand_dist={min_hand_dist:.4f}")
    print(f"final_hand_dist={final_hand_dist:.4f}")
    print(f"success={float(last_info.get('is_success', 0.0)):.3f}")
    print(f"return={ep_ret:.4f}")

    env.close()


if __name__ == "__main__":
    main()
