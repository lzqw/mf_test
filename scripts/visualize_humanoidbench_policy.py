import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.humanoidbench_safe_wrapper import SafeHumanoidBenchWrapper
from relax.safety.humanoidbench_filter import HumanoidSafeFilterConfig
from scripts.train_safe_humanoidbench import make_algo


def load_args(log_dir: Path):
    args_path = log_dir / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"args.json not found: {args_path}")
    with open(args_path, "r") as f:
        args_dict = json.load(f)
    return SimpleNamespace(**args_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="checkpoint.pkl",
        help="Checkpoint filename inside log_dir, e.g. checkpoint.pkl, best_return_checkpoint.pkl, best_hand_dist_checkpoint.pkl",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional explicit checkpoint path. If provided, overrides log_dir/checkpoint_name.",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render_mode", type=str, default="human", choices=["human", "rgb_array"])
    parser.add_argument("--sleep", type=float, default=0.01)
    parser.add_argument("--deterministic_seed", type=int, default=123)
    parser.add_argument("--no_filter", action="store_true")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    train_args = load_args(log_dir)

    ckpt_path = Path(args.checkpoint_path) if args.checkpoint_path else (log_dir / args.checkpoint_name)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    print("=" * 80)
    print("Visualizing HumanoidBench policy")
    print("log_dir:", log_dir)
    print("checkpoint:", ckpt_path)
    print("env_name:", train_args.env_name)
    print("policy_type:", train_args.policy_type)
    print("policy_path:", train_args.policy_path)
    print("use_filter:", not args.no_filter)
    print("=" * 80)

    blocked_hands = getattr(train_args, "blocked_hands", None)
    small_obs = getattr(train_args, "small_obs", None)

    env = SafeHumanoidBenchWrapper(
        train_args.env_name,
        use_filter=(not args.no_filter),
        render_mode=args.render_mode,
        filter_cfg=HumanoidSafeFilterConfig(
            residual_radius=getattr(train_args, "residual_radius", 0.35),
            smooth_radius=getattr(train_args, "smooth_radius", 0.25),
            max_delta=getattr(train_args, "max_delta", 0.1),
            target_step_radius=getattr(train_args, "target_step_radius", 0.08),
            reachable_radius=getattr(train_args, "reachable_radius", 0.45),
            z_min_safe=getattr(train_args, "z_min_safe", 0.4),
            z_max_safe=getattr(train_args, "z_max_safe", 1.8),
        ),
        policy_path=getattr(train_args, "policy_path", None),
        mean_path=getattr(train_args, "mean_path", None),
        var_path=getattr(train_args, "var_path", None),
        policy_type=getattr(train_args, "policy_type", None),
        blocked_hands=blocked_hands,
        small_obs=small_obs,
    )

    obs, _ = env.reset(seed=args.seed)

    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)
    print("obs_dim:", env.observation_space.shape[0])
    print("act_dim:", env.action_space.shape[0])

    agent = make_algo(train_args, env.observation_space.shape[0], env.action_space.shape[0])

    with open(ckpt_path, "rb") as f:
        agent.state = pickle.load(f)

    key = jax.random.PRNGKey(args.deterministic_seed)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        ep_ret = 0.0
        far = []
        apr = []
        falls = []
        hand_dists = []
        target_dists = []

        print(f"\n[Episode {ep}] start")

        for step in range(args.max_steps):
            key, subkey = jax.random.split(key)
            raw_action = np.asarray(agent.get_action(subkey, obs[None])[0])

            obs, reward, terminated, truncated, info = env.step(raw_action)
            done = terminated or truncated
            ep_ret += float(reward)

            far.append(float(info.get("filter_active", 0.0)))
            apr.append(float(info.get("projection_residual", 0.0)))
            falls.append(float(info.get("fall", 0.0)))

            if "hand_dist" in info:
                hand_dists.append(float(info["hand_dist"]))
            if "target_dist" in info:
                target_dists.append(float(info["target_dist"]))

            if args.render_mode == "human":
                env.render()
                if args.sleep > 0:
                    time.sleep(args.sleep)

            if step % 100 == 0:
                msg = (
                    f"step={step:04d}, "
                    f"reward={float(reward):.3f}, "
                    f"return={ep_ret:.3f}, "
                    f"FAR={np.mean(far):.3f}, "
                    f"APR={np.mean(apr):.4f}, "
                    f"fall={np.max(falls):.0f}"
                )
                if hand_dists:
                    msg += f", hand_dist={hand_dists[-1]:.3f}"
                if target_dists:
                    msg += f", target_dist={target_dists[-1]:.3f}"
                print(msg)

            if done:
                print(f"episode terminated at step {step}")
                break

        print(
            f"[Episode {ep}] "
            f"return={ep_ret:.3f}, "
            f"steps={step + 1}, "
            f"FAR={np.mean(far):.3f}, "
            f"APR={np.mean(apr):.4f}, "
            f"fall={np.max(falls):.0f}, "
            f"success={float(info.get('is_success', 0.0)):.3f}"
        )
        if hand_dists:
            print(f"final hand_dist={hand_dists[-1]:.3f}")
        if target_dists:
            print(f"final target_dist={target_dists[-1]:.3f}")

    env.close()


if __name__ == "__main__":
    main()
