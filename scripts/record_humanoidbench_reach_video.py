import argparse
import json
import pickle
import sys
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


def load_args(log_dir: Path) -> SimpleNamespace:
    args_path = log_dir / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"args.json not found: {args_path}")
    with open(args_path, "r", encoding="utf-8") as f:
        args_dict = json.load(f)
    return SimpleNamespace(**args_dict)


def _build_env(train_args: SimpleNamespace, use_filter: bool, reference_filter_mode, reference_filter_threshold, reference_filter_type):
    return SafeHumanoidBenchWrapper(
        train_args.env_name,
        use_filter=use_filter,
        render_mode="rgb_array",
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
        augment_reach_obs=getattr(train_args, "augment_reach_obs", False),
        reference_filter_mode=reference_filter_mode,
        reference_filter_threshold=reference_filter_threshold,
        reference_filter_type=reference_filter_type,
        blocked_hands=getattr(train_args, "blocked_hands", None),
        small_obs=getattr(train_args, "small_obs", None),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--checkpoint_name", type=str, default="checkpoint.pkl")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--output", type=str, default="videos/reach_ours_filter_demo.mp4")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--max_steps_per_goal", type=int, default=350)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--render_every", type=int, default=1)
    parser.add_argument("--deterministic_seed", type=int, default=123)
    parser.add_argument("--success_threshold", type=float, default=0.06)
    parser.add_argument("--hold_success_frames", type=int, default=20)
    parser.add_argument("--hold_reset_frames", type=int, default=15)
    parser.add_argument("--no_filter", action="store_true")
    parser.add_argument("--force_reference_filter_mode", type=str, default=None, choices=["none", "goal"])
    parser.add_argument("--force_reference_filter_threshold", type=float, default=None)
    parser.add_argument("--force_reference_filter_type", type=str, default=None, choices=["replace", "ball"])
    args = parser.parse_args()

    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise RuntimeError(
            "imageio/imageio-ffmpeg is required. Please run: pip install imageio imageio-ffmpeg"
        ) from exc

    log_dir = Path(args.log_dir)
    train_args = load_args(log_dir)
    ckpt_path = Path(args.checkpoint_path) if args.checkpoint_path else (log_dir / args.checkpoint_name)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    reference_filter_mode = args.force_reference_filter_mode or getattr(train_args, "reference_filter_mode", "none")
    reference_filter_threshold = (
        args.force_reference_filter_threshold
        if args.force_reference_filter_threshold is not None
        else getattr(train_args, "reference_filter_threshold", 0.25)
    )
    reference_filter_type = args.force_reference_filter_type or getattr(train_args, "reference_filter_type", "replace")

    env = _build_env(
        train_args,
        use_filter=(not args.no_filter),
        reference_filter_mode=reference_filter_mode,
        reference_filter_threshold=reference_filter_threshold,
        reference_filter_type=reference_filter_type,
    )

    agent = make_algo(train_args, env.observation_space.shape[0], env.action_space.shape[0])
    with open(ckpt_path, "rb") as f:
        agent.state = pickle.load(f)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(args.deterministic_seed)

    print(f"Saving video to: {out_path}")
    print(f"use_filter={not args.no_filter}, ref_mode={reference_filter_mode}, ref_threshold={reference_filter_threshold}, ref_type={reference_filter_type}")

    with imageio.get_writer(str(out_path), fps=args.fps, codec="libx264", quality=8) as writer:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            frame = env.render()
            for _ in range(max(args.hold_reset_frames, 0)):
                writer.append_data(frame)

            ep_return = 0.0
            min_hand_dist = np.inf
            success_any = False
            ref_active = 0.0
            raw_to_ref = 0.0
            total_projection = 0.0
            steps = 0
            final_hand_dist = np.nan

            for step in range(args.max_steps_per_goal):
                key, subkey = jax.random.split(key)
                raw_action = np.asarray(agent.get_action(subkey, obs[None])[0])
                obs, reward, terminated, truncated, info = env.step(raw_action)
                ep_return += float(reward)
                steps = step + 1

                if step % max(args.render_every, 1) == 0:
                    writer.append_data(env.render())

                hand_dist = float(info.get("hand_dist", info.get("target_dist", np.nan)))
                if np.isfinite(hand_dist):
                    final_hand_dist = hand_dist
                    min_hand_dist = min(min_hand_dist, hand_dist)

                ref_active = max(ref_active, float(info.get("reference_correction_active", 0.0)))
                raw_to_ref = max(raw_to_ref, float(info.get("raw_to_reference_dist", 0.0)))
                total_projection += float(info.get("total_projection_residual", info.get("projection_residual", 0.0)))

                reached = (np.isfinite(hand_dist) and hand_dist < args.success_threshold) or float(info.get("is_success", 0.0)) > 0
                if reached:
                    success_any = True
                    hold_frame = env.render()
                    for _ in range(max(args.hold_success_frames, 0)):
                        writer.append_data(hold_frame)
                    break

                if terminated or truncated:
                    break

            min_hand_dist_print = min_hand_dist if np.isfinite(min_hand_dist) else np.nan
            print(
                f"[ep {ep}] return={ep_return:.3f}, steps={steps}, min_hand_dist={min_hand_dist_print:.4f}, "
                f"success_any={int(success_any)}, final_hand_dist={final_hand_dist:.4f}, "
                f"ref_active={ref_active:.1f}, raw_to_ref={raw_to_ref:.4f}, total_projection={total_projection:.4f}"
            )

    env.close()
    print(f"Video saved: {out_path}")


if __name__ == "__main__":
    main()
