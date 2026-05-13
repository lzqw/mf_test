from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import os
import time

import gymnasium as gym
import numpy as np

import relax.env.drive.lane_change  # noqa: F401
from relax.safety.metadrive_filtered_manual_policy import FilteredManualControlPolicy
from relax.safety.metadrive_sample_filter import SampleVehicleFilterConfig


def make_status_panel(lines, width=860, height=420):
    import cv2

    panel = np.zeros((height, width, 3), dtype=np.uint8)
    y = 25
    for line in lines:
        cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
    return panel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env_name", default="FlatThreeLaneStraight")
    p.add_argument("--filter_type", default="sample_vo", choices=["none", "rate", "sample_vo"])
    p.add_argument("--use_filter", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_local_samples", type=int, default=64)
    p.add_argument("--num_prev_samples", type=int, default=32)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--safe_radius", type=float, default=4.0)
    p.add_argument("--ttc_min", type=float, default=1.5)
    p.add_argument("--h_vo_margin", type=float, default=0.2)
    p.add_argument("--lane_margin_min", type=float, default=0.3)
    p.add_argument("--show_status_panel", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--print_every", type=int, default=20)
    return p.parse_args()


def get_current_policy(env):
    base = env.unwrapped
    agent = getattr(base, "agent", None)
    agent_id = getattr(agent, "id", None)
    if agent_id is None:
        agent_id = "default_agent"
    return base.engine.get_policy(agent_id)


def main():
    args = parse_args()

    cv2 = None
    status_panel_enabled = args.show_status_panel
    if status_panel_enabled:
        if os.environ.get("DISPLAY", "") == "":
            print("DISPLAY is missing; disabling OpenCV status panel.")
            status_panel_enabled = False
        else:
            try:
                import cv2  # type: ignore
            except Exception as exc:
                print(f"OpenCV unavailable; disabling status panel: {exc}")
                status_panel_enabled = False

    cfg = SampleVehicleFilterConfig(
        num_local_samples=args.num_local_samples,
        num_prev_samples=args.num_prev_samples,
        horizon=args.horizon,
        dt=args.dt,
        safe_radius=args.safe_radius,
        ttc_min=args.ttc_min,
        h_vo_margin=args.h_vo_margin,
        lane_margin_min=args.lane_margin_min,
    )

    FilteredManualControlPolicy.configure(
        filter_cfg=cfg,
        filter_type=args.filter_type,
        use_filter=args.use_filter,
        env_ref=None,
    )

    env = gym.make(
        args.env_name,
        use_render=True,
        manual_control=True,
        controller="keyboard",
        agent_policy=FilteredManualControlPolicy,
    )
    FilteredManualControlPolicy.env_ref = env.unwrapped

    try:
        _, _ = env.reset(seed=args.seed)
        policy = get_current_policy(env)
        if hasattr(policy, "reset_filter_state"):
            policy.reset_filter_state()
        if hasattr(policy, "expert_takeover"):
            policy.expert_takeover = False
        if hasattr(env.unwrapped.agent, "expert_takeover"):
            env.unwrapped.agent.expert_takeover = False

        print("Click the MetaDrive 3D window and use WASD. If the car does not respond, press T once to toggle manual/expert mode.")

        step = 0
        startup_warned = False
        while True:
            _, _, terminated, truncated, info = env.step([0.0, 0.0])
            env.render()
            step += 1

            policy = get_current_policy(env)
            filter_info = getattr(policy, "last_filter_info", {})
            raw = np.asarray(getattr(policy, "last_raw_action", np.zeros(2)), dtype=np.float32)
            exec_action = np.asarray(getattr(policy, "last_exec_action", np.zeros(2)), dtype=np.float32)
            speed = float(getattr(env.unwrapped.agent, "speed", 0.0))

            if (not startup_warned) and step <= 50 and float(np.linalg.norm(raw)) > 1e-4:
                print("Warning: ManualControlPolicy is outputting nonzero raw action at startup. Check expert/takeover mode or stuck key state.")
                startup_warned = True

            if step <= 50:
                print(
                    f"[step {step:03d}] raw_action={raw} exec_action={exec_action} "
                    f"filter_active={filter_info.get('filter_active', 0)} "
                    f"projection_residual={filter_info.get('projection_residual', 0):.4f} "
                    f"selected_candidate_type={filter_info.get('selected_candidate_type', 'n/a')} "
                    f"speed={speed:.3f}"
                )

            lines = [
                f"raw action={raw}",
                f"exec action={exec_action}",
                f"filter_type={args.filter_type} use_filter={args.use_filter}",
                f"filter_active={filter_info.get('filter_active', 0)}",
                f"projection_residual={filter_info.get('projection_residual', 0):.3f}",
                f"num_valid_candidates / num_candidates={filter_info.get('num_valid_candidates', 0)}/{filter_info.get('num_candidates', 0)}",
                f"valid_candidate_ratio={filter_info.get('valid_candidate_ratio', 0):.3f}",
                f"fallback={filter_info.get('fallback', 0)}",
                f"no_safe_candidate={filter_info.get('no_safe_candidate', 0)}",
                f"min_pred_ttc={filter_info.get('min_pred_ttc', np.inf):.3f}",
                f"min_pred_h_vo={filter_info.get('min_pred_h_vo', np.inf):.3f}",
                f"min_pred_dist={filter_info.get('min_pred_dist', np.inf):.3f}",
                f"selected_candidate_type={filter_info.get('selected_candidate_type', 'n/a')}",
                f"selected_is_maneuver={filter_info.get('selected_is_maneuver', 0)}",
                f"predicted_opposite_lane={filter_info.get('predicted_opposite_lane', 0)}",
                f"min_corridor_margin={filter_info.get('min_corridor_margin', np.inf):.3f}",
                f"max_abs_lateral={filter_info.get('max_abs_lateral', 0):.3f}",
                f"longitudinal_progress={filter_info.get('longitudinal_progress', 0):.3f}",
                f"pass_obstacle_bonus={filter_info.get('pass_obstacle_bonus', 0):.3f}",
                f"filter_time_ms={filter_info.get('filter_time_ms', 0):.2f}",
                f"speed={speed:.3f}",
                f"crash={info.get('crash', 0)}",
                f"out_of_road={info.get('out_of_road', 0)}",
                f"cost={info.get('cost', 0)}",
                f"is_success={info.get('is_success', 0)}",
            ]

            if status_panel_enabled and cv2 is not None:
                panel = make_status_panel(lines)
                cv2.imshow("Manual Control + Safety Filter", panel)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
            elif step % max(args.print_every, 1) == 0:
                print(" | ".join(lines))

            if terminated or truncated:
                _, _ = env.reset()
                policy = get_current_policy(env)
                if hasattr(policy, "reset_filter_state"):
                    policy.reset_filter_state()
                if hasattr(policy, "expert_takeover"):
                    policy.expert_takeover = False
                if hasattr(env.unwrapped.agent, "expert_takeover"):
                    env.unwrapped.agent.expert_takeover = False

            if not status_panel_enabled:
                time.sleep(0.001)
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting cleanly.")
    finally:
        env.close()
        if status_panel_enabled and cv2 is not None:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
