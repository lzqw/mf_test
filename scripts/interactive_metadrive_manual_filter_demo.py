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
from relax.safety.metadrive_filtered_manual_policy import build_default_filter_info, rate_filter
from relax.safety.metadrive_sample_filter import SampleBasedVehicleSafetyFilter, SampleVehicleFilterConfig
from relax.safety.metadrive_mpc_cbf_filter import MPCVehicleCBFSafetyFilter, MPCVehicleCBFConfig


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
    p.add_argument("--filter_type", default="sample_vo", choices=["none", "rate", "sample_vo", "mpc_cbf"])
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

    p.add_argument("--max_dsteer", type=float, default=0.12)
    p.add_argument("--max_daccel", type=float, default=0.20)
    p.add_argument("--allowed_lane_change", type=int, default=1)
    p.add_argument("--lane_corridor_margin", type=float, default=0.30)
    p.add_argument("--max_abs_lateral_from_start_lane", type=float, default=5.8)
    p.add_argument("--vo_activation_distance", type=float, default=12.0)
    p.add_argument("--ttc_activation_threshold", type=float, default=3.0)
    p.add_argument("--min_closing_speed", type=float, default=0.5)
    p.add_argument("--num_maneuver_samples", type=int, default=32)
    p.add_argument("--show_status_panel", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--strict_filter_check", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--print_every", type=int, default=20)
    return p.parse_args()


def get_current_policy(env):
    base = env.unwrapped
    agent = getattr(base, "agent", None)
    agent_id = getattr(agent, "id", None)
    if agent_id is None:
        agent_id = "default_agent"
    return base.engine.get_policy(agent_id)


def attach_filter_to_active_policy(env, filt, args):
    policy = get_current_policy(env)

    if getattr(policy, "_safe_filter_is_patched", False):
        return policy

    original_act = policy.act

    policy._safe_filter_is_patched = True
    policy._safe_filter_original_act = original_act
    policy._safe_filter_prev_exec_action = np.zeros(2, dtype=np.float32)
    policy._safe_filter_last_raw_action = np.zeros(2, dtype=np.float32)
    policy._safe_filter_last_exec_action = np.zeros(2, dtype=np.float32)
    policy._safe_filter_last_info = {}
    policy._safe_filter_act_call_count = 0

    def filtered_act(*a, **kw):
        raw_action = np.asarray(original_act(*a, **kw), dtype=np.float32).reshape(2)

        if (not args.use_filter) or args.filter_type == "none":
            exec_action = np.clip(raw_action, -1.0, 1.0).astype(np.float32)
            filter_info = build_default_filter_info(raw_action, exec_action)
        elif args.filter_type == "rate":
            exec_action, filter_info = rate_filter(raw_action, policy._safe_filter_prev_exec_action)
        else:
            exec_action, filter_info = filt.project(
                raw_action,
                env=env.unwrapped,
                prev_exec_action=policy._safe_filter_prev_exec_action,
            )

        policy._safe_filter_act_call_count += 1
        policy._safe_filter_prev_exec_action = np.asarray(exec_action, dtype=np.float32).copy()
        policy._safe_filter_last_raw_action = raw_action.copy()
        policy._safe_filter_last_exec_action = policy._safe_filter_prev_exec_action.copy()

        filter_info = dict(filter_info)
        filter_info["policy_type"] = type(policy).__name__
        filter_info["act_call_count"] = policy._safe_filter_act_call_count
        filter_info["is_policy_patched"] = 1.0
        policy._safe_filter_last_info = filter_info

        return exec_action

    policy.act = filtered_act
    return policy


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
        max_dsteer=args.max_dsteer,
        max_daccel=args.max_daccel,
        allowed_lane_change=args.allowed_lane_change,
        lane_corridor_margin=args.lane_corridor_margin,
        max_abs_lateral_from_start_lane=args.max_abs_lateral_from_start_lane,
        vo_activation_distance=args.vo_activation_distance,
        ttc_activation_threshold=args.ttc_activation_threshold,
        min_closing_speed=args.min_closing_speed,
        num_maneuver_samples=args.num_maneuver_samples,
    )
    if args.filter_type == "mpc_cbf":
        filt = MPCVehicleCBFSafetyFilter(MPCVehicleCBFConfig(dt=args.dt, horizon=args.horizon))
    else:
        filt = SampleBasedVehicleSafetyFilter(cfg)

    env = gym.make(
        args.env_name,
        use_render=True,
        manual_control=True,
        controller="keyboard",
    )

    try:
        _, _ = env.reset(seed=args.seed)
        filt.reset()
        base = env.unwrapped
        agent_id = getattr(getattr(base, "agent", None), "id", "default_agent")
        policy = attach_filter_to_active_policy(env, filt, args)
        print(f"agent_id={agent_id}")
        print(f"active_policy_type={type(policy).__name__}")
        print(f"is_policy_patched={getattr(policy, '_safe_filter_is_patched', False)}")
        print(f"manual_control={getattr(base, 'manual_control', 'n/a')}")
        if not getattr(policy, "_safe_filter_is_patched", False):
            raise RuntimeError("Failed to patch active policy.")

        print("Click the MetaDrive 3D window and use WASD. If the car does not respond, press T once to toggle manual/expert mode.")

        step = 0
        while True:
            _, _, terminated, truncated, info = env.step([0.0, 0.0])
            env.render()
            step += 1

            policy = get_current_policy(env)
            filter_info = getattr(policy, "_safe_filter_last_info", {})
            raw = np.asarray(getattr(policy, "_safe_filter_last_raw_action", np.zeros(2)), dtype=np.float32)
            exec_action = np.asarray(getattr(policy, "_safe_filter_last_exec_action", np.zeros(2)), dtype=np.float32)
            act_call_count = int(getattr(policy, "_safe_filter_act_call_count", 0))
            speed = float(getattr(env.unwrapped.agent, "speed", 0.0))

            if args.strict_filter_check and step > 3 and args.use_filter and args.filter_type == "sample_vo":
                if act_call_count == 0:
                    raise RuntimeError("Filtered policy patch is not being called.")
                if int(filter_info.get("num_candidates", 0)) == 0:
                    raise RuntimeError("sample_vo filter is not running: num_candidates=0.")

            lines = [
                f"policy_type={type(policy).__name__}",
                f"is_policy_patched={getattr(policy, '_safe_filter_is_patched', False)}",
                f"act_call_count={act_call_count}",
                f"raw action={raw}",
                f"exec action={exec_action}",
                f"num_valid_candidates / num_candidates={filter_info.get('num_valid_candidates', 0)}/{filter_info.get('num_candidates', 0)}",
                f"selected_candidate_type={filter_info.get('selected_candidate_type', 'n/a')}",
                f"end_lateral_error_to_center={filter_info.get('end_lateral_error_to_center', 0):.3f}",
                f"heading_error={filter_info.get('heading_error', 0):.3f}",
                f"edge_penalty={filter_info.get('edge_penalty', 0):.3f}",
                f"active_maneuver_type={filter_info.get('active_maneuver_type', None)}",
                f"active_maneuver_steps_left={filter_info.get('active_maneuver_steps_left', 0)}",
                f"low_speed_count={filter_info.get('low_speed_count', 0)}",
                f"mpc_cbf_active={filter_info.get('mpc_cbf_active', 0):.1f}",
                f"mpc_success={filter_info.get('mpc_success', 0)} status={filter_info.get('mpc_status', 'n/a')}",
                f"min_pred_h_cbf={filter_info.get('min_pred_h_cbf', np.nan):.3f} min_pred_cbf={filter_info.get('min_pred_cbf', np.nan):.3f}",
                f"cbf_violation={filter_info.get('cbf_violation', 0):.1f}",
                f"mpc_steer={filter_info.get('mpc_steer', 0):.3f} mpc_alpha_min={filter_info.get('mpc_alpha_min', 0):.3f}",
                f"sign_s={filter_info.get('sign_s', 0):.1f}",
                f"filter_time_ms={filter_info.get('filter_time_ms', 0):.2f}",
                f"speed={speed:.3f}",
                f"crash={info.get('crash', 0)} cost={info.get('cost', 0)}",
            ]

            if step <= 50 or step % max(args.print_every, 1) == 0:
                print(" | ".join(lines))

            if status_panel_enabled and cv2 is not None:
                panel = make_status_panel(lines)
                cv2.imshow("Manual Control + Safety Filter", panel)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break

            if terminated or truncated:
                _, _ = env.reset()
                filt.reset()
                policy = attach_filter_to_active_policy(env, filt, args)
                if not getattr(policy, "_safe_filter_is_patched", False):
                    raise RuntimeError("Failed to patch active policy after reset.")

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
