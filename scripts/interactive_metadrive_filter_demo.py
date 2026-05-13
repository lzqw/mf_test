from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import os
import numpy as np
import relax.env.drive.lane_change  # noqa: F401
from envs.metadrive_safe_wrapper import SafeMetaDriveSampleWrapper
from relax.safety.metadrive_sample_filter import SampleVehicleFilterConfig


def render_native(env):
    try:
        return env.unwrapped.render()
    except TypeError:
        try:
            return env.unwrapped.render(mode="human")
        except TypeError:
            return env.render()


def make_status_panel(lines, width=760, height=280):
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
    p.add_argument("--no_status_panel", action="store_true")
    p.add_argument("--render_3d", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args = parse_args()

    status_panel_enabled = not args.no_status_panel
    display_missing = os.environ.get("DISPLAY", "") == ""

    if display_missing and args.render_3d:
        print("Warning: DISPLAY not set. MetaDrive 3D rendering may fail depending on your setup.")

    if args.no_status_panel:
        print(
            "Keyboard input requires the OpenCV status panel. Without it, this script will only render MetaDrive and print metrics."
        )

    cv2 = None
    if status_panel_enabled:
        try:
            import cv2  # type: ignore
        except Exception as e:
            print(f"OpenCV is required for the keyboard/status panel: {e}")
            return 1
        if display_missing:
            print("OpenCV keyboard/status panel requires a display or X server.")
            return 1

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
    env = SafeMetaDriveSampleWrapper(
        args.env_name,
        use_filter=args.use_filter,
        filter_type=args.filter_type,
        filter_cfg=cfg,
        use_render=True,
    )

    env.reset(seed=args.seed)
    raw = np.zeros(2, dtype=np.float32)
    step = 0
    term = False
    trunc = False

    while True:
        key = 255
        if status_panel_enabled:
            key = cv2.waitKey(30) & 0xFF

        raw *= 0.90
        if key in [ord("a"), ord("A")]:
            raw[0] -= 0.08
        if key in [ord("d"), ord("D")]:
            raw[0] += 0.08
        if key in [ord("w"), ord("W")]:
            raw[1] += 0.10
        if key in [ord("s"), ord("S")]:
            raw[1] -= 0.10
        if key == 32:
            raw[1] = -1.0
        if key in [ord("x"), ord("X")]:
            raw[:] = 0.0
        if key in [ord("f"), ord("F")]:
            env.use_filter = not env.use_filter
        if key == ord("1"):
            env.filter_type = "none"
            env.use_filter = False
        if key == ord("2"):
            env.filter_type = "rate"
            env.use_filter = True
        if key == ord("3"):
            env.filter_type = "sample_vo"
            env.use_filter = True
        if key in [ord("r"), ord("R")] or term or trunc:
            env.reset()
            raw[:] = 0.0
        if key in [ord("q"), ord("Q"), 27]:
            break

        raw = np.clip(raw, -1.0, 1.0)
        _, _, term, trunc, info = env.step(raw)
        step += 1

        if args.render_3d:
            render_native(env)

        lines = [
            f"raw action={raw}",
            f"executed action={info.get('exec_action')}",
            f"filter_type={env.filter_type} use_filter={env.use_filter}",
            f"filter_active={info.get('filter_active', 0)}",
            f"projection_residual={info.get('projection_residual', 0):.3f}",
            f"num_valid_candidates / num_candidates={info.get('num_valid_candidates', 0)}/{info.get('num_candidates', 0)}",
            f"valid_candidate_ratio={info.get('valid_candidate_ratio', 0):.2f}",
            f"fallback={info.get('fallback', 0)}",
            f"no_safe_candidate={info.get('no_safe_candidate', 0)}",
            f"min_pred_ttc={info.get('min_pred_ttc', np.inf):.3f}",
            f"min_pred_h_vo={info.get('min_pred_h_vo', np.inf):.3f}",
            f"min_pred_dist={info.get('min_pred_dist', np.inf):.3f}",
            f"filter_time_ms={info.get('filter_time_ms', 0):.2f}",
            f"crash={info.get('crash', 0)}",
            f"out_of_road={info.get('out_of_road', 0)}",
            f"cost={info.get('cost', 0)}",
            f"is_success={info.get('is_success', 0)}",
        ]

        if status_panel_enabled:
            panel = make_status_panel(lines)
            cv2.imshow("Filter Status / Keyboard Control", panel)

        if step % 20 == 0:
            print(lines)

    env.close()
    if status_panel_enabled:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
