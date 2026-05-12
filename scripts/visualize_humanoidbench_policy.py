import argparse
import json
from pathlib import Path

from relax.safety.humanoidbench_filter import HumanoidSafeFilterConfig


def load_train_args(log_dir: str):
    args_path = Path(log_dir) / "args.json"
    data = json.loads(args_path.read_text())

    class Args:
        pass

    args = Args()
    for k, v in data.items():
        setattr(args, k, v)
    return args


def build_filter_cfg(train_args):
    return HumanoidSafeFilterConfig(
        residual_radius=getattr(train_args, "residual_radius", 0.35),
        smooth_radius=getattr(train_args, "smooth_radius", 0.25),
        max_delta=getattr(train_args, "max_delta", 0.1),
        target_step_radius=getattr(train_args, "target_step_radius", 0.08),
        reachable_radius=getattr(train_args, "reachable_radius", 0.45),
        z_min_safe=getattr(train_args, "z_min_safe", 0.4),
        z_max_safe=getattr(train_args, "z_max_safe", 1.8),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", type=str, required=True)
    args = p.parse_args()
    train_args = load_train_args(args.log_dir)
    filter_cfg = build_filter_cfg(train_args)
    print("Loaded filter config:", filter_cfg)


if __name__ == "__main__":
    main()
