#!/usr/bin/env python3
"""Stylized point-obstacle case-study visualization.

This is a stylized case-study visualization. It illustrates the qualitative
effect of safety-anisotropic FLAC-style reference energy. The dynamics-aware
version encourages boundary-following exploration near safety-critical regions
during early training, while converging to a clean safe trajectory later. The
generalization panels illustrate that the learned safety-aware behavior can
adapt to obstacle-size and obstacle-shape changes without retraining, whereas
a standard SAC-like policy may collide.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.patches import Circle

START = np.array([-2.7, 0.0])
GOAL = np.array([2.8, 0.0])
CENTER = np.array([0.0, 0.0])
OBSTACLE_RADIUS = 0.62
SAFETY_RADIUS = 0.78


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def obstacle_boundary_y(x: np.ndarray, radius: float, sign: float = 1.0) -> np.ndarray:
    dx = x - CENTER[0]
    inner = np.maximum(0.0, radius**2 - dx**2)
    return CENTER[1] + sign * np.sqrt(inner)


def draw_circle_obstacle(ax: plt.Axes, radius: float, safety_radius: float) -> None:
    ax.add_patch(Circle(CENTER, radius, facecolor="white", edgecolor="black", lw=2.0, zorder=1))
    ax.add_patch(
        Circle(CENTER, safety_radius, facecolor="none", edgecolor="gray", lw=1.4, ls="--", alpha=0.9, zorder=1)
    )


def draw_bumped_obstacle(ax: plt.Axes) -> None:
    main_c = np.array([0.0, 0.0])
    main_r = 0.62
    bump_c = np.array([0.0, 0.62])
    bump_r = 0.25
    inflate = 0.16

    for c, r in [(main_c, main_r), (bump_c, bump_r)]:
        ax.add_patch(Circle(c, r, facecolor="white", edgecolor="black", lw=2.0, zorder=1))
        ax.add_patch(Circle(c, r + inflate, facecolor="none", edgecolor="gray", lw=1.3, ls="--", alpha=0.9, zorder=1))


def _common_axis_style(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect("equal")
    ax.grid(True, ls="--", lw=0.55, alpha=0.45)
    ax.set_xlabel(r"$p_x$")
    ax.set_ylabel(r"$p_y$")
    ax.plot(START[0], START[1], marker="*", color="red", markersize=11, markeredgecolor="black", zorder=5)
    ax.plot(GOAL[0], GOAL[1], marker="o", color="green", markersize=8, markeredgecolor="black", zorder=5)


def make_filter_only_trajectory(progress: float, upper: bool, rng: np.random.Generator, safety_radius: float = SAFETY_RADIUS):
    sign = 1.0 if upper else -1.0
    x = np.linspace(START[0], GOAL[0], 220)
    gate = smoothstep(-1.25, -0.6, x) * (1.0 - smoothstep(0.55, 1.25, x))
    peak = sign * (safety_radius + 0.2)
    final_path = peak * np.exp(-((x - 0.0) ** 2) / (2.0 * 0.7**2))

    rough = rng.normal(0.0, 0.09 * (1.0 - progress), size=x.shape)
    low_freq = 0.12 * (1.0 - progress) * np.sin(2.8 * x + rng.uniform(-0.5, 0.5))
    early = final_path + sign * 0.25 * gate + rough + low_freq
    y = progress * final_path + (1.0 - progress) * early
    return x, y


def make_dynamics_aware_trajectory(progress: float, upper: bool, rng: np.random.Generator, safety_radius: float = SAFETY_RADIUS):
    sign = 1.0 if upper else -1.0
    x = np.linspace(START[0], GOAL[0], 240)
    final_peak = sign * (safety_radius + 0.2)
    final_path = final_peak * np.exp(-((x - 0.0) ** 2) / (2.0 * 0.68**2))

    approach_path = sign * 0.55 * np.exp(-((x - 0.0) ** 2) / (2.0 * 1.05**2))
    edge_path = obstacle_boundary_y(x, safety_radius, sign=sign) + sign * 0.055

    local_edge_gate = smoothstep(-1.35, -0.75, x) * (1.0 - smoothstep(0.75, 1.35, x))
    early_shape = (1.0 - local_edge_gate) * approach_path + local_edge_gate * edge_path

    small_noise = rng.normal(0.0, 0.04 * (1.0 - progress), size=x.shape)
    y = progress * final_path + (1.0 - progress) * early_shape + small_noise
    return x, y


def make_sac_collision_trajectory(rng: np.random.Generator, drift: float = 0.0):
    x = np.linspace(START[0], GOAL[0], 160)
    y = 0.03 * np.sin(1.2 * x + 0.4) + drift + rng.normal(0.0, 0.02, size=x.shape)
    return x, y


def make_test_trajectory(kind: str, method: str, upper: bool, rng: np.random.Generator):
    if method == "sac":
        return make_sac_collision_trajectory(rng, drift=0.02 if upper else -0.02)

    if kind == "base":
        sr = SAFETY_RADIUS
    elif kind == "larger":
        sr = 1.02
    else:
        sr = 0.86

    if method == "filter":
        x, y = make_filter_only_trajectory(0.93, upper=upper, rng=rng, safety_radius=sr)
        y += (0.1 if upper else -0.1) * (1.0 if kind != "base" else 0.5)
        return x, y

    x, y = make_dynamics_aware_trajectory(0.95, upper=upper, rng=rng, safety_radius=sr)
    y += (0.06 if upper else -0.06) * (1.0 if kind == "larger" else 0.4)
    return x, y


def first_collision_idx(x: np.ndarray, y: np.ndarray, circles: list[tuple[np.ndarray, float]]) -> int | None:
    for i in range(len(x)):
        p = np.array([x[i], y[i]])
        for c, r in circles:
            if np.linalg.norm(p - c) <= r:
                return i
    return None


def plot_training_samples(out_dir: Path, formats: list[str], seed: int):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    progress_list = np.linspace(0.05, 0.98, 14)

    for ax, title in zip(
        axes,
        [
            "Panel A: Filter-only anisotropic reference energy",
            "Panel B: Dynamics-aware anisotropic reference energy",
        ],
    ):
        _common_axis_style(ax, title)
        draw_circle_obstacle(ax, OBSTACLE_RADIUS, SAFETY_RADIUS)

    for p in progress_list:
        c_top = cm.Reds(0.25 + 0.7 * p)
        c_bot = cm.Blues(0.25 + 0.7 * p)
        x1, y1 = make_filter_only_trajectory(float(p), upper=True, rng=rng)
        x2, y2 = make_filter_only_trajectory(float(p), upper=False, rng=rng)
        axes[0].plot(x1, y1, color=c_top, lw=1.5)
        axes[0].plot(x2, y2, color=c_bot, lw=1.5)

        x3, y3 = make_dynamics_aware_trajectory(float(p), upper=True, rng=rng)
        x4, y4 = make_dynamics_aware_trajectory(float(p), upper=False, rng=rng)
        axes[1].plot(x3, y3, color=c_top, lw=1.5)
        axes[1].plot(x4, y4, color=c_bot, lw=1.5)

    fig.suptitle(
        "Dynamics-aware anisotropic reference energy encourages early safety-boundary exploration,\n"
        "but finally converges to a near-optimal safe trajectory.",
        fontsize=12,
    )

    for ext in formats:
        fig.savefig(out_dir / f"point_obstacle_training_samples.{ext}", dpi=220)
    plt.close(fig)


def plot_generalization_test(out_dir: Path, formats: list[str], seed: int):
    rng = np.random.default_rng(seed + 17)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    titles = ["Test 1: training obstacle", "Test 2: larger obstacle", "Test 3: obstacle with a small top bump"]
    kinds = ["base", "larger", "bump"]

    for ax, t, k in zip(axes, titles, kinds):
        _common_axis_style(ax, t)
        if k == "base":
            draw_circle_obstacle(ax, OBSTACLE_RADIUS, SAFETY_RADIUS)
            circles = [(CENTER, OBSTACLE_RADIUS)]
        elif k == "larger":
            draw_circle_obstacle(ax, 0.85, 1.02)
            circles = [(CENTER, 0.85)]
        else:
            draw_bumped_obstacle(ax)
            circles = [(np.array([0.0, 0.0]), 0.62), (np.array([0.0, 0.62]), 0.25)]

        xs, ys = make_test_trajectory(k, "sac", upper=True, rng=rng)
        idx = first_collision_idx(xs, ys, circles)
        if idx is None:
            idx = len(xs) - 1
        ax.plot(xs[: idx + 1], ys[: idx + 1], color="0.25", ls="--", lw=2.0, label="SAC")
        ax.plot(xs[idx], ys[idx], marker="x", color="0.1", ms=9, mew=2)

        xf, yf = make_test_trajectory(k, "filter", upper=True, rng=rng)
        ax.plot(xf, yf, color="#3b6bd6", lw=2.3, label="Filter-only")
        xo, yo = make_test_trajectory(k, "ours", upper=False, rng=rng)
        ax.plot(xo, yo, color="#d43f3a", lw=2.3, label="Ours")

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="lower left", frameon=True)
    fig.suptitle("No retraining under obstacle-shape changes", fontsize=13)

    for ext in formats:
        fig.savefig(out_dir / f"point_obstacle_generalization_test.{ext}", dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot stylized point-obstacle case-study figures.")
    parser.add_argument("--out-dir", type=Path, default=Path("figures/case_study"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_training_samples(args.out_dir, args.formats, seed=args.seed)
    plot_generalization_test(args.out_dir, args.formats, seed=args.seed)
    print(f"Saved case-study figures to: {args.out_dir}")


if __name__ == "__main__":
    main()
