import argparse
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_first(root, relative_patterns):
    root = Path(root)
    matches = []
    for pattern in relative_patterns:
        matches.extend(root.rglob(pattern))
    return sorted(matches)[-1] if matches else None


def load_latest_rollouts(root, tag):
    root = Path(root)
    candidates = []
    candidates += list(root.rglob(f"*{tag}*/eval_rollouts/**/*.npz"))
    candidates += list(root.rglob(f"*{tag}*/**/rollouts.npz"))
    if candidates:
        npz_path = sorted(candidates)[-1]
        return np.load(npz_path, allow_pickle=True)
    return None


def plot_scene(ax, trajectories, color, label):
    for traj in trajectories:
        if traj is None:
            continue
        tr = np.asarray(traj, dtype=np.float32)
        if tr.ndim != 2 or tr.shape[1] < 2 or tr.shape[0] < 2:
            continue
        ax.plot(tr[:, 0], tr[:, 1], color=color, alpha=0.3, lw=1.0)

    if label:
        ax.plot([], [], color=color, label=label)


def plot_pullback_scene(vanilla_npz, curvature_npz, outdir: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Double-integrator pullback scene")
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(-2.2, 2.2)
    obs = plt.Circle((0.0, 0.0), 0.8, fill=False, ls="--", lw=1.2)
    safe = plt.Circle((0.0, 0.0), 0.88, fill=False, ls=":", lw=1.2)
    ax.add_patch(obs)
    ax.add_patch(safe)

    ax.scatter([2.6], [0.0], c="tab:green", marker="*", s=110, label="goal")
    ax.scatter([-2.6], [0.0], c="black", marker="o", s=28, label="start")

    if vanilla_npz is not None and "positions" in vanilla_npz.files:
        trajs = vanilla_npz["positions"]
        plot_scene(ax, [trajs[i, : trajs.shape[1], :] for i in range(min(trajs.shape[0], 30))], "tab:blue", "Vanilla Flow")

    if curvature_npz is not None and "positions" in curvature_npz.files:
        trajs = curvature_npz["positions"]
        plot_scene(ax, [trajs[i, : trajs.shape[1], :] for i in range(min(trajs.shape[0], 30))], "tab:red", "Curvature-Shaped Flow")

    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.2)

    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / "fig_pullback_scene.png", dpi=220)
    fig.savefig(outdir / "fig_pullback_scene.pdf")
    plt.close(fig)


def make_distribution_plot(csv_path, outdir: Path):
    df = pd.read_csv(csv_path)

    fig, axs = plt.subplots(1, 4, figsize=(15, 3.5))
    methods = [
        "Nominal LQR",
        "Safety-shaped",
        "Robust-shaped",
        "Curvature-Shaped Flow",
    ]
    colors = ["tab:blue", "tab:green", "tab:purple", "tab:red"]

    for ax, method, color in zip(axs, methods, colors):
        sub = df[df["method"] == method]
        if sub.empty:
            ax.set_axis_off()
            continue
        ax.scatter(
            sub["normal_var"],
            sub["tangent_var"],
            s=35,
            alpha=0.8,
            label=method,
            color=color,
        )
        ax.set_title(method)
        ax.set_xlabel("Normal variance")
        ax.set_ylabel("Tangent variance")
        ax.grid(True, alpha=0.2)

    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / "fig_policy_distribution_geometry.png", dpi=220)
    fig.savefig(outdir / "fig_policy_distribution_geometry.pdf")
    plt.close(fig)

    rows = []
    for m in methods:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        rows.append(
            {
                "Method": m,
                "NormalVar": float(sub["normal_var"].mean()),
                "TangentVar": float(sub["tangent_var"].mean()),
                "NLR": float(sub["nlr"].mean()),
                "TCR": float(sub["tcr"].mean()),
            }
        )

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        "Method & Normal Var.$\\downarrow$ & Tangent Var. & NLR$\\downarrow$ & TCR$\\uparrow$ \\\\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['Method']} & {r['NormalVar']:.4f} & {r['TangentVar']:.4f} & {r['NLR']:.4f} & {r['TCR']:.4f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    (outdir / "table_distribution_geometry.tex").write_text("\n".join(lines), encoding="utf-8")


def plot_domain_shift(csv_path, outdir: Path):
    df = pd.read_csv(csv_path)

    vanilla = df[df["method"] == "Vanilla Flow"]
    curvature = df[df["method"] == "Curvature-Shaped Flow"]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    if not vanilla.empty:
        axs[0].plot(vanilla["delta"], vanilla["J_eval_mean"], marker="o", label="Vanilla Flow")
        axs[1].plot(vanilla["delta"], vanilla["violation_rate_mean"], marker="o", label="Vanilla Flow")
        axs[2].plot(vanilla["delta"], vanilla["h_min_mean"], marker="o", label="Vanilla Flow")
    if not curvature.empty:
        axs[0].plot(curvature["delta"], curvature["J_eval_mean"], marker="o", label="Curvature-Shaped Flow")
        axs[1].plot(curvature["delta"], curvature["violation_rate_mean"], marker="o", label="Curvature-Shaped Flow")
        axs[2].plot(curvature["delta"], curvature["h_min_mean"], marker="o", label="Curvature-Shaped Flow")

    axs[0].set_xlabel("delta")
    axs[0].set_ylabel("J_eval")
    axs[0].set_title("Cost")
    axs[1].set_xlabel("delta")
    axs[1].set_ylabel("violation rate")
    axs[1].set_title("Violation rate")
    axs[2].set_xlabel("delta")
    axs[2].set_ylabel("h_min")
    axs[2].set_title("h_min")

    for ax in axs:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / "fig_domain_shift_rollout.png", dpi=220)
    fig.savefig(outdir / "fig_domain_shift_rollout.pdf")
    plt.close(fig)

    selected = df[df["delta"].isin([0.0, 0.2, 0.4])].copy()
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"delta & Vanilla J$_{eval}$ & Curvature J$_{eval}$ & Vanilla h$_{min}$ / Curvature h$_{min}$ \\",
        r"\midrule",
    ]
    for dlt in [0.0, 0.2, 0.4]:
        sub = selected[selected["delta"] == dlt]
        if sub.empty:
            continue
        v = float(sub[sub["method"] == "Vanilla Flow"]["J_eval_mean"].iloc[0]) if not sub[sub["method"] == "Vanilla Flow"].empty else 0.0
        c = float(sub[sub["method"] == "Curvature-Shaped Flow"]["J_eval_mean"].iloc[0]) if not sub[sub["method"] == "Curvature-Shaped Flow"].empty else 0.0
        hv = float(sub[sub["method"] == "Vanilla Flow"]["h_min_mean"].iloc[0]) if not sub[sub["method"] == "Vanilla Flow"].empty else 0.0
        hc = float(sub[sub["method"] == "Curvature-Shaped Flow"]["h_min_mean"].iloc[0]) if not sub[sub["method"] == "Curvature-Shaped Flow"].empty else 0.0
        lines.append(f"{dlt:.1f} & {v:.4f} & {c:.4f} & {hv:.4f} / {hc:.4f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    (outdir / "table_domain_shift_rollout.tex").write_text("\n".join(lines), encoding="utf-8")

    # Copy to expected filename for downstream scripts.
    df.to_csv(outdir / "domain_shift_rollout.csv", index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", required=True)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    eval_dir = Path(args.eval_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    vanilla_rollout = load_latest_rollouts(eval_dir, "vanilla")
    curvature_rollout = load_latest_rollouts(eval_dir, "curvature")

    dist_csv = find_first(eval_dir, ["**/distribution_geometry.csv", "distribution/distribution_geometry.csv"])
    dom_csv = find_first(eval_dir, ["**/domain_shift_rollout.csv", "domain_shift/domain_shift_rollout.csv"])

    if dist_csv is None:
        dist_csv = find_first(eval_dir, ["**/distribution_geometry.csv"])
    if dom_csv is None:
        dom_csv = find_first(eval_dir, ["**/domain_shift_rollout.csv"])

    plot_pullback_scene(vanilla_rollout, curvature_rollout, outdir)

    if dist_csv is not None:
        make_distribution_plot(dist_csv, outdir)

    if dom_csv is not None:
        plot_domain_shift(dom_csv, outdir)


if __name__ == "__main__":
    main()
