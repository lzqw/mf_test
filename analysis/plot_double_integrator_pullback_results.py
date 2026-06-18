import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import numpy as np
import pandas as pd

from relax.utils.pullback_geometry import local_normal_tangent, to_normal_tangent_frame


plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def require_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"missing required {label}: {path}")
    return path


def _load_rollout_npz(path: Path):
    return np.load(path, allow_pickle=True)


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _valid_positions(npz_data, limit=None):
    positions = np.asarray(npz_data["positions"], dtype=np.float32)
    valid_lengths = np.asarray(npz_data["valid_lengths"], dtype=np.int32)
    total = positions.shape[0] if limit is None else min(int(limit), positions.shape[0])
    out = []
    for i in range(total):
        out.append(positions[i, : int(valid_lengths[i]), :])
    return out


def _representative_index(npz_data):
    valid_lengths = np.asarray(npz_data["valid_lengths"], dtype=np.int32)
    success = np.asarray(npz_data["is_success"], dtype=bool) if "is_success" in npz_data.files else np.ones_like(valid_lengths, dtype=bool)
    candidates = np.where(success)[0]
    if candidates.size == 0:
        candidates = np.arange(valid_lengths.shape[0])
    cand_lengths = valid_lengths[candidates]
    order = np.argsort(cand_lengths)
    return int(candidates[order[len(order) // 2]])


def _extract_workspace(eval_dir: Path):
    cfg = _load_json(require_path(eval_dir / "env_config.json", "env_config.json"))
    start_x = float(cfg.get("start_x", -2.6))
    start_y_range = float(cfg.get("start_y_range", 0.45))
    goal_x = float(cfg.get("goal_x", 2.6))
    goal_y = float(cfg.get("goal_y", 0.0))
    obstacle_center = np.asarray(cfg.get("obstacle_center", [0.0, 0.0]), dtype=np.float64)
    obstacle_radius = float(cfg.get("obstacle_radius", 0.8))
    eps_obs = float(cfg.get("eps_obs", 0.08))
    return {
        "start_x": start_x,
        "start_y_range": start_y_range,
        "goal": np.array([goal_x, goal_y], dtype=np.float64),
        "obstacle_center": obstacle_center,
        "obstacle_radius": obstacle_radius,
        "eps_obs": eps_obs,
    }


def _plot_workspace(ax, workspace):
    obstacle = Circle(tuple(workspace["obstacle_center"]), workspace["obstacle_radius"], fill=False, lw=1.2, ec="0.2")
    safe = Circle(
        tuple(workspace["obstacle_center"]),
        workspace["obstacle_radius"] + workspace["eps_obs"],
        fill=False,
        lw=1.2,
        ls="--",
        ec="0.55",
    )
    ax.add_patch(obstacle)
    ax.add_patch(safe)
    ax.scatter(
        [workspace["start_x"]],
        [0.0],
        c="k",
        marker="*",
        s=55,
        label="Start",
        zorder=5,
    )
    ax.scatter(
        [workspace["goal"][0]],
        [workspace["goal"][1]],
        c="tab:green",
        marker="o",
        s=35,
        label="Goal",
        zorder=5,
    )
    ax.plot([], [], color="0.2", lw=1.2, label="Obstacle")
    ax.plot([], [], color="0.55", lw=1.2, ls="--", label="Safety buffer")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-2.0, 2.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.18)


def _plot_traj_bundle(ax, npz_data, color, label, limit=30):
    trajectories = _valid_positions(npz_data, limit=limit)
    for traj in trajectories:
        ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.18, lw=1.1)
    ridx = _representative_index(npz_data)
    rep = _valid_positions(npz_data, limit=npz_data["positions"].shape[0])[ridx]
    ax.plot(rep[:, 0], rep[:, 1], color=color, alpha=1.0, lw=2.0, label=label, zorder=4)
    return ridx, rep


def _critical_state_marker(npz_data, rep_idx, workspace):
    pos = np.asarray(npz_data["positions"][rep_idx], dtype=np.float64)
    valid = int(np.asarray(npz_data["valid_lengths"], dtype=np.int32)[rep_idx])
    traj = pos[:valid]
    center = workspace["obstacle_center"]
    clearance = np.linalg.norm(traj - center[None, :], axis=1) - workspace["obstacle_radius"]
    state_idx = int(np.argmin(clearance))
    critical = traj[state_idx]
    normal, tangent = local_normal_tangent(critical, center)
    return critical, normal, tangent


def plot_pullback_scene(
    nominal_vanilla_npz,
    nominal_curvature_npz,
    shifted_vanilla_npz,
    shifted_curvature_npz,
    workspace,
    outdir: Path,
):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)

    panels = [
        ("(a) Nominal rollouts, $\\delta=0$", nominal_vanilla_npz, nominal_curvature_npz),
        ("(b) Shifted rollouts, $\\delta=0.4$", shifted_vanilla_npz, shifted_curvature_npz),
    ]
    crit_drawn = False
    for ax, (title, vanilla_npz, curvature_npz) in zip(axs, panels):
        _plot_workspace(ax, workspace)
        ax.set_title(title)
        _, _ = _plot_traj_bundle(ax, vanilla_npz, "tab:blue", "Vanilla Flow")
        rep_idx, _ = _plot_traj_bundle(ax, curvature_npz, "tab:red", "Curvature-Shaped Flow")
        if not crit_drawn:
            critical, normal, tangent = _critical_state_marker(curvature_npz, rep_idx, workspace)
            ax.scatter([critical[0]], [critical[1]], c="k", marker="x", s=30, zorder=6)
            ax.arrow(
                critical[0],
                critical[1],
                0.28 * normal[0],
                0.28 * normal[1],
                width=0.01,
                head_width=0.06,
                head_length=0.08,
                color="k",
                length_includes_head=True,
                zorder=6,
            )
            ax.arrow(
                critical[0],
                critical[1],
                0.28 * tangent[0],
                0.28 * tangent[1],
                width=0.01,
                head_width=0.06,
                head_length=0.08,
                color="0.35",
                length_includes_head=True,
                zorder=6,
            )
            ax.text(critical[0] + 0.05, critical[1] + 0.18, "n", fontsize=7)
            ax.text(critical[0] - 0.16, critical[1] + 0.12, "t", fontsize=7, color="0.35")
            crit_drawn = True

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.savefig(outdir / "fig_pullback_scene.png", dpi=240, bbox_inches="tight")
    fig.savefig(outdir / "fig_pullback_scene.pdf", bbox_inches="tight")
    plt.close(fig)


def _local_covariance_from_world(sigma_world, pos, center):
    normal, tangent = local_normal_tangent(pos, center)
    basis = np.stack([normal, tangent], axis=1)
    cov_local = basis.T @ np.asarray(sigma_world, dtype=np.float64) @ basis
    return 0.5 * (cov_local + cov_local.T)


def _ellipse_from_cov(ax, cov, color, lw=1.6, ls="-", alpha=1.0):
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-10, None)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width = 2.0 * np.sqrt(eigvals[0])
    height = 2.0 * np.sqrt(eigvals[1])
    ell = Ellipse((0.0, 0.0), width=width, height=height, angle=angle, fill=False, lw=lw, ls=ls, ec=color, alpha=alpha)
    ax.add_patch(ell)


def _choose_distribution_state(dist_npz):
    samples = list(np.asarray(dist_npz["samples"], dtype=object))
    items = [s.item() if hasattr(s, "item") else s for s in samples]
    items = sorted(items, key=lambda d: float(d["clearance"]))
    return items[0]


def make_distribution_plot(dist_csv_path: Path, dist_npz_path: Path, outdir: Path):
    df = pd.read_csv(dist_csv_path)
    df.to_csv(outdir / "distribution_geometry.csv", index=False)

    dist_npz = np.load(dist_npz_path, allow_pickle=True)
    sample = _choose_distribution_state(dist_npz)
    pos = np.array([float(sample["px"]), float(sample["py"])], dtype=np.float64)
    center = np.zeros(2, dtype=np.float64)
    normal, tangent = local_normal_tangent(pos, center)

    nominal_local = _local_covariance_from_world(sample["Sigma_nominal"], pos, center)
    safety_local = _local_covariance_from_world(sample["Sigma_safe"], pos, center)
    robust_local = _local_covariance_from_world(sample["Sigma_robust"], pos, center)

    vanilla_local = to_normal_tangent_frame(sample["actions_vanilla"], normal, tangent)
    curvature_local = to_normal_tangent_frame(sample["actions_curvature"], normal, tangent)
    vanilla_cov_local = np.cov(vanilla_local.T, bias=False)
    curvature_cov_local = np.cov(curvature_local.T, bias=False)

    rng = np.random.default_rng(0)
    synthetic = {
        "Nominal": rng.multivariate_normal(np.zeros(2), nominal_local, size=512),
        "Safety-shaped": rng.multivariate_normal(np.zeros(2), safety_local, size=512),
        "Robust-shaped": rng.multivariate_normal(np.zeros(2), robust_local, size=512),
    }

    extent = np.concatenate(
        [
            synthetic["Nominal"],
            synthetic["Safety-shaped"],
            synthetic["Robust-shaped"],
            vanilla_local,
            curvature_local,
        ],
        axis=0,
    )
    lim = float(np.quantile(np.abs(extent), 0.995))
    lim = max(lim, 0.25)

    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.4), constrained_layout=True)
    panel_data = [
        ("(a) Nominal", synthetic["Nominal"], nominal_local, "tab:blue"),
        ("(b) Safety-shaped", synthetic["Safety-shaped"], safety_local, "tab:green"),
        ("(c) Robust-shaped", synthetic["Robust-shaped"], robust_local, "tab:purple"),
        ("(d) Sampled Flow", curvature_local, curvature_cov_local, "tab:red"),
    ]

    for ax, (title, points, cov, color) in zip(axs, panel_data):
        ax.axhline(0.0, color="0.75", lw=0.8)
        ax.axvline(0.0, color="0.75", lw=0.8)
        if title == "(d) Sampled Flow":
            ax.scatter(vanilla_local[:, 0], vanilla_local[:, 1], s=6, alpha=0.10, c="tab:blue", label="Vanilla raw")
            ax.scatter(points[:, 0], points[:, 1], s=6, alpha=0.18, c=color, label="Curvature raw")
            _ellipse_from_cov(ax, vanilla_cov_local, "tab:blue", lw=1.2, ls="--", alpha=0.9)
            _ellipse_from_cov(ax, cov, color, lw=1.6, ls="-", alpha=1.0)
        else:
            ax.scatter(points[:, 0], points[:, 1], s=6, alpha=0.12, c=color)
            _ellipse_from_cov(ax, cov, color, lw=1.6)
        ax.set_title(title)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.18)
        ax.set_xlabel("Normal action")
        if ax is axs[0]:
            ax.set_ylabel("Tangent action")

    handles, labels = axs[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.83, 1.06))
    fig.savefig(outdir / "fig_policy_distribution_geometry.png", dpi=240, bbox_inches="tight")
    fig.savefig(outdir / "fig_policy_distribution_geometry.pdf", bbox_inches="tight")
    plt.close(fig)

    method_order = [
        "Nominal LQR",
        "Safety-shaped",
        "Robust-shaped",
        "Vanilla Flow",
        "Curvature-Shaped Flow",
    ]
    label_map = {
        "Nominal LQR": "Nominal",
        "Safety-shaped": "Safety-shaped",
        "Robust-shaped": "Robust-shaped",
        "Vanilla Flow": "Vanilla Flow",
        "Curvature-Shaped Flow": "Curvature-Shaped Flow",
    }
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Normal Var.$\downarrow$ & Tangent Var. & NLR$\downarrow$ & TCR$\uparrow$ \\",
        r"\midrule",
    ]
    for method in method_order:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        lines.append(
            f"{label_map[method]} & "
            f"{sub['normal_var'].mean():.4f} $\\pm$ {sub['normal_var'].std(ddof=0):.4f} & "
            f"{sub['tangent_var'].mean():.4f} $\\pm$ {sub['tangent_var'].std(ddof=0):.4f} & "
            f"{sub['nlr'].mean():.4f} $\\pm$ {sub['nlr'].std(ddof=0):.4f} & "
            f"{sub['tcr'].mean():.4f} $\\pm$ {sub['tcr'].std(ddof=0):.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (outdir / "table_distribution_geometry.tex").write_text("\n".join(lines), encoding="utf-8")


def plot_domain_shift(csv_path: Path, outdir: Path):
    df = pd.read_csv(csv_path)
    df.to_csv(outdir / "domain_shift_rollout.csv", index=False)

    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
    methods = [("Vanilla Flow", "tab:blue"), ("Curvature-Shaped Flow", "tab:red")]
    metric_specs = [
        ("J_eval_mean", "J_eval_std", r"(a) $J_{\rm eval}$ vs $\delta$", r"$J_{\rm eval}$"),
        ("violation_rate_mean", "violation_rate_std", r"(b) Violation rate vs $\delta$", "Violation rate"),
        ("h_min_mean", "h_min_std", r"(c) $h_{\min}$ vs $\delta$", r"$h_{\min}$"),
    ]

    for ax, (mean_col, std_col, title, ylabel) in zip(axs, metric_specs):
        for method, color in methods:
            sub = df[df["method"] == method].sort_values("delta")
            ax.errorbar(
                sub["delta"],
                sub[mean_col],
                yerr=sub[std_col] if std_col in sub else None,
                marker="o",
                lw=1.4,
                ms=3.5,
                capsize=2.0,
                color=color,
                label=method,
            )
        if mean_col == "h_min_mean":
            ax.axhline(0.0, color="0.35", lw=1.0, ls="--")
        ax.set_title(title)
        ax.set_xlabel(r"$\delta$")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.18)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.savefig(outdir / "fig_domain_shift_rollout.png", dpi=240, bbox_inches="tight")
    fig.savefig(outdir / "fig_domain_shift_rollout.pdf", bbox_inches="tight")
    plt.close(fig)

    lines = [
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"$\delta$ & Method & $J_{\rm eval}\downarrow$ & Viol.$\downarrow$ & $h_{\min}\uparrow$ \\",
        r"\midrule",
    ]
    for delta in [0.0, 0.2, 0.4]:
        sub = df[np.isclose(df["delta"], delta)]
        for method in ["Vanilla Flow", "Curvature-Shaped Flow"]:
            row = sub[sub["method"] == method]
            if row.empty:
                continue
            row = row.iloc[0]
            label = "Vanilla" if method == "Vanilla Flow" else "Ours"
            lines.append(
                f"{delta:.1f} & {label} & {row['J_eval_mean']:.4f} & "
                f"{row['violation_rate_mean']:.4f} & {row['h_min_mean']:.4f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (outdir / "table_domain_shift_rollout.tex").write_text("\n".join(lines), encoding="utf-8")


def write_latex_snippets(outdir: Path):
    text = r"""\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/fig_pullback_scene.pdf}
\caption{Circular-obstacle pullback environment and representative rollouts. The agent moves from the start region to the goal while avoiding the circular obstacle. The curvature-shaped flow policy preserves tangent motion around the safety boundary and remains less sensitive to actuator-gain shifts.}
\label{fig:pullback_scene}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/fig_policy_distribution_geometry.pdf}
\caption{Local inverse-curvature policy distributions near the obstacle boundary. Safety curvature suppresses obstacle-normal stochasticity while preserving tangent stochasticity. The sampled curvature-shaped flow policy follows the same anisotropic structure.}
\label{fig:policy_distribution_geometry}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/fig_domain_shift_rollout.pdf}
\caption{Robust rollout under structured actuator-gain domain shift. The curvature-shaped flow policy degrades more gracefully as the shift magnitude increases, maintaining lower violation rate and larger minimum safety margin.}
\label{fig:domain_shift_rollout}
\end{figure}

\input{tables/table_distribution_geometry}
\input{tables/table_domain_shift_rollout}
"""
    (outdir / "latex_snippets.tex").write_text(text, encoding="utf-8")


def _resolve_trial_inputs(trial_dir: Path):
    eval_curvature_main = require_path(trial_dir / "eval_curvature_main" / "rollouts.npz", "curvature nominal rollouts")
    eval_vanilla_main = require_path(trial_dir / "eval_vanilla_main" / "rollouts.npz", "vanilla nominal rollouts")
    dist_csv = require_path(trial_dir / "distribution_main" / "distribution_geometry.csv", "distribution_geometry.csv")
    dist_npz = require_path(trial_dir / "distribution_main" / "distribution_geometry.npz", "distribution_geometry.npz")
    dom_csv = require_path(trial_dir / "domain_shift_main" / "domain_shift_rollout.csv", "domain_shift_rollout.csv")
    dom_v = require_path(trial_dir / "domain_shift_main" / "rollouts_delta_0.4_vanilla_flow.npz", "delta=0.4 vanilla rollouts")
    dom_c = require_path(trial_dir / "domain_shift_main" / "rollouts_delta_0.4_curvature_flow.npz", "delta=0.4 curvature rollouts")
    env_dir = require_path(trial_dir / "eval_curvature_main", "eval_curvature_main directory")
    return {
        "nominal_vanilla_rollouts": eval_vanilla_main,
        "nominal_curvature_rollouts": eval_curvature_main,
        "domain_vanilla_rollouts": dom_v,
        "domain_curvature_rollouts": dom_c,
        "distribution_csv": dist_csv,
        "distribution_npz": dist_npz,
        "domain_shift_csv": dom_csv,
        "workspace_env_dir": env_dir,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", default="")
    parser.add_argument("--trial_dir", default="")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    if not args.trial_dir and not args.eval_dir:
        raise ValueError("Provide either --trial_dir or --eval_dir")
    if args.eval_dir and not args.trial_dir:
        raise ValueError("--eval_dir fallback is no longer supported for final paper plots; use --trial_dir")

    trial_dir = Path(args.trial_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = _resolve_trial_inputs(trial_dir)
    workspace = _extract_workspace(paths["workspace_env_dir"])

    plot_pullback_scene(
        _load_rollout_npz(paths["nominal_vanilla_rollouts"]),
        _load_rollout_npz(paths["nominal_curvature_rollouts"]),
        _load_rollout_npz(paths["domain_vanilla_rollouts"]),
        _load_rollout_npz(paths["domain_curvature_rollouts"]),
        workspace,
        outdir,
    )
    make_distribution_plot(paths["distribution_csv"], paths["distribution_npz"], outdir)
    plot_domain_shift(paths["domain_shift_csv"], outdir)
    write_latex_snippets(outdir)


if __name__ == "__main__":
    main()
