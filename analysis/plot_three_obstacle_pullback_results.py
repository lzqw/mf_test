import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Ellipse
import numpy as np
import pandas as pd

from envs.safe_obstacle_double_integrator_2d import SafeObstacleDoubleIntegrator2DEnv


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


def load_npz(path: Path):
    return np.load(path, allow_pickle=True)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_env_from_eval_config(eval_dir: Path):
    cfg = load_json(require_path(eval_dir / "env_config.json", "env_config.json"))
    return SafeObstacleDoubleIntegrator2DEnv(
        noise_sigma=(0.0, 0.0),
        use_filter=False,
        map_id=cfg.get("map_id", "three_circles"),
        obs_mode=cfg.get("obs_mode", "all_obstacles"),
        start_y_range=cfg.get("start_y_range", 0.9),
        dt=cfg.get("dt", 0.08),
        a_max=cfg.get("a_max", 3.5),
        v_max=cfg.get("v_max", 2.5),
        damping=cfg.get("damping", 0.98),
        episode_len=cfg.get("episode_len", 250),
        eps_obs=cfg.get("eps_obs", 0.06),
        reward_mode=cfg.get("reward_mode", "multi_route_progress"),
        reward_cfg=cfg.get("reward_cfg", None),
    )


def _valid_traj(npz_data, idx):
    length = int(np.asarray(npz_data["valid_lengths"], dtype=np.int32)[idx])
    return np.asarray(npz_data["positions"][idx, :length], dtype=np.float32)


def _start_group(y):
    if y > 0.25:
        return "upper"
    if y < -0.25:
        return "lower"
    return "middle"


def _representative_idx(npz_data, route=None):
    route_tags = np.asarray(npz_data["route_tags"])
    success = np.asarray(npz_data["is_success"], dtype=bool)
    valid_lengths = np.asarray(npz_data["valid_lengths"], dtype=np.int32)
    indices = np.where(success)[0]
    if route is not None:
        indices = np.array([i for i in indices if route_tags[i] == route], dtype=np.int32)
    if indices.size == 0:
        indices = np.arange(valid_lengths.shape[0])
    subset = valid_lengths[indices]
    return int(indices[np.argsort(subset)[len(indices) // 2]])


def _draw_workspace(ax, env):
    for center, radius in zip(env.obstacle_centers, env.obstacle_radii):
        ax.add_patch(Circle(tuple(center), float(radius), fill=False, lw=1.2, ec="0.2"))
        ax.add_patch(Circle(tuple(center), float(radius + env.eps_obs), fill=False, lw=1.0, ls="--", ec="0.55"))
    ax.scatter([env.start_center[0]], [0.0], c="k", marker="*", s=55, label="Start region")
    ax.scatter([env.goal[0]], [env.goal[1]], c="tab:green", marker="o", s=35, label="Goal")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.16)


def plot_routes_figure(trial_dir: Path, outdir: Path):
    cur_npz = load_npz(require_path(trial_dir / "eval_curvature_main" / "rollouts.npz", "curvature main rollouts"))
    van_npz = load_npz(require_path(trial_dir / "eval_vanilla_main" / "rollouts.npz", "vanilla main rollouts"))
    dom_cur_npz = load_npz(require_path(trial_dir / "domain_shift_main" / "rollouts_delta_0.4_curvature_flow.npz", "delta=0.4 curvature rollouts"))
    dom_van_npz = load_npz(require_path(trial_dir / "domain_shift_main" / "rollouts_delta_0.4_vanilla_flow.npz", "delta=0.4 vanilla rollouts"))
    env = build_env_from_eval_config(trial_dir / "eval_curvature_main")

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)

    ax = axs[0]
    _draw_workspace(ax, env)
    ax.set_title(r"(a) Curvature-Shaped Flow, $\delta = 0$")
    start_y = np.asarray(cur_npz["start_y"], dtype=np.float32)
    colors = {"upper": "tab:orange", "middle": "0.45", "lower": "tab:blue"}
    labels_done = set()
    for i in range(min(cur_npz["positions"].shape[0], 120)):
        traj = _valid_traj(cur_npz, i)
        group = _start_group(float(start_y[i]))
        label = f"{group} start" if group not in labels_done else None
        labels_done.add(group)
        ax.plot(traj[:, 0], traj[:, 1], color=colors[group], alpha=0.16, lw=1.0, label=label)

    for route, color in [("upper", "tab:red"), ("lower", "tab:blue")]:
        ridx = _representative_idx(cur_npz, route=route)
        rep = _valid_traj(cur_npz, ridx)
        ax.plot(rep[:, 0], rep[:, 1], color=color, lw=2.0, alpha=1.0)

    ax = axs[1]
    _draw_workspace(ax, env)
    ax.set_title(r"(b) Shifted rollouts, $\delta = 0.4$")
    for npz_data, color, label in [(dom_van_npz, "tab:blue", "Vanilla Flow"), (dom_cur_npz, "tab:red", "Curvature-Shaped Flow")]:
        for i in range(min(npz_data["positions"].shape[0], 80)):
            traj = _valid_traj(npz_data, i)
            ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.12, lw=1.0)
        ridx = _representative_idx(npz_data)
        rep = _valid_traj(npz_data, ridx)
        ax.plot(rep[:, 0], rep[:, 1], color=color, lw=2.0, alpha=1.0, label=label)

    handles, labels = axs[0].get_legend_handles_labels()
    h2, l2 = axs[1].get_legend_handles_labels()
    fig.legend(handles + h2, labels + l2, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.savefig(outdir / "fig_three_obstacle_routes.png", dpi=240, bbox_inches="tight")
    fig.savefig(outdir / "fig_three_obstacle_routes.pdf", bbox_inches="tight")
    plt.close(fig)


def _ellipse(ax, cov, color, ls="-", lw=1.5):
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-10, None)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width = 2.0 * np.sqrt(eigvals[0])
    height = 2.0 * np.sqrt(eigvals[1])
    ax.add_patch(Ellipse((0.0, 0.0), width, height, angle=angle, fill=False, ec=color, lw=lw, ls=ls))


def plot_distribution_figure(trial_dir: Path, outdir: Path):
    dist_csv = pd.read_csv(require_path(trial_dir / "distribution_main" / "distribution_geometry.csv", "distribution_geometry.csv"))
    dist_csv.to_csv(outdir / "distribution_geometry.csv", index=False)
    dist_npz = load_npz(require_path(trial_dir / "distribution_main" / "distribution_geometry.npz", "distribution_geometry.npz"))
    samples = [item.item() if hasattr(item, "item") else item for item in np.asarray(dist_npz["samples"], dtype=object)]
    env = build_env_from_eval_config(trial_dir / "eval_curvature_main")

    nominal_cov = np.mean(np.stack([s["Sigma_nominal_local"] for s in samples], axis=0), axis=0)
    safety_cov = np.mean(np.stack([s["Sigma_safe_local"] for s in samples], axis=0), axis=0)
    robust_cov = np.mean(np.stack([s["Sigma_robust_local"] for s in samples], axis=0), axis=0)
    vanilla_local = np.concatenate([s["actions_vanilla_local"] for s in samples], axis=0)
    curvature_local = np.concatenate([s["actions_curvature_local"] for s in samples], axis=0)
    vanilla_cov = np.cov(vanilla_local.T, bias=False)
    curvature_cov = np.cov(curvature_local.T, bias=False)

    rng = np.random.default_rng(0)
    synthetic_nom = rng.multivariate_normal(np.zeros(2), nominal_cov, size=512)
    synthetic_safe = rng.multivariate_normal(np.zeros(2), safety_cov, size=512)
    synthetic_rob = rng.multivariate_normal(np.zeros(2), robust_cov, size=512)
    extent = np.concatenate([synthetic_nom, synthetic_safe, synthetic_rob, vanilla_local, curvature_local], axis=0)
    lim = max(0.3, float(np.quantile(np.abs(extent), 0.995)))

    fig = plt.figure(figsize=(7.2, 4.2), constrained_layout=True)
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 2.2])
    ax_map = fig.add_subplot(gs[0, :])
    _draw_workspace(ax_map, env)
    ax_map.set_title("Selected safety-critical states")
    for sample in samples:
        ax_map.scatter([sample["px"]], [sample["py"]], c="k", s=12)

    panels = [
        ("(a) Nominal", synthetic_nom, nominal_cov, "tab:blue"),
        ("(b) Safety-shaped", synthetic_safe, safety_cov, "tab:green"),
        ("(c) Robust-shaped", synthetic_rob, robust_cov, "tab:purple"),
        ("(d) Sampled Flow", curvature_local, curvature_cov, "tab:red"),
    ]
    for j, (title, points, cov, color) in enumerate(panels):
        ax = fig.add_subplot(gs[1, j])
        ax.axhline(0.0, color="0.75", lw=0.8)
        ax.axvline(0.0, color="0.75", lw=0.8)
        if j < 3:
            ax.scatter(points[:, 0], points[:, 1], s=6, alpha=0.12, c=color)
            _ellipse(ax, cov, color)
        else:
            ax.scatter(vanilla_local[:, 0], vanilla_local[:, 1], s=5, alpha=0.06, c="tab:blue", label="Vanilla raw")
            ax.scatter(points[:, 0], points[:, 1], s=5, alpha=0.12, c="tab:red", label="Curvature raw")
            _ellipse(ax, vanilla_cov, "tab:blue", ls="--", lw=1.2)
            _ellipse(ax, curvature_cov, "tab:red", lw=1.5)
        ax.set_title(title)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.16)
        ax.set_xlabel("Normal action")
        if j == 0:
            ax.set_ylabel("Tangent action")
        if j == 3:
            ax.legend(loc="upper left", frameon=False)

    fig.savefig(outdir / "fig_three_obstacle_distribution_geometry.png", dpi=240, bbox_inches="tight")
    fig.savefig(outdir / "fig_three_obstacle_distribution_geometry.pdf", bbox_inches="tight")
    plt.close(fig)

    order = ["Nominal", "Safety-shaped", "Robust-shaped", "Vanilla Flow", "Curvature-Shaped Flow"]
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Normal Var.$\downarrow$ & Tangent Var. & NLR$\downarrow$ & TCR$\uparrow$ \\",
        r"\midrule",
    ]
    for method in order:
        sub = dist_csv[dist_csv["method"] == method]
        lines.append(
            f"{method} & "
            f"{sub['normal_var'].mean():.4f} $\\pm$ {sub['normal_var'].std(ddof=0):.4f} & "
            f"{sub['tangent_var'].mean():.4f} $\\pm$ {sub['tangent_var'].std(ddof=0):.4f} & "
            f"{sub['nlr'].mean():.4f} $\\pm$ {sub['nlr'].std(ddof=0):.4f} & "
            f"{sub['tcr'].mean():.4f} $\\pm$ {sub['tcr'].std(ddof=0):.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (outdir / "table_distribution_geometry.tex").write_text("\n".join(lines), encoding="utf-8")


def plot_domain_shift_figure(trial_dir: Path, outdir: Path):
    df = pd.read_csv(require_path(trial_dir / "domain_shift_main" / "domain_shift_rollout.csv", "domain_shift_rollout.csv"))
    df.to_csv(outdir / "domain_shift_rollout.csv", index=False)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), constrained_layout=True)
    methods = [("Vanilla Flow", "tab:blue"), ("Curvature-Shaped Flow", "tab:red")]
    specs = [
        ("J_eval_mean", "J_eval_std", r"(a) $J_{\rm eval}$ vs $\delta$", r"$J_{\rm eval}$"),
        ("violation_rate_mean", "violation_rate_std", r"(b) Violation rate vs $\delta$", "Violation rate"),
        ("h_min_mean", "h_min_std", r"(c) $h_{\min}$ vs $\delta$", r"$h_{\min}$"),
    ]
    for ax, (mean_col, std_col, title, ylabel) in zip(axs, specs):
        for method, color in methods:
            sub = df[df["method"] == method].sort_values("delta")
            ax.errorbar(sub["delta"], sub[mean_col], yerr=sub[std_col], color=color, marker="o", lw=1.4, ms=3.5, capsize=2, label=method)
        if mean_col == "h_min_mean":
            ax.axhline(0.0, color="0.35", lw=1.0, ls="--")
        ax.set_title(title)
        ax.set_xlabel(r"$\delta$")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.16)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.savefig(outdir / "fig_three_obstacle_domain_shift.png", dpi=240, bbox_inches="tight")
    fig.savefig(outdir / "fig_three_obstacle_domain_shift.pdf", bbox_inches="tight")
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
            row = sub[sub["method"] == method].iloc[0]
            label = "Vanilla" if method == "Vanilla Flow" else "Ours"
            lines.append(f"{delta:.1f} & {label} & {row['J_eval_mean']:.4f} & {row['violation_rate_mean']:.4f} & {row['h_min_mean']:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (outdir / "table_domain_shift_rollout.tex").write_text("\n".join(lines), encoding="utf-8")


def write_route_conditioned_table(trial_dir: Path, outdir: Path):
    summary = load_json(require_path(trial_dir / "eval_curvature_main" / "eval_summary.json", "curvature main summary"))
    rows = [
        dict(
            start_group="upper_start",
            success_rate=summary.get("upper_start_success_rate", 0.0),
            collision_rate=summary.get("upper_start_collision_rate", 0.0),
            upper_route_fraction=summary.get("upper_start_upper_route_fraction", 0.0),
            lower_route_fraction=summary.get("upper_start_lower_route_fraction", 0.0),
            h_min=summary.get("upper_start_h_min_mean", 0.0),
            J_eval=summary.get("upper_start_J_eval_mean", 0.0),
        ),
        dict(
            start_group="middle_start",
            success_rate=summary.get("middle_start_success_rate", 0.0),
            collision_rate=summary.get("middle_start_collision_rate", 0.0),
            upper_route_fraction=summary.get("middle_start_upper_route_fraction", 0.0),
            lower_route_fraction=summary.get("middle_start_lower_route_fraction", 0.0),
            h_min=summary.get("middle_start_h_min_mean", 0.0),
            J_eval=summary.get("middle_start_J_eval_mean", 0.0),
        ),
        dict(
            start_group="lower_start",
            success_rate=summary.get("lower_start_success_rate", 0.0),
            collision_rate=summary.get("lower_start_collision_rate", 0.0),
            upper_route_fraction=summary.get("lower_start_upper_route_fraction", 0.0),
            lower_route_fraction=summary.get("lower_start_lower_route_fraction", 0.0),
            h_min=summary.get("lower_start_h_min_mean", 0.0),
            J_eval=summary.get("lower_start_J_eval_mean", 0.0),
        ),
        dict(
            start_group="all",
            success_rate=summary.get("success_rate", 0.0),
            collision_rate=summary.get("collision_rate", 0.0),
            upper_route_fraction=summary.get("route_upper_fraction", 0.0),
            lower_route_fraction=summary.get("route_lower_fraction", 0.0),
            h_min=summary.get("h_min_mean", 0.0),
            J_eval=summary.get("J_eval_mean", 0.0),
        ),
    ]
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "route_conditioned_performance.csv", index=False)
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Start group & Success rate & Collision rate & Upper-route frac. & Lower-route frac. & $h_{\min}$ & $J_{\rm eval}$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['start_group']} & {row['success_rate']:.4f} & {row['collision_rate']:.4f} & "
            f"{row['upper_route_fraction']:.4f} & {row['lower_route_fraction']:.4f} & "
            f"{row['h_min']:.4f} & {row['J_eval']:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (outdir / "table_route_conditioned_performance.tex").write_text("\n".join(lines), encoding="utf-8")


def write_latex_snippets(outdir: Path):
    text = r"""\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/fig_three_obstacle_routes.pdf}
\caption{Multi-obstacle pullback environment and route-conditioned rollouts. The curvature-shaped flow policy learns multiple safe homotopy classes: upper-start trajectories preferentially select the upper route, whereas lower-start trajectories select the lower route.}
\label{fig:three_obstacle_routes}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/fig_three_obstacle_distribution_geometry.pdf}
\caption{Local policy distributions in the nearest-obstacle normal--tangent frame. Safety and robust curvature suppress boundary-normal stochasticity while preserving tangent stochasticity. The sampled curvature-shaped flow policy reproduces this anisotropic structure in the multi-obstacle scene.}
\label{fig:three_obstacle_distribution_geometry}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/fig_three_obstacle_domain_shift.pdf}
\caption{Robust rollout under structured actuator-gain domain shift. The curvature-shaped flow policy maintains lower violation rate and larger minimum clearance as the shift magnitude increases.}
\label{fig:three_obstacle_domain_shift}
\end{figure}

\input{tables/table_route_conditioned_performance}
\input{tables/table_distribution_geometry}
\input{tables/table_domain_shift_rollout}
"""
    (outdir / "latex_snippets.tex").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial_dir", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    trial_dir = Path(args.trial_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_routes_figure(trial_dir, outdir)
    plot_distribution_figure(trial_dir, outdir)
    plot_domain_shift_figure(trial_dir, outdir)
    write_route_conditioned_table(trial_dir, outdir)
    write_latex_snippets(outdir)


if __name__ == "__main__":
    main()
