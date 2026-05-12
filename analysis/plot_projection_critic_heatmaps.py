import argparse
import csv
import sys
from pathlib import Path

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.safe_obstacle_navigation_2d import SafeObstacleNavigation2DEnv
from eval.eval_safe_obstacle_navigation import load_agent
from relax.safety.obstacle_navigation_filter import ObstacleNavConfig, is_action_feasible_np, project_action_np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--algo", default="safe_pullback_rf2")
    ap.add_argument("--method_name", required=True)
    ap.add_argument("--state", "--states", dest="states", nargs=2, type=float, action="append", required=True)
    ap.add_argument("--grid_size", type=int, default=101)
    ap.add_argument("--num_policy_samples", type=int, default=200)
    ap.add_argument("--out_dir", default="paper_outputs/figures/projection_critic_heatmaps")
    ap.add_argument("--tau_c", type=float, default=0.2)
    ap.add_argument("--plot_vp_state_heatmap", action="store_true")
    ap.add_argument("--state_grid_x", type=int, default=181)
    ap.add_argument("--state_grid_y", type=int, default=121)
    return ap.parse_args()


def percentile_clip(arr, lo=5.0, hi=95.0):
    return np.percentile(arr, lo), np.percentile(arr, hi)


def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt(np.sum(x * x)) * np.sqrt(np.sum(y * y))) + 1e-12
    return float(np.sum(x * y) / denom)


def spearman_corr(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return pearson_corr(rx, ry)


def compute_state_panels(agent, env, cfg, action_grid, pos, tau_c, num_policy_samples, seed):
    obs = env._get_obs_from_state(pos)
    n = action_grid.shape[0]
    obs_batch = np.repeat(obs[None, :], n, axis=0).astype(np.float32)

    exec_actions = np.zeros_like(action_grid)
    d_s = np.zeros((n,), dtype=np.float32)
    feasible = np.zeros((n,), dtype=bool)
    residual = np.zeros((n,), dtype=np.float32)

    for i, raw in enumerate(action_grid):
        ea, _, gap, _, _ = project_action_np(pos, raw, action_grid, cfg)
        exec_actions[i] = ea
        residual[i] = float(gap)
        d_s[i] = float(np.sum((ea - raw) ** 2))
        feasible[i] = is_action_feasible_np(pos, raw, cfg)

    q_s = np.asarray(agent.agent.get_qp(agent.state.params.qp, obs_batch, action_grid), dtype=np.float32).reshape(-1)
    v_s = float(np.asarray(agent.agent.get_vp(agent.state.params.vp, obs[None, :])).reshape(-1)[0])
    q1 = np.asarray(agent.agent.q(agent.state.params.q1, obs_batch, exec_actions), dtype=np.float32).reshape(-1)
    q2 = np.asarray(agent.agent.q(agent.state.params.q2, obs_batch, exec_actions), dtype=np.float32).reshape(-1)
    q_r = np.minimum(q1, q2)

    f_s = np.exp(-np.clip(q_s, 0.0, np.inf) / max(tau_c, 1e-6))

    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, num_policy_samples)
    pol_raw = np.asarray([np.asarray(agent.get_action(keys[i], obs[None, :])[0], dtype=np.float32) for i in range(num_policy_samples)])
    pol_raw = np.clip(pol_raw, -1.0, 1.0)
    pol_exec = np.zeros_like(pol_raw)
    pol_ds = np.zeros((num_policy_samples,), dtype=np.float32)
    pol_qs = np.zeros((num_policy_samples,), dtype=np.float32)
    pol_res = np.zeros((num_policy_samples,), dtype=np.float32)
    pol_feas = np.zeros((num_policy_samples,), dtype=bool)
    for i in range(num_policy_samples):
        ea, _, gap, _, _ = project_action_np(pos, pol_raw[i], action_grid, cfg)
        pol_exec[i] = ea
        pol_ds[i] = np.sum((ea - pol_raw[i]) ** 2)
        pol_res[i] = float(gap)
        pol_feas[i] = is_action_feasible_np(pos, pol_raw[i], cfg)
    pol_qs = np.asarray(agent.agent.get_qp(agent.state.params.qp, np.repeat(obs[None, :], num_policy_samples, axis=0), pol_raw), dtype=np.float32).reshape(-1)

    return dict(obs=obs, d_s=d_s, q_s=q_s, f_s=f_s, q_r=q_r, feasible=feasible, exec_actions=exec_actions,
                residual=residual, pol_raw=pol_raw, pol_exec=pol_exec, pol_ds=pol_ds, pol_qs=pol_qs,
                pol_res=pol_res, pol_feas=pol_feas, v_s=v_s)


def plot_state_figure(out_path_base, method_name, pos, grid_x, grid_y, data):
    shape = grid_x.shape
    d_s = data["d_s"].reshape(shape)
    q_s = data["q_s"].reshape(shape)
    f_s = data["f_s"].reshape(shape)
    feas = data["feasible"].reshape(shape).astype(float)

    qvmin, qvmax = percentile_clip(data["q_s"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    extent = [-1, 1, -1, 1]

    im0 = axes[0, 0].imshow(d_s, extent=extent, origin="lower", cmap="magma", aspect="equal")
    axes[0, 0].contour(grid_x, grid_y, feas, levels=[0.5], colors="w", linewidths=1.1)
    axes[0, 0].set_title("True projection cost $d_S$")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(q_s, extent=extent, origin="lower", cmap="viridis", aspect="equal", vmin=qvmin, vmax=qvmax)
    axes[0, 1].contour(grid_x, grid_y, feas, levels=[0.5], colors="w", linewidths=1.1)
    axes[0, 1].set_title("Learned projection critic $Q_S$")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[1, 0].imshow(f_s, extent=extent, origin="lower", cmap="viridis", aspect="equal")
    axes[1, 0].contour(grid_x, grid_y, feas, levels=[0.5], colors="w", linewidths=1.1)
    axes[1, 0].set_title("Compatibility $F_S = e^{-Q_S^+/\\tau_c}$")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    im3 = axes[1, 1].imshow(q_s, extent=extent, origin="lower", cmap="viridis", aspect="equal", vmin=qvmin, vmax=qvmax)
    axes[1, 1].contour(grid_x, grid_y, feas, levels=[0.5], colors="w", linewidths=1.1)
    raw, exe = data["pol_raw"], data["pol_exec"]
    axes[1, 1].scatter(raw[:, 0], raw[:, 1], s=12, c="tab:orange", alpha=0.55, label="policy raw")
    axes[1, 1].scatter(exe[:, 0], exe[:, 1], s=12, c="tab:cyan", alpha=0.55, label="executed")
    for i in np.linspace(0, len(raw) - 1, min(80, len(raw)), dtype=int):
        axes[1, 1].plot([raw[i, 0], exe[i, 0]], [raw[i, 1], exe[i, 1]], color="k", alpha=0.2, lw=0.6)
    axes[1, 1].set_title("$Q_S$ with policy samples and projection arrows")
    axes[1, 1].legend(loc="upper left", fontsize=8)
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    for ax in axes.ravel():
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("raw action dim-1")
        ax.set_ylabel("raw action dim-2")

    fig.suptitle(f"{method_name} | state=({pos[0]:.2f}, {pos[1]:.2f})", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(str(out_path_base) + ".png", dpi=240)
    fig.savefig(str(out_path_base) + ".pdf")
    plt.close(fig)


def plot_vp_heatmap(agent, env, cfg, out_path, method_name, gx, gy):
    xs = np.linspace(-3.5, 3.5, gx)
    ys = np.linspace(-2.0, 2.0, gy)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1).astype(np.float32)
    obs = np.asarray([env._get_obs_from_state(p) for p in pts], dtype=np.float32)
    vp = np.asarray(agent.agent.get_vp(agent.state.params.vp, obs), dtype=np.float32).reshape(-1)

    inside_workspace = (pts[:, 0] >= cfg.x_min) & (pts[:, 0] <= cfg.x_max) & (pts[:, 1] >= cfg.y_min) & (pts[:, 1] <= cfg.y_max)
    obs_d = np.linalg.norm(pts - cfg.obstacle_center[None, :], axis=1)
    outside_obstacle = obs_d >= cfg.obstacle_radius
    mask_safe = inside_workspace & outside_obstacle
    vp_masked = np.full_like(vp, np.nan)
    vp_masked[mask_safe] = vp[mask_safe]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(vp_masked.reshape(yy.shape), origin="lower", extent=[-3.5, 3.5, -2.0, 2.0], cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, label="$V_S(s)$")

    ax.add_patch(Circle(tuple(cfg.obstacle_center), cfg.obstacle_radius, fill=False, ec="red", lw=2.0, label="obstacle"))
    ax.add_patch(Circle(tuple(cfg.obstacle_center), cfg.obstacle_radius_tight, fill=False, ec="orange", lw=1.8, ls="--", label="tightened obstacle"))
    goal = env.goal
    ax.add_patch(Circle((float(goal[0]), float(goal[1])), env.goal_radius, fill=False, ec="lime", lw=2.0, label="goal"))
    ax.add_patch(Rectangle((cfg.x_min, cfg.y_min), cfg.x_max - cfg.x_min, cfg.y_max - cfg.y_min, fill=False, ec="white", lw=1.6, ls=":"))
    ax.set_title(f"{method_name} | $V_S$ state-space heatmap")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out_path) + ".png", dpi=240)
    fig.savefig(str(out_path) + ".pdf")
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = Path("paper_outputs/tables/projection_critic_heatmap_stats.csv")
    table_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = ObstacleNavConfig()
    env = SafeObstacleNavigation2DEnv()
    agent = load_agent(args.checkpoint, args.algo)

    lin = np.linspace(-1.0, 1.0, args.grid_size)
    grid_x, grid_y = np.meshgrid(lin, lin)
    action_grid = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1).astype(np.float32)

    rows = []
    for i, st in enumerate(args.states):
        pos = np.asarray(st, dtype=np.float32)
        data = compute_state_panels(agent, env, cfg, action_grid, pos, args.tau_c, args.num_policy_samples, seed=1000 + i)
        plot_state_figure(out_dir / f"projection_critic_heatmap_state{i}", args.method_name, pos, grid_x, grid_y, data)

        feas = data["feasible"]
        infeas = ~feas
        row = {
            "method_name": args.method_name,
            "checkpoint": args.checkpoint,
            "state_x": float(pos[0]),
            "state_y": float(pos[1]),
            "corr_QS_dS": pearson_corr(data["q_s"], data["d_s"]),
            "spearman_QS_dS": spearman_corr(data["q_s"], data["d_s"]),
            "mean_QS_feasible": float(np.mean(data["q_s"][feas])) if np.any(feas) else np.nan,
            "mean_QS_infeasible": float(np.mean(data["q_s"][infeas])) if np.any(infeas) else np.nan,
            "mean_dS_feasible": float(np.mean(data["d_s"][feas])) if np.any(feas) else np.nan,
            "mean_dS_infeasible": float(np.mean(data["d_s"][infeas])) if np.any(infeas) else np.nan,
            "policy_QS_mean": float(np.mean(data["pol_qs"])),
            "policy_dS_mean": float(np.mean(data["pol_ds"])),
            "policy_projection_residual_mean": float(np.mean(data["pol_res"])),
            "policy_feasible_ratio": float(np.mean(data["pol_feas"].astype(np.float32))),
        }
        rows.append(row)

    if args.plot_vp_state_heatmap:
        plot_vp_heatmap(agent, env, cfg, out_dir / "vp_state_space_heatmap", args.method_name, args.state_grid_x, args.state_grid_y)

    fieldnames = [
        "method_name", "checkpoint", "state_x", "state_y", "corr_QS_dS", "spearman_QS_dS",
        "mean_QS_feasible", "mean_QS_infeasible", "mean_dS_feasible", "mean_dS_infeasible",
        "policy_QS_mean", "policy_dS_mean", "policy_projection_residual_mean", "policy_feasible_ratio",
    ]
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
