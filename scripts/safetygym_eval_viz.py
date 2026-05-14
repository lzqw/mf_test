import csv
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


def _safe_float(v, default=np.nan):
    try:
        if isinstance(v, (str, bytes)):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def extract_goal_from_env(env):
    u = getattr(env, "unwrapped", env)
    task = getattr(u, "task", None)
    if task is None:
        return None
    goal_obj = getattr(task, "goal", None)
    candidates = []
    if goal_obj is not None:
        candidates.extend([getattr(goal_obj, "pos", None), getattr(goal_obj, "position", None)])
        fn = getattr(goal_obj, "get_position", None)
        if callable(fn):
            try:
                candidates.append(fn())
            except Exception:
                pass
    candidates.extend([getattr(task, "goal_xy", None), getattr(task, "goal_pos", None), getattr(task, "goal_position", None)])
    for c in candidates:
        if c is None:
            continue
        try:
            arr = np.asarray(c, dtype=np.float32).reshape(-1)
            if arr.size >= 2:
                return arr[:2]
        except Exception:
            continue
    return None


def collect_scene(env, safe_filter):
    return {
        "hazards": safe_filter._extract_hazards_from_env(env),
        "objects": safe_filter._extract_objects_from_env(env),
        "goal": extract_goal_from_env(env),
    }


def save_records(records, save_prefix: Path):
    with open(str(save_prefix) + "_records.pkl", "wb") as f:
        pickle.dump(records, f)
    with open(str(save_prefix) + "_records.csv", "w", newline="") as f:
        if not records:
            return
        fieldnames = sorted(records[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            out = dict(row)
            for k in ["raw_action", "exec_action"]:
                if k in out:
                    out[k] = np.asarray(out[k]).tolist()
            writer.writerow(out)


def plot_safetygym_eval_trajectory(records, scene, save_path, title="", arrow_stride=25, arrow_scale=0.25):
    if not records:
        return
    xs = np.asarray([r["ego_x"] for r in records], dtype=np.float32)
    ys = np.asarray([r["ego_y"] for r in records], dtype=np.float32)
    costs = np.asarray([_safe_float(r.get("cost", 0.0), 0.0) for r in records], dtype=np.float32)
    apr = np.asarray([_safe_float(r.get("projection_residual", 0.0), 0.0) for r in records], dtype=np.float32)
    emergency = np.asarray([_safe_float(r.get("emergency_active", 0.0), 0.0) for r in records], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(xs, ys, label="trajectory")
    ax.scatter(xs[0], ys[0], marker="o", s=60, label="start")
    ax.scatter(xs[-1], ys[-1], marker="x", s=60, label="end")

    for i, h in enumerate(scene.get("hazards", [])):
        pos = np.asarray(h.get("pos", [np.nan, np.nan]), dtype=np.float32).reshape(-1)
        if pos.size < 2:
            continue
        r = _safe_float(h.get("radius", 0.2), 0.2)
        ax.add_patch(Circle((pos[0], pos[1]), r, fill=False, alpha=0.5, label="hazard" if i == 0 else None))

    goal = scene.get("goal", None)
    if goal is not None:
        g = np.asarray(goal, dtype=np.float32).reshape(-1)
        if g.size >= 2:
            ax.scatter(g[0], g[1], marker="*", s=120, label="goal")
    for i, o in enumerate(scene.get("objects", [])):
        pos = np.asarray(o.get("pos", [np.nan, np.nan]), dtype=np.float32).reshape(-1)
        if pos.size >= 2:
            ax.scatter(pos[0], pos[1], marker="s", s=40, label="object" if i == 0 else None)

    idx_cost = np.where(costs > 0)[0]
    if idx_cost.size:
        ax.scatter(xs[idx_cost], ys[idx_cost], s=18, label="cost>0")
    idx_apr = np.where(apr > 0.1)[0]
    if idx_apr.size:
        ax.scatter(xs[idx_apr], ys[idx_apr], s=18, label="high_residual")
    idx_em = np.where(emergency > 0)[0]
    if idx_em.size:
        ax.scatter(xs[idx_em], ys[idx_em], s=20, label="emergency")

    raw_legend = exec_legend = False
    for i in range(0, len(records), max(int(arrow_stride), 1)):
        x, y = xs[i], ys[i]
        raw = np.asarray(records[i].get("raw_action", [0, 0]), dtype=np.float32).reshape(-1)
        exe = np.asarray(records[i].get("exec_action", [0, 0]), dtype=np.float32).reshape(-1)
        if raw.size >= 2:
            ax.arrow(x, y, arrow_scale * raw[0], arrow_scale * raw[1], alpha=0.4, linestyle="--", head_width=0.03, length_includes_head=True, label="raw_action" if not raw_legend else None)
            raw_legend = True
        if exe.size >= 2:
            ax.arrow(x, y, arrow_scale * exe[0], arrow_scale * exe[1], alpha=0.7, head_width=0.03, length_includes_head=True, label="exec_action" if not exec_legend else None)
            exec_legend = True

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_safetygym_eval_diagnostics(records, save_path, title=""):
    if not records:
        return
    t = np.arange(len(records))
    pr = np.asarray([_safe_float(r.get("projection_residual", np.nan)) for r in records], dtype=np.float32)
    rn = np.asarray([_safe_float(r.get("raw_action_norm", np.nan)) for r in records], dtype=np.float32)
    en = np.asarray([_safe_float(r.get("exec_action_norm", np.nan)) for r in records], dtype=np.float32)
    cmh = np.asarray([_safe_float(r.get("current_min_h", np.nan)) for r in records], dtype=np.float32)
    pmh = np.asarray([_safe_float(r.get("predicted_min_h", np.nan)) for r in records], dtype=np.float32)
    c = np.asarray([_safe_float(r.get("cost", 0.0), 0.0) for r in records], dtype=np.float32)
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axs[0].plot(t, pr); axs[0].set_ylabel("proj_res")
    axs[1].plot(t, rn, label="raw_norm"); axs[1].plot(t, en, label="exec_norm"); axs[1].legend(fontsize=8)
    axs[2].plot(t, cmh, label="current_min_h"); axs[2].plot(t, pmh, label="pred_min_h"); axs[2].legend(fontsize=8)
    axs[3].plot(t, c); axs[3].set_ylabel("cost"); axs[3].set_xlabel("step")
    for ax in axs:
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
