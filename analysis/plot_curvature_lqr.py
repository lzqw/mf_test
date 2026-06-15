import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from relax.utils.curvature import (  # noqa: E402
    choose_tau,
    covariance_from_curvature,
    covariance_ellipse_points,
    covariance_variances,
    covariance_variances as _covariance_variances,
    default_double_integrator,
    default_lqr_config,
    full_robust_curvature,
    nlr_tcr,
    nominal_curvature,
    safety_curvature,
    solve_discounted_riccati,
    action_normal_matrix,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick_check", action="store_true")
    return ap.parse_args()


def safe_eigvals(x):
    return np.linalg.eigvalsh(0.5 * (x + x.T)).tolist()


def run_rollouts(
    A,
    B,
    C,
    b,
    Q,
    R,
    K,
    Sigma,
    Sigma_w,
    x0,
    rho,
    T,
    N_eval,
    delta,
    seed,
):
    rng = np.random.default_rng(seed)
    eval_costs = []
    violation_rates = []
    min_margins = []

    for _ in range(N_eval):
        x = np.array(x0, dtype=np.float64)
        total_cost = 0.0
        steps_violation = 0
        min_margin = np.inf

        for _t in range(T):
            base_u = -K @ x
            eta = rng.multivariate_normal(np.zeros(2), Sigma)
            u = base_u + eta
            F_t = -delta * np.eye(2)

            noise = rng.multivariate_normal(np.zeros(4), Sigma_w)
            x_next = A @ x + B @ ((np.eye(2) + F_t) @ u) + noise

            stage = float(x @ Q @ x + u @ R @ u)
            total_cost += stage

            margin = b - C @ x
            steps_violation += int(np.any(margin < 0.0))
            min_margin = float(min(min_margin, float(np.min(margin))))

            x = x_next

        eval_costs.append(total_cost)
        violation_rates.append(steps_violation / float(T))
        min_margins.append(min_margin)

    return (
        float(np.mean(eval_costs)),
        float(np.std(eval_costs)),
        float(np.mean(violation_rates)),
        float(np.mean(min_margins)),
        float(np.std(min_margins)),
    )


def save_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_ellipse(ax, Sigma, mean, label, color):
    pts = covariance_ellipse_points(mean, Sigma, nsig=2.0, num=300)
    ax.plot(pts[:, 0], pts[:, 1], color=color, label=label, linewidth=2)


def make_figure_ellipse(
    outdir,
    Sigma_iso,
    Sigma0,
    Sigma_safe,
    Sigma_robust,
    D,
    C_active,
):
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    mean = np.zeros(2, dtype=np.float64)

    _plot_ellipse(ax, Sigma_iso, mean, "Iso", "tab:blue")
    _plot_ellipse(ax, Sigma0, mean, "Nominal", "tab:orange")
    _plot_ellipse(ax, Sigma_safe, mean, "Safe", "tab:green")
    _plot_ellipse(ax, Sigma_robust, mean, "Robust", "tab:red")

    d1 = D[:, 0]
    d1 = d1 / (np.linalg.norm(d1) + 1e-12)
    e_t = np.array([-d1[1], d1[0]], dtype=np.float64)
    ax.arrow(0, 0, d1[0], d1[1], width=0.01, head_width=0.03, length_includes_head=True, color="k")
    ax.arrow(0, 0, e_t[0], e_t[1], width=0.01, head_width=0.03, length_includes_head=True, color="tab:purple")

    ax.set_xlabel("a_x")
    ax.set_ylabel("a_y")
    ax.set_title("Action-space curvature covariance")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)

    out_png = outdir / "fig_covariance_ellipses.png"
    out_pdf = outdir / "fig_covariance_ellipses.pdf"
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_png, out_pdf


def make_figure_variance_sweep(
    outdir,
    lambda_sweep,
    normal_vars_safe,
    tangent_vars_safe,
    normal_vars_rob,
    tangent_vars_rob,
    nlr_safe,
    tcr_safe,
    nlr_rob,
    tcr_rob,
):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    xs = np.asarray(lambda_sweep, dtype=np.float64)

    axes[0].plot(xs, normal_vars_safe, marker="o", label="Safe")
    axes[0].plot(xs, normal_vars_rob, marker="o", label="Robust")
    axes[0].set_xscale("log")
    axes[0].set_title("Normal variance")
    axes[0].set_xlabel("lambda_b")
    axes[0].set_ylabel("Var")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xs, tangent_vars_safe, marker="o", label="Safe")
    axes[1].plot(xs, tangent_vars_rob, marker="o", label="Robust")
    axes[1].set_xscale("log")
    axes[1].set_title("Tangent variance")
    axes[1].set_xlabel("lambda_b")
    axes[1].set_ylabel("Var")
    axes[1].grid(True, alpha=0.3)

    for ax in axes:
        ax.legend(loc="best")

    out_png = outdir / "fig_normal_tangent_sweep.png"
    out_pdf = outdir / "fig_normal_tangent_sweep.pdf"
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_png, out_pdf


def make_figure_domain_shift(outdir, domain_rows, delta_sweep):
    methods = sorted(set(r["method"] for r in domain_rows))
    marker_cycle = ["o", "s", "^", "d"]

    fig, ax = plt.subplots(2, 2, figsize=(11.0, 8.0))
    for mi, m in enumerate(methods):
        rows = [r for r in domain_rows if r["method"] == m]
        rows.sort(key=lambda x: x["delta"])
        xs = np.array([r["delta"] for r in rows], dtype=np.float64)
        y_cost = np.array([r["eval_cost_mean"] for r in rows], dtype=np.float64)
        y_viol = np.array([r["violation_rate_mean"] for r in rows], dtype=np.float64)
        y_margin = np.array([r["min_margin_mean"] for r in rows], dtype=np.float64)

        mk = marker_cycle[mi % len(marker_cycle)]
        ax[0, 0].plot(xs, y_cost, marker=mk, label=m)
        ax[0, 1].plot(xs, y_viol, marker=mk, label=m)
        ax[1, 0].plot(xs, y_margin, marker=mk, label=m)

    for j, (row_ax, title, ylbl) in enumerate(
        [
            (ax[0, 0], "Eval cost", "cost"),
            (ax[0, 1], "Violation rate", "rate"),
            (ax[1, 0], "Min margin", "margin"),
        ]
    ):
        row_ax.set_title(title)
        row_ax.set_xlabel("delta")
        row_ax.set_ylabel(ylbl)
        row_ax.set_xlim([min(delta_sweep), max(delta_sweep)])
        row_ax.grid(True, alpha=0.3)
        row_ax.legend(loc="best", fontsize=8)

    robust_row = [r for r in domain_rows if r["method"] == "Robust"]
    robust_slack = np.array([r["robust_slack"] for r in robust_row], dtype=np.float64)
    ax[1, 1].plot([r["delta"] for r in robust_row], robust_slack, marker="o", color="tab:purple", label="Robust")
    ax[1, 1].set_title("Robust slack")
    ax[1, 1].set_xlabel("delta")
    ax[1, 1].set_ylabel("lambda min(D_tau)")
    ax[1, 1].set_xlim([min(delta_sweep), max(delta_sweep)])
    ax[1, 1].grid(True, alpha=0.3)
    ax[1, 1].legend(loc="best", fontsize=8)

    fig.tight_layout()
    out_png = outdir / "fig_domain_shift_sweep.png"
    out_pdf = outdir / "fig_domain_shift_sweep.pdf"
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_png, out_pdf


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = default_lqr_config()
    if args.quick_check:
        cfg["lambda_sweep"] = [0, 1, 100, 1000]
        cfg["delta_sweep"] = [0, 0.2, 0.4]

    A, B = default_double_integrator(dt=cfg["dt"])
    Q = cfg["Q"]
    R = cfg["R"]
    rho = cfg["rho"]
    alpha = cfg["alpha"]
    eps = cfg["eps"]
    C = cfg["C"]
    b = cfg["b"]
    Sigma_w = cfg["Sigma_w"]

    P, K = solve_discounted_riccati(A, B, Q, R, rho=rho)
    M0 = nominal_curvature(P, B, R, rho)
    Sigma0 = covariance_from_curvature(M0, alpha=alpha)
    Sigma_iso = np.trace(Sigma0) / 2.0 * np.eye(2)

    boundary_state = np.array([1.49, 0.0, 0.0, 0.0], dtype=np.float64)
    C_active = C[[0], :]
    D = action_normal_matrix(B, C_active)

    lambda_b = 100
    Lambda_b = np.array([[lambda_b]], dtype=np.float64)
    M_safe = M0 + safety_curvature(B, C_active, Lambda_b)
    Sigma_safe = covariance_from_curvature(M_safe, alpha=alpha)

    E = B
    Phi_u = np.eye(2)
    W_P = E.T @ P @ E
    tau = choose_tau(P, E, rho)
    M_robust, Pbar, D_tau = full_robust_curvature(
        P,
        B,
        R,
        C_active,
        Lambda_b * np.eye(1),
        rho,
        E,
        Phi_u,
        W_P,
        tau,
        eps=eps,
    )
    Sigma_robust = covariance_from_curvature(M_robust, alpha=alpha)

    out1 = make_figure_ellipse(outdir, Sigma_iso, Sigma0, Sigma_safe, Sigma_robust, D, C_active)

    lambda_sweep = cfg["lambda_sweep"]

    normal_vars_safe = []
    tangent_vars_safe = []
    normal_vars_rob = []
    tangent_vars_rob = []
    nlr_safe = []
    tcr_safe = []
    nlr_rob = []
    tcr_rob = []
    rows = []

    for lam in lambda_sweep:
        Lambda_b = np.array([[float(lam)]], dtype=np.float64)
        M_safe = M0 + safety_curvature(B, C_active, Lambda_b)
        Sigma_safe = covariance_from_curvature(M_safe, alpha=alpha)
        n_safe, t_safe = _covariance_variances(Sigma_safe, D)
        nlr_s, tcr_s = nlr_tcr(Sigma_safe, D)

        M_rob, Pbar, Dtau = full_robust_curvature(
            P,
            B,
            R,
            C_active,
            Lambda_b,
            rho,
            E,
            Phi_u,
            W_P,
            tau,
            eps=eps,
        )
        Sigma_rob = covariance_from_curvature(M_rob, alpha=alpha)
        n_rob, t_rob = _covariance_variances(Sigma_rob, D)
        nlr_r, tcr_r = nlr_tcr(Sigma_rob, D)

        normal_vars_safe.append(n_safe)
        tangent_vars_safe.append(t_safe)
        normal_vars_rob.append(n_rob)
        tangent_vars_rob.append(t_rob)
        nlr_safe.append(nlr_s)
        tcr_safe.append(tcr_s)
        nlr_rob.append(nlr_r)
        tcr_rob.append(tcr_r)

        rows.append(
            {
                "lambda": float(lam),
                "method": "Safe",
                "normal_var": float(n_safe),
                "tangent_var": float(t_safe),
                "nlr": float(nlr_s),
                "tcr": float(tcr_s),
                "trace_sigma": float(np.trace(Sigma_safe)),
                "cond_M": float(np.linalg.cond(M_safe)),
            }
        )
        rows.append(
            {
                "lambda": float(lam),
                "method": "Robust",
                "normal_var": float(n_rob),
                "tangent_var": float(t_rob),
                "nlr": float(nlr_r),
                "tcr": float(tcr_r),
                "trace_sigma": float(np.trace(Sigma_rob)),
                "cond_M": float(np.linalg.cond(M_rob)),
            }
        )

    out2 = make_figure_variance_sweep(
        outdir,
        lambda_sweep,
        normal_vars_safe,
        tangent_vars_safe,
        normal_vars_rob,
        tangent_vars_rob,
        nlr_safe,
        tcr_safe,
        nlr_rob,
        tcr_rob,
    )

    normal_tangent_csv = outdir / "normal_tangent_sweep.csv"
    save_csv(
        normal_tangent_csv,
        rows,
        [
            "lambda",
            "method",
            "normal_var",
            "tangent_var",
            "nlr",
            "tcr",
            "trace_sigma",
            "cond_M",
        ],
    )

    if args.quick_check:
        domain_T = 40
        domain_N_eval = 20
    else:
        domain_T = 100
        domain_N_eval = 100

    C_active = C[[0], :]
    methods = [
        ("Iso", Sigma_iso),
        ("Nominal", Sigma0),
        ("Safe", rows[0]["method"] and Sigma_safe if False else None),
    ]
    # ensure a representative safe covariance (largest safety penalty)
    safe_ref = covariance_from_curvature(M0 + safety_curvature(B, C_active, np.array([[lambda_sweep[-1]]], dtype=np.float64)), alpha=alpha)
    robust_ref = Sigma_robust
    methods = [
        ("Iso", Sigma_iso),
        ("Nominal", Sigma0),
        ("Safe", safe_ref),
        ("Robust", robust_ref),
    ]

    x0 = np.array([1.35, 0.0, 1.35, 0.0], dtype=np.float64)
    delta_sweep = cfg["delta_sweep"]
    # robustness slack from robust curvature (method-independent for D_tau)
    robust_slack = float(np.min(np.linalg.eigvalsh(0.5 * (D_tau + D_tau.T))))
    domain_rows = []
    base_seed = int(args.seed) + 100

    for delta in delta_sweep:
        for i, (name, Sigma) in enumerate(methods):
            eval_cost_mean, eval_cost_std, violation_rate_mean, min_margin_mean, min_margin_std = run_rollouts(
                A,
                B,
                C,
                b,
                Q,
                R,
                K,
                Sigma,
                Sigma_w,
                x0,
                rho,
                domain_T,
                domain_N_eval,
                delta,
                base_seed + i * 97 + int(delta * 100),
            )

            domain_rows.append(
                {
                    "delta": float(delta),
                    "method": name,
                    "eval_cost_mean": eval_cost_mean,
                    "eval_cost_std": eval_cost_std,
                    "violation_rate_mean": violation_rate_mean,
                    "min_margin_mean": min_margin_mean,
                    "min_margin_std": min_margin_std,
                    "robust_slack": robust_slack,
                }
            )

    out3 = make_figure_domain_shift(outdir, domain_rows, delta_sweep)
    domain_csv = outdir / "domain_shift_sweep.csv"
    save_csv(
        domain_csv,
        domain_rows,
        [
            "delta",
            "method",
            "eval_cost_mean",
            "eval_cost_std",
            "violation_rate_mean",
            "min_margin_mean",
            "min_margin_std",
            "robust_slack",
        ],
    )

    diag = {
        "eig_P": [float(v) for v in safe_eigvals(P)],
        "eig_M0": [float(v) for v in safe_eigvals(M0)],
        "eig_M_safe": [float(v) for v in safe_eigvals(M_safe)],
        "eig_M_robust": [float(v) for v in safe_eigvals(M_robust)],
        "tau": float(tau),
        "min_eig_D_tau": float(np.min(np.linalg.eigvalsh(D_tau))),
        "Sigma0_trace": float(np.trace(Sigma0)),
        "Sigma_safe_trace": float(np.trace(Sigma_safe)),
        "Sigma_robust_trace": float(np.trace(Sigma_robust)),
        "normal_var_lambda0": float(normal_vars_safe[0]),
        "normal_var_lambda1000": float(normal_vars_safe[-1]),
        "tangent_var_lambda0": float(tangent_vars_safe[0]),
        "tangent_var_lambda1000": float(tangent_vars_safe[-1]),
        "nlr_lambda1000": float(nlr_safe[-1]),
        "tcr_lambda1000": float(tcr_safe[-1]),
        "monotonic_normal_collapse_passed": bool(
            all(
                normal_vars_safe[i + 1] <= normal_vars_safe[i] + 1e-8
                for i in range(len(normal_vars_safe) - 1)
            )
            and normal_vars_safe[-1] < normal_vars_safe[0]
        ),
    }
    diag_path = outdir / "diagnostics.json"
    diag_path.write_text(json.dumps(diag, indent=2))

    residual = P - (Q + rho * A.T @ P @ A - rho * A.T @ P @ B @ np.linalg.solve(R + rho * B.T @ P @ B, rho * B.T @ P @ A))
    residual_norm = float(np.max(np.abs(residual)))
    min_eig_P = float(np.min(np.linalg.eigvalsh(P)))
    min_margin_safe = float(normal_vars_safe[0])

    print("[Curvature LQR Step1 Summary]")
    print(f"Riccati residual: {residual_norm:.6e}")
    print(f"min eig P: {min_eig_P:.6e}")
    print(f"tau: {tau:.6e}")
    print(f"min eig D_tau: {diag['min_eig_D_tau']:.6e}")
    print(f"normal variance lambda=0: {normal_vars_safe[0]:.6e}")
    print(f"normal variance lambda=1000: {normal_vars_safe[-1]:.6e}")
    print(f"tangent variance lambda=0: {tangent_vars_safe[0]:.6e}")
    print(f"tangent variance lambda=1000: {tangent_vars_safe[-1]:.6e}")
    print(f"NLR lambda=1000: {nlr_safe[-1]:.6e}")
    print(f"TCR lambda=1000: {tcr_safe[-1]:.6e}")
    print(f"outputs saved to: {outdir}")


if __name__ == "__main__":
    main()
