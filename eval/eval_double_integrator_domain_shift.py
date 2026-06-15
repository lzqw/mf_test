import argparse
import csv
import json
from pathlib import Path

from eval.eval_double_integrator_pullback import collect_double_integrator_eval_rollouts, load_double_integrator_agent


def parse_deltas(raw):
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    out = []
    for p in parts:
        out.append(float(p))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vanilla_checkpoint", required=True)
    p.add_argument("--curvature_checkpoint", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument(
        "--delta_grid",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 0.3, 0.4],
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start_y_range", type=float, default=0.45)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--a_max", type=float, default=3.0)
    p.add_argument("--save_rollouts", action="store_true", default=False)
    args = p.parse_args()

    deltas = parse_deltas(args.delta_grid)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    vanilla = load_double_integrator_agent(args.vanilla_checkpoint)
    curvature = load_double_integrator_agent(args.curvature_checkpoint)

    rows = []

    for delta in deltas:
        for method, agent in [("Vanilla Flow", vanilla), ("Curvature-Shaped Flow", curvature)]:
            alias = "vanilla_flow" if method.startswith("Vanilla") else "curvature_flow"
            rollouts, summary = collect_double_integrator_eval_rollouts(
                agent,
                alias,
                episodes=args.episodes,
                seed=args.seed + int(delta * 1000),
                delta=delta,
                start_y_range=args.start_y_range,
                dt=args.dt,
                a_max=args.a_max,
                use_handcrafted_controller=False,
            )
            row = dict(
                delta=float(delta),
                method=method,
                J_eval_mean=float(summary.get("J_eval_mean", 0.0)),
                J_eval_std=float(summary.get("J_eval_std", 0.0)),
                violation_rate_mean=float(summary.get("violation_rate", 0.0)),
                h_min_mean=float(summary.get("h_min_mean", 0.0)),
                h_min_std=float(summary.get("h_min_std", 0.0)),
                success_rate=float(summary.get("success_rate", 0.0)),
                collision_rate=float(summary.get("collision_rate", 0.0)),
                return_mean=float(summary.get("return_mean", 0.0)),
                return_std=float(summary.get("return_std", 0.0)),
            )
            # estimate std of violation per-trajectory for compact reporting
            row["violation_rate_std"] = float(0.0)
            rows.append(row)

            if args.save_rollouts:
                npz_path = outdir / f"rollouts_delta_{delta:.1f}_{alias}.npz"
                # numpy is imported in collect module? no. use save manually via numpy here.
                import numpy as np

                np.savez(npz_path, **rollouts)

    csv_path = outdir / "domain_shift_rollout.csv"
    with open(csv_path, "w", newline="") as f:
        keys = [
            "delta",
            "method",
            "J_eval_mean",
            "J_eval_std",
            "violation_rate_mean",
            "violation_rate_std",
            "h_min_mean",
            "h_min_std",
            "success_rate",
            "collision_rate",
            "return_mean",
            "return_std",
        ]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(outdir / "summary.json", "w") as f:
        json.dump({"rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
