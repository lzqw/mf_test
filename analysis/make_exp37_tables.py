import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METHOD_MAP = {
    "likelihood_tn_alpha005_temp007": "Full-LTN",
    "main_likelihood_tn_alpha005_temp007": "Full-LTN",
    "main_likelihood_tn_alpha005_temp007_mix010": "Full-LTN (mix=0.10)",
    "flac_tn_alpha002_temp012_mix010": "FLAC-TN",
    "ablation_no_tn_entropy": "w/o TN entropy",
    "ablation_no_weight_mix": "w/o weight mix",
    "ablation_flac_normal_tangent": "normal+tangent energy",
}

SELECTED_METHODS = [
    "Full-LTN",
    "Full-LTN (mix=0.10)",
    "FLAC-TN",
    "w/o TN entropy",
    "w/o weight mix",
    "normal+tangent energy",
]

METRICS = [
    "success_rate", "episode_return_mean", "time_to_goal_mean", "FAR", "APR",
    "feasible_raw_action_ratio", "route_entropy", "effective_route_entropy",
]


def fmt_stat(vals):
    vals = pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return "NaN (n=0)"
    std = np.std(vals, ddof=1) if vals.size > 1 else 0.0
    return f"{np.mean(vals):.4f} ± {std:.4f} (n={vals.size})"


def build_table(df, metrics, method_order=None):
    g = df.groupby("method_paper")
    rows = []
    for m, sdf in g:
        row = {"method": m}
        for k in metrics:
            row[k] = fmt_stat(sdf[k]) if k in sdf.columns else "NaN (n=0)"
        rows.append(row)
    out = pd.DataFrame(rows)
    if method_order is not None and not out.empty:
        out["_order"] = out["method"].map({m: i for i, m in enumerate(method_order)}).fillna(999)
        out = out.sort_values(["_order", "method"]).drop(columns=["_order"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="paper_outputs/tables/exp37_all_eval_summary.csv")
    ap.add_argument("--out_dir", default="paper_outputs/tables")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df = df[df["eval_type"] == "final_eval"].copy()
    df["method_paper"] = df["method"].map(METHOD_MAP).fillna(df["method"])

    main_df = build_table(df[df["method_paper"].isin(SELECTED_METHODS)], METRICS, method_order=SELECTED_METHODS)
    abl_df = build_table(df[df["method"].str.contains("ablation", na=False)], METRICS)

    for name, tdf in [("exp37_main_table", main_df), ("exp37_ablation_table", abl_df)]:
        tdf.to_csv(out_dir / f"{name}.csv", index=False)
        (out_dir / f"{name}.md").write_text(tdf.to_markdown(index=False))
        (out_dir / f"{name}.tex").write_text(tdf.to_latex(index=False, escape=True))
    print("tables written")


if __name__ == "__main__":
    main()
