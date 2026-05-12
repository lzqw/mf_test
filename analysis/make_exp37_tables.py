import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METHOD_MAP = {
    "main_likelihood_tn_alpha005_temp007": "Full-LTN",
    "main_likelihood_tn_alpha005_temp007_mix010": "Full-LTN (mix=0.10)",
    "flac_tn_alpha002_temp012_mix010": "FLAC-TN",
    "ablation_no_tn_entropy": "w/o TN entropy",
    "ablation_no_weight_mix": "w/o weight mix",
    "ablation_flac_normal_tangent": "normal+tangent energy",
}

METRICS = ["success_rate", "episode_return_mean", "time_to_goal_mean", "FAR", "APR"]


def fmt(mean, std):
    if np.isnan(mean):
        return "NaN"
    return f"{mean:.4f} ± {std:.4f}"


def build_table(df, metrics):
    g = df.groupby("method_paper")
    rows = []
    for m, sdf in g:
        row = {"method": m}
        for k in metrics:
            vals = pd.to_numeric(sdf[k], errors="coerce").to_numpy(dtype=float)
            row[k] = fmt(np.nanmean(vals), np.nanstd(vals, ddof=1) if np.sum(~np.isnan(vals)) > 1 else 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(df):
    return df.to_markdown(index=False)


def to_latex(df):
    return df.to_latex(index=False, escape=True)


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

    main_df = build_table(df, METRICS)
    abl_df = build_table(df[df["method"].str.contains("ablation", na=False)], METRICS)

    for name, tdf in [("exp37_main_table", main_df), ("exp37_ablation_table", abl_df)]:
        tdf.to_csv(out_dir / f"{name}.csv", index=False)
        (out_dir / f"{name}.md").write_text(to_markdown(tdf))
        (out_dir / f"{name}.tex").write_text(to_latex(tdf))
    print("tables written")


if __name__ == "__main__":
    main()
