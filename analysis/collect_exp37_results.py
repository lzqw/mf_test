import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List

METRICS = [
    "success_rate", "collision_rate", "state_violation_rate", "episode_return_mean", "return",
    "time_to_goal_mean", "FAR", "APR", "feasible_raw_action_ratio",
    "route_upper_ratio", "route_lower_ratio", "route_entropy",
    "effective_route_upper_ratio", "effective_route_lower_ratio", "effective_route_entropy",
    "raw_action_std", "exec_action_std", "projection_residual_mean_nonzero",
    "filter_activation_episode_rate",
]


def _extract(parts: List[str], patt: str):
    rgx = re.compile(patt)
    for p in parts:
        m = rgx.fullmatch(p)
        if m:
            return m
    return None


def parse_meta(summary_path: Path) -> Dict[str, object]:
    parts = summary_path.parts
    exp = None
    for p in parts:
        if re.fullmatch(r"exp[3-7]", p):
            exp = p
            break
    seed_m = _extract(list(parts), r"seed[_-]?(\d+)")
    seed = int(seed_m.group(1)) if seed_m else math.nan

    eval_dir = summary_path.parent.name
    eval_type = "unknown"
    start_y_range = ""
    if eval_dir == "final_eval":
        eval_type = "final_eval"
    elif eval_dir.startswith("generalization_start_y_"):
        eval_type = "generalization"
        start_y_range = eval_dir.replace("generalization_start_y_", "")

    method = "unknown"
    if exp and exp in parts:
        ei = parts.index(exp)
        if seed_m:
            # first seed marker occurrence after exp
            seed_idx = None
            for i in range(ei + 1, len(parts)):
                if re.fullmatch(r"seed[_-]?\d+", parts[i]):
                    seed_idx = i
                    break
            if seed_idx and seed_idx > ei + 1:
                method = "/".join(parts[ei + 1:seed_idx])
        if method == "unknown" and ei + 1 < len(parts):
            method = parts[ei + 1]

    return {
        "exp": exp or "unknown",
        "method": method,
        "seed": seed,
        "eval_type": eval_type,
        "start_y_range": start_y_range,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--output", default="paper_outputs/tables/exp37_all_eval_summary.csv")
    args = ap.parse_args()

    rows = []
    for root in args.roots:
        rp = Path(root)
        if not rp.exists():
            continue
        for p in rp.rglob("summary.json"):
            pn = p.parent.name
            if pn != "final_eval" and not pn.startswith("generalization_start_y_"):
                continue
            meta = parse_meta(p)
            try:
                data = json.loads(p.read_text())
            except Exception:
                data = {}
            row = {**meta, "summary_path": str(p)}
            for k in METRICS:
                v = data.get(k, math.nan)
                row[k] = v
            rows.append(row)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["exp", "method", "seed", "eval_type", "start_y_range", "summary_path"] + METRICS
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
