import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def logistic_curve(x, start, end, rate, shift):
    return start + (end - start) / (1.0 + np.exp(-rate * (x - shift)))


def synthetic_seeds_from_mean(mean_curve, rng, num_seeds, std_scale, non_negative=False, cumulative=False):
    seeds = []
    t = np.linspace(0.6, 1.0, mean_curve.shape[0])
    for _ in range(num_seeds):
        noise = rng.normal(0.0, std_scale, size=mean_curve.shape[0]) * t
        arr = mean_curve + noise
        if non_negative:
            arr = np.clip(arr, 0.0, None)
        if cumulative:
            arr = np.maximum.accumulate(arr)
        seeds.append(arr)
    return np.stack(seeds, axis=0)


def save_csv(path: Path, steps, data):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "method", "seed", "value"])
        for method, seeds_arr in data.items():
            for seed_idx in range(seeds_arr.shape[0]):
                for i, step in enumerate(steps):
                    writer.writerow([int(step), method, seed_idx, float(seeds_arr[seed_idx, i])])


def plot_curve(steps, data, ylabel, title, out_png, out_pdf):
    plt.figure(figsize=(8, 5))
    for method, seeds_arr in data.items():
        mean = seeds_arr.mean(axis=0)
        std = seeds_arr.std(axis=0)
        (line,) = plt.plot(steps, mean, label=method)
        plt.fill_between(steps, mean - std, mean + std, alpha=0.2, color=line.get_color())
    plt.xlabel("Environment steps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.text(0.98, 0.04, "Estimated / illustrative", transform=plt.gca().transAxes, ha="right", va="bottom", fontsize=10)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="figures/estimated_humanoidbench")
    parser.add_argument("--max_env_steps", type=int, default=40000)
    parser.add_argument("--num_points", type=int, default=11)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--ours_final_return", type=float, default=14500)
    parser.add_argument("--no_qc_final_return", type=float, default=12500)
    parser.add_argument("--sac_final_return", type=float, default=6500)

    parser.add_argument("--ours_final_crash", type=float, default=1)
    parser.add_argument("--no_qc_final_crash", type=float, default=7)
    parser.add_argument("--sac_final_crash", type=float, default=44)
    args = parser.parse_args()

    print("WARNING: These curves are estimated/illustrative and are not real multi-seed statistics.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    steps = np.linspace(0, args.max_env_steps, args.num_points)

    ours_mean = logistic_curve(steps, 1500, args.ours_final_return, 0.00028, 7000)
    no_qc_mean = logistic_curve(steps, 1200, args.no_qc_final_return, 0.00023, 8500)
    sac_mean = logistic_curve(steps, 800, args.sac_final_return, 0.00016, 12000)

    reward_data = {
        "Ours": synthetic_seeds_from_mean(ours_mean, rng, args.num_seeds, std_scale=650),
        "No-Qc": synthetic_seeds_from_mean(no_qc_mean, rng, args.num_seeds, std_scale=950),
        "SAC": synthetic_seeds_from_mean(sac_mean, rng, args.num_seeds, std_scale=1200),
    }

    crash_base_ours = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
    crash_base_no_qc = np.array([0, 1, 2, 3, 4, 5, 5, 6, 6, 6, 7], dtype=float)
    crash_base_sac = np.array([0, 4, 9, 15, 21, 27, 32, 36, 39, 42, 44], dtype=float)

    if args.num_points != 11:
        x_src = np.linspace(0, 1, 11)
        x_dst = np.linspace(0, 1, args.num_points)
        crash_base_ours = np.interp(x_dst, x_src, crash_base_ours)
        crash_base_no_qc = np.interp(x_dst, x_src, crash_base_no_qc)
        crash_base_sac = np.interp(x_dst, x_src, crash_base_sac)

    crash_base_ours *= (args.ours_final_crash / max(crash_base_ours[-1], 1e-6))
    crash_base_no_qc *= (args.no_qc_final_crash / max(crash_base_no_qc[-1], 1e-6))
    crash_base_sac *= (args.sac_final_crash / max(crash_base_sac[-1], 1e-6))

    crash_data = {
        "Ours": synthetic_seeds_from_mean(crash_base_ours, rng, args.num_seeds, std_scale=0.4, non_negative=True, cumulative=True),
        "No-Qc": synthetic_seeds_from_mean(crash_base_no_qc, rng, args.num_seeds, std_scale=0.8, non_negative=True, cumulative=True),
        "SAC": synthetic_seeds_from_mean(crash_base_sac, rng, args.num_seeds, std_scale=1.6, non_negative=True, cumulative=True),
    }

    plot_curve(
        steps,
        reward_data,
        ylabel="Evaluation return",
        title="Estimated HumanoidBench Reach learning curves",
        out_png=out_dir / "estimated_reward_curve.png",
        out_pdf=out_dir / "estimated_reward_curve.pdf",
    )
    plot_curve(
        steps,
        crash_data,
        ylabel="Cumulative safety violations / crashes",
        title="Estimated training safety cost",
        out_png=out_dir / "estimated_crash_curve.png",
        out_pdf=out_dir / "estimated_crash_curve.pdf",
    )

    np.savez(
        out_dir / "estimated_curves_data.npz",
        steps=steps,
        reward_ours=reward_data["Ours"],
        reward_no_qc=reward_data["No-Qc"],
        reward_sac=reward_data["SAC"],
        crash_ours=crash_data["Ours"],
        crash_no_qc=crash_data["No-Qc"],
        crash_sac=crash_data["SAC"],
    )

    save_csv(out_dir / "estimated_reward_curve.csv", steps, reward_data)
    save_csv(out_dir / "estimated_crash_curve.csv", steps, crash_data)

    print("Saved:")
    print(out_dir / "estimated_reward_curve.png")
    print(out_dir / "estimated_crash_curve.png")


if __name__ == "__main__":
    main()
