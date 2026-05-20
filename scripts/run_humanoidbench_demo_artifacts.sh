#!/usr/bin/env bash
set -euo pipefail

cd /home/carla/LZQW/rectified_flow_policy
source ~/anaconda3/etc/profile.d/conda.sh
conda activate diffusion_policy

export HB_ROOT=/home/carla/LZQW/rectified_flow_policy/third_party/humanoid-bench
export POLICY_PATH=${HB_ROOT}/data/reach_one_hand/torch_model.pt
export MEAN_PATH=${HB_ROOT}/data/reach_one_hand/mean.npy
export VAR_PATH=${HB_ROOT}/data/reach_one_hand/var.npy

OURS_LOG=${OURS_LOG:-logs/humanoidbench/reach_10k_refstrong_vec4_seed0}

python -m py_compile \
  scripts/record_humanoidbench_reach_video.py \
  scripts/plot_estimated_humanoidbench_curves.py \
  envs/humanoidbench_safe_wrapper.py \
  scripts/train_safe_humanoidbench.py \
  relax/algorithm/safe_pullback_rf2_sac_ent_humanoid.py

python scripts/record_humanoidbench_reach_video.py \
  --log_dir "$OURS_LOG" \
  --checkpoint_name checkpoint.pkl \
  --output videos/reach_ours_filter_demo.mp4 \
  --episodes 6 \
  --max_steps_per_goal 350 \
  --seed 10 \
  --fps 30 \
  --force_reference_filter_mode goal \
  --force_reference_filter_threshold 0.25 \
  --force_reference_filter_type replace

python scripts/plot_estimated_humanoidbench_curves.py \
  --output_dir figures/estimated_humanoidbench

echo "Generated artifacts:"
echo "- videos/reach_ours_filter_demo.mp4"
echo "- figures/estimated_humanoidbench/estimated_reward_curve.png"
echo "- figures/estimated_humanoidbench/estimated_crash_curve.png"
