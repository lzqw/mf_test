#!/usr/bin/env bash
set -euo pipefail

SEED=${SEED:-0}
SCENE_SEED=${SCENE_SEED:-0}
TOTAL_STEPS=${TOTAL_STEPS:-100000}

BASE_ARGS=(
  --env_id SafetyPointGoal1-v0
  --seed "$SEED"
  --total_steps "$TOTAL_STEPS"
  --start_steps 1000
  --update_after 1000
  --eval_interval 5000
  --eval_episodes 5
  --batch_size 256
  --sample_k 64
  --fixed_scene
  --scene_seed "$SCENE_SEED"
  --eval_scene_seed "$SCENE_SEED"
  --eval_stop_on_goal
)

python scripts/train_safe_safetygym.py "${BASE_ARGS[@]}" \
  --policy_noise_scale 0.05 \
  --fixed_alpha --alpha_value 0.01 \
  --eval_filter_mode off \
  --log_dir "logs/safetygym_fixed_raweval100k/point_goal/rf2_noise005_alpha001/scene${SCENE_SEED}_seed${SEED}"

python scripts/train_safe_safetygym.py "${BASE_ARGS[@]}" \
  --use_filter --filter_type gt_shield \
  --use_filter_surrogate --use_tn_energy --use_projection_critic \
  --lambda_p 0.03 --policy_noise_scale 0.01 \
  --fixed_alpha --alpha_value 0.03 \
  --eval_filter_mode both \
  --log_dir "logs/safetygym_fixed_raweval100k/point_goal/ours_lamp003/scene${SCENE_SEED}_seed${SEED}"

python scripts/train_safe_safetygym.py "${BASE_ARGS[@]}" \
  --use_filter --filter_type gt_shield \
  --use_filter_surrogate --use_tn_energy --use_projection_critic \
  --lambda_p 0.01 --policy_noise_scale 0.01 \
  --fixed_alpha --alpha_value 0.01 \
  --eval_filter_mode both \
  --log_dir "logs/safetygym_fixed_raweval100k/point_goal/ours_lamp001/scene${SCENE_SEED}_seed${SEED}"
