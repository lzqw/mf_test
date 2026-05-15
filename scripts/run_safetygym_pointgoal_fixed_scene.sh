#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate diffusion_policy
export PYTHONPATH=.
unset MUJOCO_GL
unset PYOPENGL_PLATFORM
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.42}

MAX_PARALLEL=${MAX_PARALLEL:-2}
TOTAL_STEPS=${TOTAL_STEPS:-100000}
START_STEPS=${START_STEPS:-5000}
UPDATE_AFTER=${UPDATE_AFTER:-5000}
EVAL_INTERVAL=${EVAL_INTERVAL:-5000}
EVAL_EPISODES=${EVAL_EPISODES:-1}
BATCH_SIZE=${BATCH_SIZE:-128}
SAMPLE_K=${SAMPLE_K:-128}
SEED=${SEED:-0}
SCENE_SEED=${SCENE_SEED:-0}
ENV_ID=${ENV_ID:-SafetyPointGoal1-v0}

mkdir -p logs/run_logs logs/safetygym_fixed/point_goal

python -m py_compile \
  envs/safety_gym_safe_wrapper.py \
  relax/safety/safety_gym_filter.py \
  scripts/train_safe_safetygym.py \
  scripts/summarize_safetygym_runs.py

declare -a CMDS
CMDS+=("rf2_sac_ent_noise001_alpha001_fixed::python scripts/train_safe_safetygym.py --env_id $ENV_ID --seed $SEED --total_steps $TOTAL_STEPS --start_steps $START_STEPS --update_after $UPDATE_AFTER --eval_interval $EVAL_INTERVAL --eval_episodes $EVAL_EPISODES --batch_size $BATCH_SIZE --sample_k $SAMPLE_K --filter_type none --policy_noise_scale 0.01 --fixed_alpha --alpha_value 0.01 --lambda_p 0.0 --lambda_raw_norm 0.0 --entropy_reg_mode legacy --fixed_scene --scene_seed $SCENE_SEED --eval_scene_seed $SCENE_SEED --save_eval_trajectories --eval_traj_episodes 1 --eval_traj_stride 25 --log_dir logs/safetygym_fixed/point_goal/rf2_sac_ent_noise001_alpha001_fixed/scene${SCENE_SEED}_seed${SEED}")
CMDS+=("rf2_sac_ent_noise005_alpha001_fixed::python scripts/train_safe_safetygym.py --env_id $ENV_ID --seed $SEED --total_steps $TOTAL_STEPS --start_steps $START_STEPS --update_after $UPDATE_AFTER --eval_interval $EVAL_INTERVAL --eval_episodes $EVAL_EPISODES --batch_size $BATCH_SIZE --sample_k $SAMPLE_K --filter_type none --policy_noise_scale 0.05 --fixed_alpha --alpha_value 0.01 --lambda_p 0.0 --lambda_raw_norm 0.0 --entropy_reg_mode legacy --fixed_scene --scene_seed $SCENE_SEED --eval_scene_seed $SCENE_SEED --save_eval_trajectories --eval_traj_episodes 1 --eval_traj_stride 25 --log_dir logs/safetygym_fixed/point_goal/rf2_sac_ent_noise005_alpha001_fixed/scene${SCENE_SEED}_seed${SEED}")
CMDS+=("ours_gt_shield_lamp01_fixed::python scripts/train_safe_safetygym.py --env_id $ENV_ID --seed $SEED --total_steps $TOTAL_STEPS --start_steps $START_STEPS --update_after $UPDATE_AFTER --eval_interval $EVAL_INTERVAL --eval_episodes $EVAL_EPISODES --batch_size $BATCH_SIZE --sample_k $SAMPLE_K --use_filter --filter_type gt_shield --use_filter_surrogate --surrogate_warmup_steps 0 --surrogate_loss_coef 1.0 --use_tn_energy --use_projection_critic --lambda_p 0.1 --lambda_raw_norm 0.0 --entropy_reg_mode flac_tn --policy_noise_scale 0.01 --fixed_alpha --alpha_value 0.01 --fixed_scene --scene_seed $SCENE_SEED --eval_scene_seed $SCENE_SEED --save_eval_trajectories --eval_traj_episodes 1 --eval_traj_stride 25 --log_dir logs/safetygym_fixed/point_goal/ours_gt_shield_lamp01_fixed/scene${SCENE_SEED}_seed${SEED}")
CMDS+=("ours_gt_shield_lamp003_fixed::python scripts/train_safe_safetygym.py --env_id $ENV_ID --seed $SEED --total_steps $TOTAL_STEPS --start_steps $START_STEPS --update_after $UPDATE_AFTER --eval_interval $EVAL_INTERVAL --eval_episodes $EVAL_EPISODES --batch_size $BATCH_SIZE --sample_k $SAMPLE_K --use_filter --filter_type gt_shield --use_filter_surrogate --surrogate_warmup_steps 0 --surrogate_loss_coef 1.0 --use_tn_energy --use_projection_critic --lambda_p 0.03 --lambda_raw_norm 0.0 --entropy_reg_mode flac_tn --policy_noise_scale 0.01 --fixed_alpha --alpha_value 0.01 --fixed_scene --scene_seed $SCENE_SEED --eval_scene_seed $SCENE_SEED --save_eval_trajectories --eval_traj_episodes 1 --eval_traj_stride 25 --log_dir logs/safetygym_fixed/point_goal/ours_gt_shield_lamp003_fixed/scene${SCENE_SEED}_seed${SEED}")

pids=()
for item in "${CMDS[@]}"; do
  name=${item%%::*}; cmd=${item#*::}
  while [[ $(jobs -rp | wc -l) -ge $MAX_PARALLEL ]]; do sleep 5; done
  bash -lc "$cmd" > "logs/run_logs/${name}_scene${SCENE_SEED}_seed${SEED}.log" 2>&1 &
  pids+=("$!")
  sleep 10
done

for p in "${pids[@]}"; do
  wait "$p"
done

python scripts/summarize_safetygym_runs.py \
  --root logs/safetygym_fixed \
  --out logs/safetygym_fixed/point_goal_summary.txt

cat logs/safetygym_fixed/point_goal_summary.txt || true
