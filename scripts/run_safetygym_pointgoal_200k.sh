#!/usr/bin/env bash
set -euo pipefail
cd ~/FLAME
source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate diffusion_policy
export PYTHONPATH=.
unset MUJOCO_GL
unset PYOPENGL_PLATFORM
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.36}
MAX_PARALLEL=${MAX_PARALLEL:-2}
TOTAL_STEPS=${TOTAL_STEPS:-200000}
START_STEPS=10000
UPDATE_AFTER=10000
EVAL_INTERVAL=10000
EVAL_EPISODES=10
SEED=${SEED:-0}
RUN_OURS=${RUN_OURS:-0}
mkdir -p logs/run_logs logs/safetygym_long/point_goal

declare -a CMDS
CMDS+=("sb3_sac_200k::python scripts/train_safetygym_sb3_sac.py --env_id SafetyPointGoal1-v0 --seed $SEED --total_steps $TOTAL_STEPS --eval_interval $EVAL_INTERVAL --eval_episodes $EVAL_EPISODES --batch_size 256 --learning_starts 10000 --log_dir logs/safetygym_long/point_goal/sb3_sac_200k/seed$SEED")
CMDS+=("rf2_sac_ent_noise001_alpha001_200k::python scripts/train_safe_safetygym.py --env_id SafetyPointGoal1-v0 --seed $SEED --total_steps $TOTAL_STEPS --start_steps $START_STEPS --update_after $UPDATE_AFTER --eval_interval $EVAL_INTERVAL --eval_episodes $EVAL_EPISODES --batch_size 128 --sample_k 128 --policy_noise_scale 0.01 --fixed_alpha --alpha_value 0.01 --log_dir logs/safetygym_long/point_goal/rf2_sac_ent_noise001_alpha001_200k/seed$SEED --filter_type none --lambda_p 0.0 --entropy_reg_mode legacy")
CMDS+=("rf2_sac_ent_noise005_alpha001_200k::python scripts/train_safe_safetygym.py --env_id SafetyPointGoal1-v0 --seed $SEED --total_steps $TOTAL_STEPS --start_steps $START_STEPS --update_after $UPDATE_AFTER --eval_interval $EVAL_INTERVAL --eval_episodes $EVAL_EPISODES --batch_size 128 --sample_k 128 --policy_noise_scale 0.05 --fixed_alpha --alpha_value 0.01 --log_dir logs/safetygym_long/point_goal/rf2_sac_ent_noise005_alpha001_200k/seed$SEED --filter_type none --lambda_p 0.0 --entropy_reg_mode legacy")
if [[ "$RUN_OURS" == "1" ]]; then
CMDS+=("ours_gt_shield_lamp01_200k::python scripts/train_safe_safetygym.py --env_id SafetyPointGoal1-v0 --seed $SEED --total_steps $TOTAL_STEPS --start_steps $START_STEPS --update_after $UPDATE_AFTER --eval_interval $EVAL_INTERVAL --eval_episodes $EVAL_EPISODES --batch_size 128 --sample_k 128 --policy_noise_scale 0.01 --fixed_alpha --alpha_value 0.01 --log_dir logs/safetygym_long/point_goal/ours_gt_shield_lamp01_200k/seed$SEED --use_filter --filter_type hybrid --lambda_p 0.1 --entropy_reg_mode legacy")
fi
pids=()
for item in "${CMDS[@]}"; do
  name=${item%%::*}; cmd=${item#*::}
  while [[ $(jobs -rp | wc -l) -ge $MAX_PARALLEL ]]; do sleep 5; done
  bash -lc "$cmd" > "logs/run_logs/${name}_seed${SEED}.log" 2>&1 &
  pids+=("$!")
  sleep 30
done
for p in "${pids[@]}"; do wait "$p"; done
python scripts/summarize_safetygym_runs.py --root logs/safetygym_long --out logs/safetygym_long/point_goal_summary.txt
