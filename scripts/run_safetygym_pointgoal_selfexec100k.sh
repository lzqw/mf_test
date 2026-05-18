#!/usr/bin/env bash
set -euo pipefail

cd ~/FLAME
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion_policy

export PYTHONPATH=.
unset MUJOCO_GL || true
unset PYOPENGL_PLATFORM || true
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false

SEED=${SEED:-0}
SCENE_SEED=${SCENE_SEED:-0}
TOTAL_STEPS=${TOTAL_STEPS:-100000}
START_STEPS=${START_STEPS:-5000}
UPDATE_AFTER=${UPDATE_AFTER:-5000}
EVAL_INTERVAL=${EVAL_INTERVAL:-10000}
MAX_PARALLEL=${MAX_PARALLEL:-2}

BASE=logs/safetygym_selfexec100k/point_goal
RUN_LOG_DIR=logs/run_logs
mkdir -p "${RUN_LOG_DIR}"

python -m py_compile \
  envs/safety_gym_safe_wrapper.py \
  relax/safety/safety_gym_filter.py \
  relax/algorithm/safe_pullback_rf2_sac_ent_safetygym.py \
  scripts/train_safe_safetygym.py \
  scripts/summarize_safetygym_runs.py

COMMON=(
  --env_id SafetyPointGoal1-v0
  --seed "${SEED}"
  --fixed_scene
  --scene_seed "${SCENE_SEED}"
  --eval_scene_seed "${SCENE_SEED}"
  --total_steps "${TOTAL_STEPS}"
  --start_steps "${START_STEPS}"
  --update_after "${UPDATE_AFTER}"
  --eval_interval "${EVAL_INTERVAL}"
  --eval_episodes 1
  --batch_size 128
  --sample_k 128
  --save_eval_trajectories
  --eval_traj_episodes 1
  --eval_traj_stride 25
)

run_job() {
  local method="$1"
  shift
  local log_dir="${BASE}/${method}/scene${SCENE_SEED}_seed${SEED}"
  local log_file="${RUN_LOG_DIR}/${method}_scene${SCENE_SEED}_seed${SEED}.log"
  mkdir -p "${log_dir}"
  python scripts/train_safe_safetygym.py "${COMMON[@]}" --log_dir "${log_dir}" "$@" >"${log_file}" 2>&1 &
}

wait_for_slot() {
  while [ "$(jobs -pr | wc -l)" -ge "${MAX_PARALLEL}" ]; do
    sleep 1
  done
}

wait_for_slot
run_job rf2_noise005_alpha001_raw \
  --filter_type none \
  --policy_noise_scale 0.05 \
  --fixed_alpha \
  --alpha_value 0.01 \
  --eval_filter_mode off \
  --eval_stop_on_goal

wait_for_slot
run_job ours_lamp003_execbc \
  --use_filter \
  --filter_type gt_shield \
  --use_filter_surrogate \
  --surrogate_warmup_steps 0 \
  --surrogate_loss_coef 1.0 \
  --use_tn_energy \
  --use_projection_critic \
  --lambda_p 0.03 \
  --lambda_raw_norm 0.0 \
  --entropy_reg_mode flac_tn \
  --policy_noise_scale 0.01 \
  --fixed_alpha \
  --alpha_value 0.01 \
  --include_exec_candidate \
  --num_exec_local_candidates 16 \
  --exec_candidate_noise_scale 0.03 \
  --exec_bc_coef 0.5 \
  --self_exec_coef 0.0 \
  --safe_actor_scale_mode alpha \
  --eval_filter_mode both \
  --eval_stop_on_goal \
  --train_terminate_on_goal

wait_for_slot
run_job ours_lamp003_selfexec \
  --use_filter \
  --filter_type gt_shield \
  --use_filter_surrogate \
  --surrogate_warmup_steps 0 \
  --surrogate_loss_coef 1.0 \
  --use_tn_energy \
  --use_projection_critic \
  --lambda_p 0.03 \
  --lambda_raw_norm 0.0 \
  --entropy_reg_mode flac_tn \
  --policy_noise_scale 0.01 \
  --fixed_alpha \
  --alpha_value 0.01 \
  --include_exec_candidate \
  --num_exec_local_candidates 16 \
  --exec_candidate_noise_scale 0.03 \
  --exec_bc_coef 0.5 \
  --self_exec_coef 0.3 \
  --safe_actor_scale_mode direct \
  --actor_safe_coef 0.1 \
  --eval_filter_mode both \
  --eval_stop_on_goal \
  --train_terminate_on_goal

wait
python scripts/summarize_safetygym_runs.py --base_dir "${BASE}"
