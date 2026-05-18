#!/usr/bin/env bash
set -euo pipefail

SEED=${SEED:-0}
SCENE_SEED=${SCENE_SEED:-0}
TOTAL_STEPS=${TOTAL_STEPS:-100000}
START_STEPS=${START_STEPS:-5000}
UPDATE_AFTER=${UPDATE_AFTER:-5000}
EVAL_INTERVAL=${EVAL_INTERVAL:-10000}

COMMON="--env_id SafetyPointGoal1-v0 --seed ${SEED} --fixed_scene --scene_seed ${SCENE_SEED} --eval_scene_seed ${SCENE_SEED} --total_steps ${TOTAL_STEPS} --start_steps ${START_STEPS} --update_after ${UPDATE_AFTER} --eval_interval ${EVAL_INTERVAL} --eval_episodes 1 --batch_size 128 --sample_k 128 --policy_noise_scale 0.01 --fixed_alpha --alpha_value 0.01 --eval_stop_on_goal"
BASE=logs/safetygym_selfexec100k/point_goal

python scripts/train_safe_safetygym.py $COMMON --log_dir ${BASE}/rf2_noise005_alpha001_raw/scene${SCENE_SEED}_seed${SEED} --eval_filter_mode off

python scripts/train_safe_safetygym.py $COMMON --use_filter --filter_type hybrid --eval_filter_mode both --train_terminate_on_goal --lambda_p 0.03 --include_exec_candidate --num_exec_local_candidates 16 --exec_bc_coef 0.5 --self_exec_coef 0.0 --safe_actor_scale_mode alpha --log_dir ${BASE}/ours_lamp003_execbc/scene${SCENE_SEED}_seed${SEED}

python scripts/train_safe_safetygym.py $COMMON --use_filter --filter_type hybrid --eval_filter_mode both --train_terminate_on_goal --lambda_p 0.03 --include_exec_candidate --num_exec_local_candidates 16 --exec_bc_coef 0.5 --self_exec_coef 0.3 --safe_actor_scale_mode direct --actor_safe_coef 0.1 --log_dir ${BASE}/ours_lamp003_selfexec/scene${SCENE_SEED}_seed${SEED}
