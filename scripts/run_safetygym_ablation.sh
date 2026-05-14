#!/usr/bin/env bash
set -euo pipefail
ENV_ID=$1
TOTAL_STEPS=${2:-50000}
SEED=${3:-0}
case "$ENV_ID" in
  SafetyPointGoal1-v0) ENV_SHORT=point_goal ;;
  SafetyPointPush1-v0) ENV_SHORT=point_push ;;
  SafetyCarGoal1-v0) ENV_SHORT=car_goal ;;
  SafetyCarPush1-v0) ENV_SHORT=car_push ;;
  *) echo "Unsupported ENV_ID: $ENV_ID"; exit 1 ;;
esac
BASE="python scripts/train_safe_safetygym.py --env_id ${ENV_ID} --seed ${SEED} --total_steps ${TOTAL_STEPS} --start_steps 1000 --update_after 1000 --eval_interval 1000 --eval_episodes 3 --batch_size 256"
$BASE --log_dir logs/safetygym/${ENV_SHORT}/ours/seed${SEED} --use_filter --filter_type hybrid --use_tn_energy --use_projection_critic --lambda_p 0.03 --entropy_reg_mode flac_tn
$BASE --log_dir logs/safetygym/${ENV_SHORT}/no_tn/seed${SEED} --use_filter --filter_type hybrid --use_projection_critic --lambda_p 0.03
$BASE --log_dir logs/safetygym/${ENV_SHORT}/tn_only/seed${SEED} --use_filter --filter_type hybrid --use_tn_energy --lambda_p 0.0 --entropy_reg_mode flac_tn
$BASE --log_dir logs/safetygym/${ENV_SHORT}/no_filter/seed${SEED} --filter_type none --use_tn_energy --lambda_p 0.0
