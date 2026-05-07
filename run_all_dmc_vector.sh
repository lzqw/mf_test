#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH
# 如果任何命令失败，脚本将立即退出
set -e


# --- 指令列表 ---
#python scripts/train_dmc_vector.py --env dm_control_vector_humanoid_walk-v0  --alg sac --delay_alpha_update 200

#python scripts/train_dmc_vector.py --env dm_control_vector_dog_walk-v0 --alg sac  --delay_alpha_update 200

#python scripts/train_dmc_vector.py --env dm_control_vector_manipulator_ball-v0 --alg sac --delay_alpha_update 200

python scripts/train_dmc_vector.py --env dm_control_vector_humanoid_walk-v0  --alg rf2_sac_ent --delay_alpha_update 20 --fix_alpha True --noise_scale 0.01
python scripts/train_dmc_vector.py --env dm_control_vector_humanoid_walk-v0  --alg rf2_sac_ent --delay_alpha_update 20

python scripts/train_dmc_vector.py --env dm_control_vector_dog_walk-v0 --alg rf2_sac_ent  --delay_alpha_update 20

#python scripts/train_dmc_vector.py --env dm_control_vector_manipulator_insert_ball-v0 --alg rf2_sac_ent --delay_alpha_update 20
