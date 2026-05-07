#!/bin/bash
# \u5982\u679c\u4efb\u4f55\u547d\u4ee4\u5931\u8d25\uff0c\u811a\u672c\u5c06\u7acb\u5373\u9000\u51fa
set -e
export PYTHONPATH=$(pwd):$PYTHONPATH

# --- \u6307\u4ee4\u5217\u8868 ---

python scripts/train_mujoco.py --env Hopper-v5 --diffusion_steps 20 --alg rf2  --noise_scale 0.001 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 1 --seed 1

python scripts/train_mujoco.py --env Ant-v5 --diffusion_steps 20 --alg rf2  --noise_scale 0.001 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 1 --seed 1

python scripts/train_mujoco.py --env HalfCheetah-v5 --diffusion_steps 20 --alg rf2  --noise_scale 0.001 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 1 --seed 1

python scripts/train_mujoco.py --env Walker2d-v5 --diffusion_steps 20 --alg rf2  --noise_scale 0.001 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 1 --seed 1

python scripts/train_mujoco.py --env Swimmer-v5 --diffusion_steps 20 --alg rf2  --noise_scale 0.001 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 1 --seed 1

python scripts/train_mujoco.py --env InvertedPendulum-v5 --diffusion_steps 20 --alg rf2  --noise_scale 0.001 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 1 --seed 1

python scripts/train_mujoco.py --env Reacher-v5 --diffusion_steps 20 --alg rf2  --noise_scale 0.001 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 1 --seed 1

python scripts/train_mujoco.py --env Pusher-v5 --diffusion_steps 20 --alg rf2  --noise_scale 0.001 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 1 --seed 1

python scripts/train_mujoco.py --env Humanoid-v5 --diffusion_steps 20 --alg rf2  --noise_scale 0.001 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 1 --seed 1

python scripts/train_mujoco.py --env InvertedDoublePendulum-v5 --diffusion_steps 20 --alg rf2  --noise_scale 0.001 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 1 --seed 1


python scripts/train_mujoco.py --env Hopper-v5 --diffusion_steps 20 --alg rf2  --noise_scale 0.001 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 1 --seed 2

python scripts/train_mujoco.py --env Ant-v5 --diffusion_steps 20 --alg rf2  --noise_scale 0.001 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 1 --seed 2

python scripts/train_mujoco.py --env HalfCheetah-v5 --diffusion_steps 20 --alg rf2  --noise_scale 0.001 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 1 --seed 2

python scripts/train_mujoco.py --env Walker2d-v5 --diffusion_steps 20 --alg rf2  --noise_scale 0.001 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 1 --seed 2

python scripts/train_mujoco.py --env Swimmer-v5 --diffusion_steps 20 --alg rf2  --noise_scale 0.001 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 1 --seed 2

python scripts/train_mujoco.py --env InvertedPendulum-v5 --diffusion_steps 20 --alg rf2  --noise_scale 0.001 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 1 --seed 2

python scripts/train_mujoco.py --env Reacher-v5 --diffusion_steps 20 --alg rf2  --noise_scale 0.001 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 1 --seed 2

python scripts/train_mujoco.py --env Pusher-v5 --diffusion_steps 20 --alg rf2  --noise_scale 0.001 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 1 --seed 2

python scripts/train_mujoco.py --env Humanoid-v5 --diffusion_steps 20 --alg rf2  --noise_scale 0.001 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 1 --seed 2

python scripts/train_mujoco.py --env InvertedDoublePendulum-v5 --diffusion_steps 20 --alg rf2  --noise_scale 0.001 --target_entropy_scale 1.0 --total_step 1000000 --diffusion_steps_test 1 --seed 2

