#!/usr/bin/env bash
set -euo pipefail

METHODS=(rf2_filter safe_pullback_rf2 safe_pullback_rf2_no_entropy goal_filter)
for m in "${METHODS[@]}"; do
  python scripts/train_safe_obstacle_navigation.py --algo "$m" --seed 0 --total_steps 50000 --start_steps 5000 --update_after 5000 --batch_size 256 --log_dir "logs/obstacle/${m}_seed0"
  python eval/eval_safe_obstacle_navigation.py --checkpoint "logs/obstacle/${m}_seed0/checkpoint.pkl" --algo "$m" --eval_episodes 200 --save_dir "results/obstacle/$m"
done

python eval/eval_prefilter_risk_grid.py --checkpoint logs/obstacle/safe_pullback_rf2_seed0/checkpoint.pkl --save_dir results/obstacle/safe_pullback_rf2
python plot/plot_obstacle_rollouts.py --methods rf2_filter safe_pullback_rf2 safe_pullback_rf2_no_entropy goal_filter --base_dir results/obstacle --out_dir figures
python plot/plot_obstacle_occupancy.py --methods rf2_filter safe_pullback_rf2 safe_pullback_rf2_no_entropy goal_filter --base_dir results/obstacle --out_dir figures
python plot/plot_prefilter_risk_heatmap.py --input results/obstacle/safe_pullback_rf2/prefilter_risk_grid.npz --out_dir figures


python scripts/train_safe_obstacle_navigation.py \
  --algo safe_pullback_rf2 \
  --seed 0 \
  --total_steps 200000 \
  --start_steps 10000 \
  --update_after 10000 \
  --batch_size 256 \
  --sample_k 256 \
  --lambda_p 0.1 \
  --log_dir logs/obstacle/safe_pullback_rf2_seed0_fix

python eval/eval_safe_obstacle_navigation.py \
  --checkpoint logs/obstacle/safe_pullback_rf2_seed0_fix/checkpoint.pkl \
  --algo safe_pullback_rf2 \
  --eval_episodes 200 \
  --save_dir results/obstacle/safe_pullback_rf2_fix

python plot/plot_obstacle_rollouts.py \
  --methods safe_pullback_rf2_fix \
  --base_dir results/obstacle \
  --out_dir figures


python scripts/train_safe_obstacle_navigation.py \
  --algo safe_pullback_rf2 \
  --seed 0 \
  --total_steps 200000 \
  --start_steps 20000 \
  --update_after 10000 \
  --batch_size 256 \
  --sample_k 256 \
  --lambda_p 0.1 \
  --init_alpha 0.1 \
  --policy_noise_scale 0.3 \
  --start_y_range 0.4 \
  --weight_mix 0.05 \
  --log_dir logs/obstacle/safe_pullback_rf2_seed0_sym_tangent_fix

python eval/eval_safe_obstacle_navigation.py \
  --checkpoint logs/obstacle/safe_pullback_rf2_seed0_sym_tangent_fix/checkpoint.pkl \
  --algo safe_pullback_rf2 \
  --eval_episodes 200 \
  --save_dir results/obstacle/safe_pullback_rf2_sym_tangent_fix

python plot/plot_obstacle_rollouts.py \
  --methods safe_pullback_rf2_sym_tangent_fix \
  --base_dir results/obstacle \
  --out_dir figures
