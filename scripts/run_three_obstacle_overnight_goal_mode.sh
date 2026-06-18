#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.60}
export TF_FORCE_GPU_ALLOW_GROWTH=${TF_FORCE_GPU_ALLOW_GROWTH:-true}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mplconfig}
export XLA_FLAGS=${XLA_FLAGS:---xla_gpu_enable_command_buffer=}

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

CONDA_BIN=${CONDA_BIN:-/home/carla/anaconda3/bin/conda}
PYTHON_BIN=${PYTHON_BIN:-python}

OVERNIGHT_DIR="outputs/double_integrator_pullback/trials/overnight_goal_mode"
STATUS="$OVERNIGHT_DIR/overnight_status.md"
TRIALS_CSV="$OVERNIGHT_DIR/overnight_trials.csv"
PLAN_JSON="$OVERNIGHT_DIR/overnight_plan.json"
FINAL_REPORT="outputs/double_integrator_pullback/final_experiment_report.md"
SOURCE_CKPT="outputs/double_integrator_pullback/trials/trial_three_obstacle_013/curvature/checkpoint_latest.pkl"

mkdir -p "$OVERNIGHT_DIR" "$MPLCONFIGDIR"

log_status() {
  local msg="$1"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$msg" | tee -a "$STATUS"
}

run_py() {
  "$CONDA_BIN" run --no-capture-output -n value-flows "$PYTHON_BIN" "$@"
}

ensure_no_other_jobs() {
  local matches
  matches=$(pgrep -af "train_double_integrator_pullback.py|eval_double_integrator_pullback.py|eval_double_integrator_distribution_geometry.py|eval_double_integrator_domain_shift.py" || true)
  if [ -n "$matches" ]; then
    log_status "Refusing to start because another train/eval process is present:"
    printf '%s\n' "$matches" | tee -a "$STATUS"
    exit 1
  fi
}

COMMON_ENV_ARGS=(
  --map_id three_circles
  --route_variant exit_pull_v2
  --obs_mode all_obstacles
  --reward_mode multi_route_progress
  --goal_radius 0.25
  --start_y_range 1.0
  --dt 0.08
  --a_max 4.0
  --v_max 2.5
  --damping 0.98
  --episode_len 300
)

COMMON_REWARD_ARGS=(
  --reward_progress_coef 26.0
  --reward_success_bonus 320.0
  --reward_collision_penalty 120.0
  --reward_near_obs_coef 4.5
  --reward_safety_buffer 0.18
  --reward_action_coef 0.015
  --reward_speed_coef 0.005
  --reward_time_coef 0.02
  --reward_route_softmin_beta 4.0
  --reward_route_start_bias_scale 0.42
)

write_plan() {
  cat > "$PLAN_JSON" <<'JSON'
{
  "fixed": {
    "map_id": "three_circles",
    "route_variant": "exit_pull_v2",
    "obs_mode": "all_obstacles",
    "reward_mode": "multi_route_progress",
    "goal_radius": 0.25,
    "source_checkpoint": "outputs/double_integrator_pullback/trials/trial_three_obstacle_013/curvature/checkpoint_latest.pkl"
  },
  "queue": [
    {"trial": "trial_three_obstacle_017_mix075", "variable": "reward_goal_progress_mix", "value": 0.75},
    {"trial": "trial_three_obstacle_018_mix090", "variable": "reward_goal_progress_mix", "value": 0.90, "condition": "safe_but_success_insufficient"},
    {"trial": "trial_three_obstacle_019_safety", "variable": "reward_near_obs_coef", "value": 5.625, "condition": "unsafe_after_mix"},
    {"trial": "trial_three_obstacle_020_terminal_pull", "variable": "terminal_goal_bonus_coef", "value": 2.0, "terminal_goal_bonus_radius": 1.20, "condition": "safe_but_upper_success_low"},
    {"trial": "trial_three_obstacle_021_longer_best", "variable": "total_steps", "value": 100000, "condition": "best_safe_candidate_success_0.60_to_0.75"}
  ]
}
JSON
}

init_logs() {
  : > "$STATUS"
  if [ ! -f "$TRIALS_CSV" ]; then
    echo "trial,total_steps,mix,near_obs_coef,terminal_radius,terminal_coef,success_rate,collision_rate,violation_rate,h_min_mean,J_eval_mean,return_mean,upper_start_success_rate,middle_start_success_rate,lower_start_success_rate,upper_start_upper_route_fraction,lower_start_lower_route_fraction,route_upper_fraction,route_lower_fraction,final_distance_mean,final_distance_q50,final_distance_q75,final_distance_q90,checkpoint,status,next_action" > "$TRIALS_CSV"
  fi
}

write_trial_files() {
  local trial_dir="$1"
  local total_steps="$2"
  local mix="$3"
  local near_obs="$4"
  local terminal_radius="$5"
  local terminal_coef="$6"
  local resume="$7"
  mkdir -p "$trial_dir"
  cat > "$trial_dir/config.json" <<JSON
{
  "map_id": "three_circles",
  "route_variant": "exit_pull_v2",
  "obs_mode": "all_obstacles",
  "reward_mode": "multi_route_progress",
  "goal_radius": 0.25,
  "reward_goal_progress_mix": $mix,
  "reward_near_obs_coef": $near_obs,
  "terminal_goal_bonus_radius": $terminal_radius,
  "terminal_goal_bonus_coef": $terminal_coef,
  "source_checkpoint": "$resume",
  "total_steps": $total_steps,
  "seed": 0,
  "notes": "Overnight goal-running mode. Single-variable queue; no route variant or goal-radius change."
}
JSON
  cat > "$trial_dir/notes.txt" <<TXT
Why this trial was launched:
- Overnight goal-running mode after goal-radius sweep showed terminal radius is not the blocker.

Fixed settings:
- map_id=three_circles
- route_variant=exit_pull_v2
- obs_mode=all_obstacles
- reward_mode=multi_route_progress
- goal_radius=0.25

Changed variables for this trial:
- reward_goal_progress_mix=$mix
- reward_near_obs_coef=$near_obs
- terminal_goal_bonus_radius=$terminal_radius
- terminal_goal_bonus_coef=$terminal_coef

Resume checkpoint:
- $resume
TXT
}

train_curvature() {
  local trial_dir="$1"
  local total_steps="$2"
  local mix="$3"
  local near_obs="$4"
  local terminal_radius="$5"
  local terminal_coef="$6"
  local resume="$7"
  local outdir="$trial_dir/curvature"
  mkdir -p "$outdir"
  local cmd=(
    scripts/train_double_integrator_pullback.py
    --algo curvature_flow
    --seed 0
    --outdir "$outdir"
    --total_steps "$total_steps"
    --save_interval 5000
    --eval_interval 0
    --resume_checkpoint "$resume"
    "${COMMON_ENV_ARGS[@]}"
    "${COMMON_REWARD_ARGS[@]}"
    --reward_near_obs_coef "$near_obs"
    --reward_goal_progress_mix "$mix"
    --terminal_goal_bonus_radius "$terminal_radius"
    --terminal_goal_bonus_coef "$terminal_coef"
  )
  printf '%q ' "${cmd[@]}" > "$trial_dir/train_command.txt"
  printf '\n' >> "$trial_dir/train_command.txt"
  log_status "Training curvature: $trial_dir total_steps=$total_steps mix=$mix near_obs=$near_obs terminal=($terminal_radius,$terminal_coef)"
  run_py "${cmd[@]}" 2>&1 | tee "$outdir/train.log"
}

eval_curvature() {
  local trial_dir="$1"
  local mix="$2"
  local near_obs="$3"
  local terminal_radius="$4"
  local terminal_coef="$5"
  local ckpt="$trial_dir/curvature/checkpoint_latest.pkl"
  log_status "Evaluating curvature: $trial_dir"
  run_py eval/eval_double_integrator_pullback.py \
    --checkpoint "$ckpt" \
    --algo curvature_flow \
    --outdir "$trial_dir/eval_curvature_main" \
    --episodes 200 \
    --delta 0.0 \
    --save_rollouts \
    "${COMMON_ENV_ARGS[@]}" \
    "${COMMON_REWARD_ARGS[@]}" \
    --reward_near_obs_coef "$near_obs" \
    --reward_goal_progress_mix "$mix" \
    --terminal_goal_bonus_radius "$terminal_radius" \
    --terminal_goal_bonus_coef "$terminal_coef" \
    | tee "$trial_dir/eval_curvature_main/eval.log"
}

record_trial() {
  local trial_dir="$1"
  local total_steps="$2"
  local mix="$3"
  local near_obs="$4"
  local terminal_radius="$5"
  local terminal_coef="$6"
  local status="$7"
  local next_action="$8"
  run_py - <<PY
import csv, json, pathlib
trial_dir = pathlib.Path("$trial_dir")
s = json.loads((trial_dir / "eval_curvature_main" / "eval_summary.json").read_text())
row = {
    "trial": trial_dir.name,
    "total_steps": "$total_steps",
    "mix": "$mix",
    "near_obs_coef": "$near_obs",
    "terminal_radius": "$terminal_radius",
    "terminal_coef": "$terminal_coef",
    "success_rate": s.get("success_rate"),
    "collision_rate": s.get("collision_rate"),
    "violation_rate": s.get("violation_rate"),
    "h_min_mean": s.get("h_min_mean"),
    "J_eval_mean": s.get("J_eval_mean"),
    "return_mean": s.get("return_mean"),
    "upper_start_success_rate": s.get("upper_start_success_rate"),
    "middle_start_success_rate": s.get("middle_start_success_rate"),
    "lower_start_success_rate": s.get("lower_start_success_rate"),
    "upper_start_upper_route_fraction": s.get("upper_start_upper_route_fraction"),
    "lower_start_lower_route_fraction": s.get("lower_start_lower_route_fraction"),
    "route_upper_fraction": s.get("route_upper_fraction"),
    "route_lower_fraction": s.get("route_lower_fraction"),
    "final_distance_mean": s.get("final_distance_mean"),
    "final_distance_q50": s.get("final_distance_q50"),
    "final_distance_q75": s.get("final_distance_q75"),
    "final_distance_q90": s.get("final_distance_q90"),
    "checkpoint": str(trial_dir / "curvature" / "checkpoint_latest.pkl"),
    "status": "$status",
    "next_action": "$next_action",
}
csv_path = pathlib.Path("$TRIALS_CSV")
with csv_path.open("a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    writer.writerow(row)
PY
}

write_decision_env() {
  local trial_dir="$1"
  local decision_env="$OVERNIGHT_DIR/last_decision.env"
  run_py - <<PY
import json, pathlib
s = json.loads((pathlib.Path("$trial_dir") / "eval_curvature_main" / "eval_summary.json").read_text())
success = float(s.get("success_rate", 0.0))
collision = float(s.get("collision_rate", 1.0))
violation = float(s.get("violation_rate", 1.0))
h_min = float(s.get("h_min_mean", -1.0))
upper_route = float(s.get("upper_start_upper_route_fraction", 0.0))
lower_route = float(s.get("lower_start_lower_route_fraction", 0.0))
route_upper = float(s.get("route_upper_fraction", 0.0))
route_lower = float(s.get("route_lower_fraction", 0.0))
upper_success = float(s.get("upper_start_success_rate", 0.0))
safe = collision <= 0.15 and violation <= 0.03 and h_min >= 0.10
route_ok = upper_route >= 0.65 and lower_route >= 0.65 and route_upper >= 0.25 and route_lower >= 0.25
passed = success >= 0.75 and safe and route_ok
improved = success > 0.43 or upper_success > 0.0
mid_success = safe and route_ok and 0.60 <= success < 0.75
with open("$decision_env", "w") as f:
    f.write(f"SUCCESS={success}\\n")
    f.write(f"COLLISION={collision}\\n")
    f.write(f"VIOLATION={violation}\\n")
    f.write(f"H_MIN={h_min}\\n")
    f.write(f"UPPER_SUCCESS={upper_success}\\n")
    f.write(f"SAFE={int(safe)}\\n")
    f.write(f"ROUTE_OK={int(route_ok)}\\n")
    f.write(f"PASSED={int(passed)}\\n")
    f.write(f"IMPROVED={int(improved)}\\n")
    f.write(f"MID_SUCCESS={int(mid_success)}\\n")
PY
}

run_trial() {
  local trial_name="$1"
  local total_steps="$2"
  local mix="$3"
  local near_obs="$4"
  local terminal_radius="$5"
  local terminal_coef="$6"
  local resume="$7"
  local trial_dir="outputs/double_integrator_pullback/trials/$trial_name"
  write_trial_files "$trial_dir" "$total_steps" "$mix" "$near_obs" "$terminal_radius" "$terminal_coef" "$resume"
  train_curvature "$trial_dir" "$total_steps" "$mix" "$near_obs" "$terminal_radius" "$terminal_coef" "$resume"
  eval_curvature "$trial_dir" "$mix" "$near_obs" "$terminal_radius" "$terminal_coef"
  write_decision_env "$trial_dir"
  # shellcheck disable=SC1090
  source "$OVERNIGHT_DIR/last_decision.env"
  if [ "$PASSED" = "1" ]; then
    record_trial "$trial_dir" "$total_steps" "$mix" "$near_obs" "$terminal_radius" "$terminal_coef" "pass" "train_vanilla_and_final_eval"
    echo "$trial_dir" > "$OVERNIGHT_DIR/FOUND_CANDIDATE.txt"
    echo "$trial_dir/curvature/checkpoint_latest.pkl" >> "$OVERNIGHT_DIR/FOUND_CANDIDATE.txt"
    return 0
  fi
  record_trial "$trial_dir" "$total_steps" "$mix" "$near_obs" "$terminal_radius" "$terminal_coef" "fail" "continue_queue"
  return 1
}

train_vanilla_and_final_eval() {
  local trial_dir="$1"
  local total_steps="$2"
  local mix="$3"
  local near_obs="$4"
  local terminal_radius="$5"
  local terminal_coef="$6"
  log_status "Training vanilla fair baseline for $trial_dir"
  run_py scripts/train_double_integrator_pullback.py \
    --algo vanilla_flow \
    --seed 0 \
    --outdir "$trial_dir/vanilla" \
    --total_steps "$total_steps" \
    --save_interval 5000 \
    --eval_interval 0 \
    "${COMMON_ENV_ARGS[@]}" \
    "${COMMON_REWARD_ARGS[@]}" \
    --reward_near_obs_coef "$near_obs" \
    --reward_goal_progress_mix "$mix" \
    --terminal_goal_bonus_radius "$terminal_radius" \
    --terminal_goal_bonus_coef "$terminal_coef" \
    2>&1 | tee "$trial_dir/vanilla/train.log"

  log_status "Evaluating vanilla baseline for $trial_dir"
  run_py eval/eval_double_integrator_pullback.py \
    --checkpoint "$trial_dir/vanilla/checkpoint_latest.pkl" \
    --algo vanilla_flow \
    --outdir "$trial_dir/eval_vanilla_main" \
    --episodes 200 \
    --delta 0.0 \
    --save_rollouts \
    "${COMMON_ENV_ARGS[@]}" \
    "${COMMON_REWARD_ARGS[@]}" \
    --reward_near_obs_coef "$near_obs" \
    --reward_goal_progress_mix "$mix" \
    --terminal_goal_bonus_radius "$terminal_radius" \
    --terminal_goal_bonus_coef "$terminal_coef"

  log_status "Running distribution geometry for $trial_dir"
  run_py eval/eval_double_integrator_distribution_geometry.py \
    --vanilla_checkpoint "$trial_dir/vanilla/checkpoint_latest.pkl" \
    --curvature_checkpoint "$trial_dir/curvature/checkpoint_latest.pkl" \
    --outdir "$trial_dir/distribution_main" \
    --num_states_per_obstacle 4 \
    --num_action_samples 4096 \
    --seed 0 \
    "${COMMON_ENV_ARGS[@]}" \
    "${COMMON_REWARD_ARGS[@]}" \
    --reward_near_obs_coef "$near_obs" \
    --reward_goal_progress_mix "$mix" \
    --terminal_goal_bonus_radius "$terminal_radius" \
    --terminal_goal_bonus_coef "$terminal_coef"

  log_status "Running domain shift for $trial_dir"
  run_py eval/eval_double_integrator_domain_shift.py \
    --vanilla_checkpoint "$trial_dir/vanilla/checkpoint_latest.pkl" \
    --curvature_checkpoint "$trial_dir/curvature/checkpoint_latest.pkl" \
    --outdir "$trial_dir/domain_shift_main" \
    --episodes 200 \
    --delta_grid 0.0 0.1 0.2 0.3 0.4 \
    --seed 0 \
    --save_rollouts \
    "${COMMON_ENV_ARGS[@]}" \
    "${COMMON_REWARD_ARGS[@]}" \
    --reward_near_obs_coef "$near_obs" \
    --reward_goal_progress_mix "$mix" \
    --terminal_goal_bonus_radius "$terminal_radius" \
    --terminal_goal_bonus_coef "$terminal_coef"

  log_status "Generating paper figures for $trial_dir"
  run_py analysis/plot_three_obstacle_pullback_results.py \
    --trial_dir "$trial_dir" \
    --outdir "$trial_dir/paper_figures"
}

append_report() {
  local outcome="$1"
  local best_trial="${2:-}"
  run_py - <<PY
from pathlib import Path
from datetime import datetime
report = Path("$FINAL_REPORT")
trials_csv = Path("$TRIALS_CSV")
found = Path("$OVERNIGHT_DIR/FOUND_CANDIDATE.txt")
failure = Path("$OVERNIGHT_DIR/overnight_failure_report.md")
section = []
section.append("\\n## Overnight goal-running mode\\n")
section.append(f"- Updated: {datetime.now().isoformat(timespec='seconds')}\\n")
section.append(f"- Outcome: $outcome\\n")
section.append(f"- Trial queue CSV: `{trials_csv}`\\n")
if found.exists():
    lines = found.read_text().splitlines()
    section.append(f"- Selected trial: `{lines[0] if lines else ''}`\\n")
    section.append(f"- Final curvature checkpoint: `{lines[1] if len(lines) > 1 else ''}`\\n")
else:
    section.append("- Selected trial: none\\n")
if "$best_trial":
    section.append(f"- Best curvature trial: `$best_trial`\\n")
if failure.exists():
    section.append(f"- Failure report: `{failure}`\\n")
if trials_csv.exists():
    section.append("\\n### Overnight trial table\\n\\n")
    section.append("```csv\\n")
    section.append(trials_csv.read_text())
    section.append("\\n```\\n")
report.write_text(report.read_text() + "".join(section))
PY
}

write_failure_report() {
  local best_trial="$1"
  cat > "$OVERNIGHT_DIR/overnight_failure_report.md" <<TXT
# Overnight goal-running failure report

No curvature trial satisfied all acceptance criteria.

Best trial:
- $best_trial

Trial metrics:

\`\`\`csv
$(cat "$TRIALS_CSV")
\`\`\`

Reason:
- The queue exhausted without satisfying success, safety, and route-conditioned criteria simultaneously.

Next single-variable suggestion:
- Inspect the best trial endpoints and choose one variable only. If all trials remain safe but upper-start success is low, increase terminal_goal_bonus_coef from 2.0 to 3.0 while keeping terminal_goal_bonus_radius=1.20.
TXT
}

select_best_trial() {
  run_py - <<PY
import csv
from pathlib import Path
rows = list(csv.DictReader(Path("$TRIALS_CSV").open()))
rows = [r for r in rows if r.get("trial") != "trial"]
def f(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except ValueError:
        return default
if not rows:
    print("")
else:
    rows.sort(key=lambda r: (f(r, "success_rate"), -f(r, "collision_rate"), -f(r, "violation_rate"), f(r, "h_min_mean")), reverse=True)
    print("outputs/double_integrator_pullback/trials/" + rows[0]["trial"])
PY
}

main() {
  ensure_no_other_jobs
  write_plan
  init_logs
  log_status "Overnight goal mode started"
  log_status "Fixed route_variant=exit_pull_v2, goal_radius=0.25. No route or goal-radius changes will be made."

  local selected=""
  local selected_steps=""
  local selected_mix=""
  local selected_near=""
  local selected_terminal_radius=""
  local selected_terminal_coef=""

  if run_trial "trial_three_obstacle_017_mix075" 50000 0.75 4.5 0.0 0.0 "$SOURCE_CKPT"; then
    selected="outputs/double_integrator_pullback/trials/trial_three_obstacle_017_mix075"
    selected_steps=50000; selected_mix=0.75; selected_near=4.5; selected_terminal_radius=0.0; selected_terminal_coef=0.0
  else
    source "$OVERNIGHT_DIR/last_decision.env"
    if [ "$SAFE" != "1" ]; then
      if run_trial "trial_three_obstacle_019_safety" 50000 0.75 5.625 0.0 0.0 "$SOURCE_CKPT"; then
        selected="outputs/double_integrator_pullback/trials/trial_three_obstacle_019_safety"
        selected_steps=50000; selected_mix=0.75; selected_near=5.625; selected_terminal_radius=0.0; selected_terminal_coef=0.0
      fi
    elif [ "$IMPROVED" = "1" ]; then
      if run_trial "trial_three_obstacle_018_mix090" 50000 0.90 4.5 0.0 0.0 "$SOURCE_CKPT"; then
        selected="outputs/double_integrator_pullback/trials/trial_three_obstacle_018_mix090"
        selected_steps=50000; selected_mix=0.90; selected_near=4.5; selected_terminal_radius=0.0; selected_terminal_coef=0.0
      else
        source "$OVERNIGHT_DIR/last_decision.env"
        if [ "$MID_SUCCESS" = "1" ]; then
          local trial_dir="outputs/double_integrator_pullback/trials/trial_three_obstacle_018_mix090"
          if run_trial "trial_three_obstacle_018_mix090" 100000 0.90 4.5 0.0 0.0 "$trial_dir/curvature/checkpoint_latest.pkl"; then
            selected="$trial_dir"
            selected_steps=100000; selected_mix=0.90; selected_near=4.5; selected_terminal_radius=0.0; selected_terminal_coef=0.0
          fi
        elif [ "$SAFE" = "1" ]; then
          if run_trial "trial_three_obstacle_020_terminal_pull" 50000 0.75 4.5 1.20 2.0 "$SOURCE_CKPT"; then
            selected="outputs/double_integrator_pullback/trials/trial_three_obstacle_020_terminal_pull"
            selected_steps=50000; selected_mix=0.75; selected_near=4.5; selected_terminal_radius=1.20; selected_terminal_coef=2.0
          fi
        else
          if run_trial "trial_three_obstacle_019_safety" 50000 0.75 5.625 0.0 0.0 "$SOURCE_CKPT"; then
            selected="outputs/double_integrator_pullback/trials/trial_three_obstacle_019_safety"
            selected_steps=50000; selected_mix=0.75; selected_near=5.625; selected_terminal_radius=0.0; selected_terminal_coef=0.0
          fi
        fi
      fi
    else
      if run_trial "trial_three_obstacle_020_terminal_pull" 50000 0.75 4.5 1.20 2.0 "$SOURCE_CKPT"; then
        selected="outputs/double_integrator_pullback/trials/trial_three_obstacle_020_terminal_pull"
        selected_steps=50000; selected_mix=0.75; selected_near=4.5; selected_terminal_radius=1.20; selected_terminal_coef=2.0
      fi
    fi
  fi

  if [ -z "$selected" ]; then
    best=$(select_best_trial)
    write_failure_report "$best"
    append_report "no_pass_candidate" "$best"
    log_status "No pass candidate found. Best trial: $best"
    exit 0
  fi

  log_status "Found candidate: $selected"
  train_vanilla_and_final_eval "$selected" "$selected_steps" "$selected_mix" "$selected_near" "$selected_terminal_radius" "$selected_terminal_coef"
  append_report "candidate_found" "$selected"
  log_status "Overnight goal mode completed with candidate: $selected"
}

main "$@"
