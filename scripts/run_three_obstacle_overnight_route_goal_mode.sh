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

ROOT="outputs/double_integrator_pullback/trials/overnight_route_goal_mode"
STATUS="$ROOT/overnight_route_status.md"
TRIALS_CSV="$ROOT/overnight_route_trials.csv"
PLAN_JSON="$ROOT/overnight_route_plan.json"
SUMMARY="$ROOT/summary.md"
FINAL_REPORT="outputs/double_integrator_pullback/final_experiment_report.md"
SOURCE_CKPT="outputs/double_integrator_pullback/trials/trial_three_obstacle_013/curvature/checkpoint_latest.pkl"

mkdir -p "$ROOT" "$MPLCONFIGDIR"

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

init_files() {
  : > "$STATUS"
  cat > "$TRIALS_CSV" <<'CSV'
trial,reward_goal_progress_mix,terminal_goal_bonus_coef,total_steps,success_rate,collision_rate,violation_rate,h_min_mean,upper_start_success_rate,middle_start_success_rate,lower_start_success_rate,upper_start_upper_route_fraction,lower_start_lower_route_fraction,route_upper_fraction,route_lower_fraction,conditioned_success_balance,status,next_action,checkpoint
CSV
  cat > "$PLAN_JSON" <<'JSON'
{
  "fixed": {
    "source_checkpoint": "outputs/double_integrator_pullback/trials/trial_three_obstacle_013/curvature/checkpoint_latest.pkl",
    "map_id": "three_circles",
    "route_variant": "exit_pull_v2",
    "obs_mode": "all_obstacles",
    "reward_mode": "multi_route_progress",
    "goal_radius": 0.25
  },
  "global_mix_sweep": [
    {"trial": "trial_three_obstacle_022_mix060", "reward_goal_progress_mix": 0.60},
    {"trial": "trial_three_obstacle_023_mix065", "reward_goal_progress_mix": 0.65},
    {"trial": "trial_three_obstacle_024_mix070", "reward_goal_progress_mix": 0.70, "condition": "mix065 route preserving but insufficient"}
  ],
  "terminal_only_pull": [
    {"trial": "trial_three_obstacle_025_terminal_coef15", "reward_goal_progress_mix": 0.50, "terminal_goal_bonus_radius": 1.20, "terminal_goal_bonus_coef": 1.5},
    {"trial": "trial_three_obstacle_026_terminal_coef20", "reward_goal_progress_mix": 0.50, "terminal_goal_bonus_radius": 1.20, "terminal_goal_bonus_coef": 2.0},
    {"trial": "trial_three_obstacle_027_terminal_coef25", "reward_goal_progress_mix": 0.50, "terminal_goal_bonus_radius": 1.20, "terminal_goal_bonus_coef": 2.5}
  ]
}
JSON
}

write_trial_files() {
  local trial_dir="$1"
  local total_steps="$2"
  local mix="$3"
  local terminal_radius="$4"
  local terminal_coef="$5"
  local resume="$6"
  mkdir -p "$trial_dir"
  cat > "$trial_dir/config.json" <<JSON
{
  "map_id": "three_circles",
  "route_variant": "exit_pull_v2",
  "obs_mode": "all_obstacles",
  "reward_mode": "multi_route_progress",
  "goal_radius": 0.25,
  "reward_goal_progress_mix": $mix,
  "terminal_goal_bonus_radius": $terminal_radius,
  "terminal_goal_bonus_coef": $terminal_coef,
  "source_checkpoint": "$resume",
  "total_steps": $total_steps,
  "seed": 0
}
JSON
  cat > "$trial_dir/notes.txt" <<TXT
Why this trial was launched:
- Second overnight route-preserving goal mode.
- trial_017 improved success but collapsed all routes to upper.
- This trial warm-starts from trial_013 to preserve route geometry.

Fixed settings:
- map_id=three_circles
- route_variant=exit_pull_v2
- obs_mode=all_obstacles
- reward_mode=multi_route_progress
- goal_radius=0.25
- source_checkpoint=$resume

Changed variable:
- reward_goal_progress_mix=$mix
- terminal_goal_bonus_radius=$terminal_radius
- terminal_goal_bonus_coef=$terminal_coef
TXT
}

train_curvature() {
  local trial_dir="$1"
  local total_steps="$2"
  local mix="$3"
  local terminal_radius="$4"
  local terminal_coef="$5"
  local resume="$6"
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
    --reward_goal_progress_mix "$mix"
    --terminal_goal_bonus_radius "$terminal_radius"
    --terminal_goal_bonus_coef "$terminal_coef"
  )
  printf '%q ' "${cmd[@]}" > "$trial_dir/train_command.txt"
  printf '\n' >> "$trial_dir/train_command.txt"
  log_status "Training curvature: $trial_dir steps=$total_steps mix=$mix terminal=($terminal_radius,$terminal_coef)"
  run_py "${cmd[@]}" 2>&1 | tee "$outdir/train.log"
}

eval_curvature() {
  local trial_dir="$1"
  local mix="$2"
  local terminal_radius="$3"
  local terminal_coef="$4"
  mkdir -p "$trial_dir/eval_curvature_main"
  log_status "Evaluating curvature: $trial_dir"
  run_py eval/eval_double_integrator_pullback.py \
    --checkpoint "$trial_dir/curvature/checkpoint_latest.pkl" \
    --algo curvature_flow \
    --outdir "$trial_dir/eval_curvature_main" \
    --episodes 200 \
    --delta 0.0 \
    --save_rollouts \
    "${COMMON_ENV_ARGS[@]}" \
    "${COMMON_REWARD_ARGS[@]}" \
    --reward_goal_progress_mix "$mix" \
    --terminal_goal_bonus_radius "$terminal_radius" \
    --terminal_goal_bonus_coef "$terminal_coef" \
    2>&1 | tee "$trial_dir/eval_curvature_main/eval.log"
}

write_decision() {
  local trial_dir="$1"
  local mix="$2"
  local terminal_coef="$3"
  local total_steps="$4"
  local decision_file="$ROOT/last_decision.env"
  run_py - <<PY
import csv, json
from pathlib import Path
trial_dir = Path("$trial_dir")
s = json.loads((trial_dir / "eval_curvature_main" / "eval_summary.json").read_text())
success = float(s.get("success_rate", 0.0))
collision = float(s.get("collision_rate", 1.0))
violation = float(s.get("violation_rate", 1.0))
h_min = float(s.get("h_min_mean", -1.0))
route_upper = float(s.get("route_upper_fraction", 0.0))
route_lower = float(s.get("route_lower_fraction", 0.0))
upper_route = float(s.get("upper_start_upper_route_fraction", 0.0))
lower_route = float(s.get("lower_start_lower_route_fraction", 0.0))
upper_success = float(s.get("upper_start_success_rate", 0.0))
middle_success = float(s.get("middle_start_success_rate", 0.0))
lower_success = float(s.get("lower_start_success_rate", 0.0))
balance = min(upper_success, lower_success)
safety_ok = collision <= 0.15 and violation <= 0.03 and h_min >= 0.10
route_ok = route_upper >= 0.25 and route_lower >= 0.25
conditioned_route_ok = upper_route >= 0.65 and lower_route >= 0.65
passed = success >= 0.75 and safety_ok and route_ok and conditioned_route_ok
status = "pass" if passed else "fail"
if passed:
    next_action = "train_vanilla_and_final_eval"
elif not route_ok or not conditioned_route_ok:
    next_action = "route_collapse"
elif safety_ok and 0.65 <= success < 0.75:
    next_action = "eligible_long_training"
else:
    next_action = "continue_queue"
row = {
    "trial": trial_dir.name,
    "reward_goal_progress_mix": "$mix",
    "terminal_goal_bonus_coef": "$terminal_coef",
    "total_steps": "$total_steps",
    "success_rate": success,
    "collision_rate": collision,
    "violation_rate": violation,
    "h_min_mean": h_min,
    "upper_start_success_rate": upper_success,
    "middle_start_success_rate": middle_success,
    "lower_start_success_rate": lower_success,
    "upper_start_upper_route_fraction": upper_route,
    "lower_start_lower_route_fraction": lower_route,
    "route_upper_fraction": route_upper,
    "route_lower_fraction": route_lower,
    "conditioned_success_balance": balance,
    "status": status,
    "next_action": next_action,
    "checkpoint": str(trial_dir / "curvature" / "checkpoint_latest.pkl"),
}
csv_path = Path("$TRIALS_CSV")
with csv_path.open("a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    writer.writerow(row)
with open("$decision_file", "w") as f:
    f.write(f"PASSED={int(passed)}\\n")
    f.write(f"SAFETY_OK={int(safety_ok)}\\n")
    f.write(f"ROUTE_OK={int(route_ok)}\\n")
    f.write(f"CONDITIONED_ROUTE_OK={int(conditioned_route_ok)}\\n")
    f.write(f"SUCCESS={success}\\n")
    f.write(f"BALANCE={balance}\\n")
    f.write(f"NEXT_ACTION={next_action}\\n")
PY
}

run_trial() {
  local trial_name="$1"
  local total_steps="$2"
  local mix="$3"
  local terminal_radius="$4"
  local terminal_coef="$5"
  local resume="$6"
  local trial_dir="outputs/double_integrator_pullback/trials/$trial_name"
  write_trial_files "$trial_dir" "$total_steps" "$mix" "$terminal_radius" "$terminal_coef" "$resume"
  train_curvature "$trial_dir" "$total_steps" "$mix" "$terminal_radius" "$terminal_coef" "$resume"
  eval_curvature "$trial_dir" "$mix" "$terminal_radius" "$terminal_coef"
  write_decision "$trial_dir" "$mix" "$terminal_coef" "$total_steps"
  source "$ROOT/last_decision.env"
  if [ "$PASSED" = "1" ]; then
    printf '%s\n%s\n%s\n%s\n%s\n' "$trial_dir" "$total_steps" "$mix" "$terminal_radius" "$terminal_coef" > "$ROOT/FOUND_CANDIDATE.txt"
    return 0
  fi
  return 1
}

train_vanilla_and_final_eval() {
  local trial_dir="$1"
  local total_steps="$2"
  local mix="$3"
  local terminal_radius="$4"
  local terminal_coef="$5"
  log_status "Training vanilla fair baseline: $trial_dir"
  run_py scripts/train_double_integrator_pullback.py \
    --algo vanilla_flow \
    --seed 0 \
    --outdir "$trial_dir/vanilla" \
    --total_steps "$total_steps" \
    --save_interval 5000 \
    --eval_interval 0 \
    "${COMMON_ENV_ARGS[@]}" \
    "${COMMON_REWARD_ARGS[@]}" \
    --reward_goal_progress_mix "$mix" \
    --terminal_goal_bonus_radius "$terminal_radius" \
    --terminal_goal_bonus_coef "$terminal_coef" \
    2>&1 | tee "$trial_dir/vanilla/train.log"

  log_status "Evaluating vanilla baseline: $trial_dir"
  run_py eval/eval_double_integrator_pullback.py \
    --checkpoint "$trial_dir/vanilla/checkpoint_latest.pkl" \
    --algo vanilla_flow \
    --outdir "$trial_dir/eval_vanilla_main" \
    --episodes 200 \
    --delta 0.0 \
    --save_rollouts \
    "${COMMON_ENV_ARGS[@]}" \
    "${COMMON_REWARD_ARGS[@]}" \
    --reward_goal_progress_mix "$mix" \
    --terminal_goal_bonus_radius "$terminal_radius" \
    --terminal_goal_bonus_coef "$terminal_coef"

  log_status "Running distribution geometry: $trial_dir"
  run_py eval/eval_double_integrator_distribution_geometry.py \
    --vanilla_checkpoint "$trial_dir/vanilla/checkpoint_latest.pkl" \
    --curvature_checkpoint "$trial_dir/curvature/checkpoint_latest.pkl" \
    --outdir "$trial_dir/distribution_main" \
    --num_states_per_obstacle 4 \
    --num_action_samples 4096 \
    --seed 0 \
    "${COMMON_ENV_ARGS[@]}" \
    "${COMMON_REWARD_ARGS[@]}" \
    --reward_goal_progress_mix "$mix" \
    --terminal_goal_bonus_radius "$terminal_radius" \
    --terminal_goal_bonus_coef "$terminal_coef"

  log_status "Running domain shift: $trial_dir"
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
    --reward_goal_progress_mix "$mix" \
    --terminal_goal_bonus_radius "$terminal_radius" \
    --terminal_goal_bonus_coef "$terminal_coef"

  log_status "Generating paper figures: $trial_dir"
  run_py analysis/plot_three_obstacle_pullback_results.py \
    --trial_dir "$trial_dir" \
    --outdir "$trial_dir/paper_figures"
}

best_diagnostic_trial() {
  run_py - <<PY
import csv
from pathlib import Path
rows = list(csv.DictReader(Path("$TRIALS_CSV").open()))
def f(r, k):
    try:
        return float(r.get(k) or 0.0)
    except ValueError:
        return 0.0
def score(r):
    route = int(f(r, "route_upper_fraction") >= 0.25 and f(r, "route_lower_fraction") >= 0.25)
    conditioned = int(f(r, "upper_start_upper_route_fraction") >= 0.65 and f(r, "lower_start_lower_route_fraction") >= 0.65)
    return (route, conditioned, f(r, "conditioned_success_balance"), f(r, "success_rate"), f(r, "h_min_mean"))
if rows:
    rows.sort(key=score, reverse=True)
    print("outputs/double_integrator_pullback/trials/" + rows[0]["trial"])
PY
}

write_summary() {
  local outcome="$1"
  local selected="${2:-none}"
  local best="${3:-none}"
  cat > "$SUMMARY" <<TXT
Outcome: $outcome

Why trial_017 failed:
- It improved success but collapsed route diversity to the upper route.
- New trials warm-started from trial_013 to preserve route geometry.

Selected final trial:
- $selected

Best diagnostic trial:
- $best

Trial table:

\`\`\`csv
$(cat "$TRIALS_CSV")
\`\`\`

Acceptance criteria:
- Candidate is paper-ready only if success, safety, route diversity, and conditioned route criteria all pass.
TXT
  python scripts/append_experiment_report.py \
    --report "$FINAL_REPORT" \
    --title "Second overnight route-preserving goal mode" \
    --body-file "$SUMMARY"
}

write_failure_report() {
  local best="$1"
  cat > "$ROOT/overnight_route_failure_report.md" <<TXT
# Second overnight route-preserving failure report

No trial satisfied all paper-ready curvature criteria.

Best diagnostic trial:
- $best

Metrics:

\`\`\`csv
$(cat "$TRIALS_CSV")
\`\`\`

Next single-variable suggestion:
- Inspect endpoints from the best diagnostic trial before changing another variable.
- If route diversity is preserved but upper success is still low, increase terminal_goal_bonus_coef by one step while keeping route_variant and goal_radius fixed.
TXT
}

main() {
  ensure_no_other_jobs
  init_files
  log_status "Second overnight route-preserving goal mode started"
  log_status "All trials warm-start from trial_013, not trial_017."

  selected=""
  selected_steps=""
  selected_mix=""
  selected_terminal_radius=""
  selected_terminal_coef=""

  if run_trial "trial_three_obstacle_022_mix060" 50000 0.60 0.0 0.0 "$SOURCE_CKPT"; then
    selected="outputs/double_integrator_pullback/trials/trial_three_obstacle_022_mix060"
    selected_steps=50000; selected_mix=0.60; selected_terminal_radius=0.0; selected_terminal_coef=0.0
  else
    run_trial "trial_three_obstacle_023_mix065" 50000 0.65 0.0 0.0 "$SOURCE_CKPT" || true
    source "$ROOT/last_decision.env"
    if [ "$PASSED" = "1" ]; then
      selected="outputs/double_integrator_pullback/trials/trial_three_obstacle_023_mix065"
      selected_steps=50000; selected_mix=0.65; selected_terminal_radius=0.0; selected_terminal_coef=0.0
    elif [ "$ROUTE_OK" = "1" ] && [ "$CONDITIONED_ROUTE_OK" = "1" ]; then
      if run_trial "trial_three_obstacle_023_mix065" 100000 0.65 0.0 0.0 "outputs/double_integrator_pullback/trials/trial_three_obstacle_023_mix065/curvature/checkpoint_latest.pkl"; then
        selected="outputs/double_integrator_pullback/trials/trial_three_obstacle_023_mix065"
        selected_steps=100000; selected_mix=0.65; selected_terminal_radius=0.0; selected_terminal_coef=0.0
      else
        source "$ROOT/last_decision.env"
        if [ "$ROUTE_OK" = "1" ] && [ "$CONDITIONED_ROUTE_OK" = "1" ]; then
          run_trial "trial_three_obstacle_024_mix070" 50000 0.70 0.0 0.0 "$SOURCE_CKPT" || true
          source "$ROOT/last_decision.env"
          if [ "$PASSED" = "1" ]; then
            selected="outputs/double_integrator_pullback/trials/trial_three_obstacle_024_mix070"
            selected_steps=50000; selected_mix=0.70; selected_terminal_radius=0.0; selected_terminal_coef=0.0
          fi
        fi
      fi
    fi
  fi

  if [ -z "$selected" ]; then
    # Terminal-only pull queue. Each run starts from trial_013.
    for spec in "trial_three_obstacle_025_terminal_coef15 1.5" "trial_three_obstacle_026_terminal_coef20 2.0" "trial_three_obstacle_027_terminal_coef25 2.5"; do
      set -- $spec
      trial_name="$1"
      coef="$2"
      if run_trial "$trial_name" 50000 0.50 1.20 "$coef" "$SOURCE_CKPT"; then
        selected="outputs/double_integrator_pullback/trials/$trial_name"
        selected_steps=50000; selected_mix=0.50; selected_terminal_radius=1.20; selected_terminal_coef="$coef"
        break
      fi
      source "$ROOT/last_decision.env"
      if [ "$ROUTE_OK" != "1" ] || [ "$CONDITIONED_ROUTE_OK" != "1" ]; then
        log_status "$trial_name did not preserve route criteria; continuing cautiously to next terminal coefficient."
      fi
    done
  fi

  if [ -z "$selected" ]; then
    best=$(best_diagnostic_trial)
    write_failure_report "$best"
    write_summary "no_pass_candidate" "none" "$best"
    log_status "No pass candidate found. Best diagnostic trial: $best"
    exit 0
  fi

  log_status "Found curvature candidate: $selected"
  train_vanilla_and_final_eval "$selected" "$selected_steps" "$selected_mix" "$selected_terminal_radius" "$selected_terminal_coef"
  write_summary "candidate_found" "$selected" "$selected"
  log_status "Second overnight route-preserving mode completed with candidate: $selected"
}

main "$@"
