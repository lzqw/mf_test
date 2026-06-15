#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_ROOT="outputs/double_integrator_pullback"
TRIAL_ROOT="${OUT_ROOT}/trials"
mkdir -p "${TRIAL_ROOT}"

SEED=0
TOTAL_STEPS=300000
EVAL_EPISODES=100
DT=0.1
A_MAX=3.0
DAMPING=0.98

train_one_trial() {
  local trial_id="$1"
  local algo="$2"
  local steps="$3"
  local outdir="${OUT_ROOT}/${algo}_seed${SEED}"
  local trial_dir="${TRIAL_ROOT}/${trial_id}"
  mkdir -p "${trial_dir}"

  local cmd_train=(
    python scripts/train_double_integrator_pullback.py
    --algo "${algo}"
    --seed "${SEED}"
    --outdir "${outdir}"
    --total_steps "${steps}"
    --dt "${DT}"
    --a_max "${A_MAX}"
    --damping "${DAMPING}"
    --eval_interval 5000
    --eval_episodes "${EVAL_EPISODES}"
    --eval_delta 0.0
    --start_y_range 0.45
  )

  printf '{\n  "trial_id": "%s",\n  "algo": "%s",\n  "seed": %d,\n  "total_steps": %d,\n  "train_cmd": "%s"\n}\n' \
    "${trial_id}" "${algo}" "${SEED}" "${steps}" "${cmd_train[*]}" > "${trial_dir}/config.json"

  set +e
  "${cmd_train[@]}" | tee "${trial_dir}/train.log"
  local train_rc=${PIPESTATUS[0]}
  set -e

  if [[ ${train_rc} -ne 0 ]]; then
    echo "train failed for ${algo} in ${trial_id} with code ${train_rc}" | tee "${trial_dir}/notes.txt"
    return 1
  fi

  local ckpt="${outdir}/checkpoint.pkl"
  if [[ ! -f "${ckpt}" ]]; then
    echo "checkpoint missing: ${ckpt}" | tee "${trial_dir}/notes.txt"
    return 1
  fi

  local eval_outdir="${trial_dir}/eval"
  mkdir -p "${eval_outdir}"
  local cmd_eval=(
    python eval/eval_double_integrator_pullback.py
    --checkpoint "${ckpt}"
    --algo "${algo}"
    --episodes "${EVAL_EPISODES}"
    --seed "${SEED}"
    --delta 0.0
    --outdir "${eval_outdir}"
    --save_rollouts
  )

  echo "running: ${cmd_eval[*]}" | tee "${trial_dir}/notes.txt"
  set +e
  "${cmd_eval[@]}" | tee -a "${trial_dir}/notes.txt"
  local eval_rc=${PIPESTATUS[0]}
  set -e

  if [[ ${eval_rc} -ne 0 ]]; then
    echo "evaluation failed for ${algo} in ${trial_id}" | tee -a "${trial_dir}/notes.txt"
    return 1
  fi

  cp "${eval_outdir}/eval_summary.json" "${trial_dir}/eval_summary.json" 2>/dev/null || true
  cp "${eval_outdir}/eval_metrics.csv" "${trial_dir}/eval_metrics.csv" 2>/dev/null || true
  cp "${eval_outdir}/rollouts.npz" "${trial_dir}/rollouts_delta_0.0_${algo}.npz" 2>/dev/null || true
}

trial=1
for algo in vanilla_flow curvature_flow; do
  id=$(printf "trial_%03d" "${trial}")
  echo "=== starting ${algo} (${id}) ==="
  if ! train_one_trial "${id}" "${algo}" "${TOTAL_STEPS}"; then
    echo "trial ${id} failed"
  else
    echo "trial ${id} finished"
  fi
  trial=$((trial + 1))
done

cat > "${TRIAL_ROOT}/README.txt" <<'EOF'
Auto-train log:
each trial stores:
  - config.json
  - eval_summary.json (if eval succeeds)
  - notes.txt
  - rollouts_delta_0.0_<algo>.npz (if evaluation saves rollouts)
EOF
