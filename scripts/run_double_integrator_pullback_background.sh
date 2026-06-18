#!/usr/bin/env bash
set -euo pipefail

ALGO=${1:?vanilla_flow or curvature_flow}
OUTDIR=${2:?output dir}
STEPS=${3:-300000}
SEED=${4:-0}
RESUME=${5:-}
CONDA_BIN=${CONDA_BIN:-/home/carla/anaconda3/bin/conda}
PYTHON_BIN=${PYTHON_BIN:-/home/carla/anaconda3/envs/value-flows/bin/python}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
XLA_PREALLOCATE=${XLA_PREALLOCATE:-false}
XLA_MEM_FRACTION=${XLA_MEM_FRACTION:-0.60}
XLA_FLAGS_VALUE=${XLA_FLAGS_VALUE:---xla_gpu_enable_command_buffer=}
TF_FORCE_GPU_ALLOW_GROWTH=${TF_FORCE_GPU_ALLOW_GROWTH:-true}
MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mplconfig}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
EVAL_INTERVAL=${EVAL_INTERVAL:-5000}
CHECKPOINT_NAME=${CHECKPOINT_NAME:-checkpoint.pkl}
MAX_WALLTIME_SEC=${MAX_WALLTIME_SEC:-}
EXTRA_ARGS=${EXTRA_ARGS:-}

mkdir -p "$OUTDIR"
LOG="$OUTDIR/train.log"

CMD="cd $REPO_ROOT && mkdir -p $MPLCONFIGDIR && CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES XLA_FLAGS=$XLA_FLAGS_VALUE XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PREALLOCATE XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_MEM_FRACTION TF_FORCE_GPU_ALLOW_GROWTH=$TF_FORCE_GPU_ALLOW_GROWTH MPLCONFIGDIR=$MPLCONFIGDIR LD_LIBRARY_PATH=:/home/carla/.mujoco/mujoco210/bin:/usr/lib/nvidia $CONDA_BIN run --no-capture-output -n value-flows $PYTHON_BIN -u scripts/train_double_integrator_pullback.py \
  --algo $ALGO \
  --seed $SEED \
  --outdir $OUTDIR \
  --total_steps $STEPS \
  --save_interval $SAVE_INTERVAL \
  --eval_interval $EVAL_INTERVAL \
  --checkpoint_name $CHECKPOINT_NAME"

if [ -n "$RESUME" ]; then
  CMD="$CMD --resume_checkpoint $RESUME"
fi

if [ -n "$MAX_WALLTIME_SEC" ]; then
  CMD="$CMD --max_walltime_sec $MAX_WALLTIME_SEC"
fi

if [ -n "$EXTRA_ARGS" ]; then
  CMD="$CMD $EXTRA_ARGS"
fi

echo "$CMD" | tee "$OUTDIR/launch_command.txt"
setsid nohup /bin/bash -lc "$CMD" > "$LOG" 2>&1 < /dev/null &
WRAPPER_PID=$!
sleep 1
TRAIN_PID=$(pgrep -n -f "train_double_integrator_pullback.py --algo $ALGO --seed $SEED --outdir $OUTDIR" || true)
if [ -z "$TRAIN_PID" ]; then
  TRAIN_PID=$WRAPPER_PID
fi
echo "$TRAIN_PID" | tee "$OUTDIR/pid.txt"
echo "Started $ALGO training with PID $(cat "$OUTDIR/pid.txt")"
echo "Log: $LOG"
