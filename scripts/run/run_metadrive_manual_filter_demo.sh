#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python scripts/interactive_metadrive_manual_filter_demo.py \
  --env_name FlatThreeLaneStraight \
  --filter_type sample_vo \
  --use_filter \
  --seed 0 \
  --show_status_panel
