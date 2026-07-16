#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:---print}"
if [[ "$MODE" != "--print" && "$MODE" != "--submit" ]]; then
  echo "usage: scripts/launch_matrix.sh [--print|--submit]" >&2
  exit 2
fi

configs=(
  configs/paper1_bfcl_qwen35_9b.yaml
  configs/paper1_bfcl_qwen35_4b.yaml
  configs/paper1_bfcl_gemma3_4b.yaml
  configs/paper1_seal_qwen35_9b.yaml
  configs/paper1_seal_qwen35_4b.yaml
  configs/paper1_seal_gemma3_4b.yaml
  configs/paper1_toolhop_qwen35_9b.yaml
  configs/paper1_toolhop_qwen35_4b.yaml
  configs/paper1_toolhop_gemma3_4b.yaml
)

for config in "${configs[@]}"; do
  if [[ "$MODE" == "--print" ]]; then
    echo "CONFIG=$config sbatch scripts/paper1_geometry.sbatch"
  else
    CONFIG="$config" sbatch --export="ALL,CONFIG=$config" scripts/paper1_geometry.sbatch
  fi
done
