#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMIT=false
if [[ "${1:-}" == "--submit" ]]; then
  SUBMIT=true
  shift
fi

if [[ "$#" -eq 0 ]]; then
  CONFIGS=(
    "$ROOT/configs/rq1_bfcl_qwen3_0_6b_nothink.yaml"
    "$ROOT/configs/rq1_bfcl_qwen3_1_7b_nothink.yaml"
    "$ROOT/configs/rq1_bfcl_qwen3_4b_nothink.yaml"
    "$ROOT/configs/rq1_bfcl_qwen3_8b_nothink.yaml"
    "$ROOT/configs/rq1_bfcl_qwen3_14b_nothink.yaml"
  )
else
  CONFIGS=("$@")
fi

for config in "${CONFIGS[@]}"; do
  config="$(realpath "$config")"
  if [[ "$SUBMIT" == true ]]; then
    extraction_job=$(
      sbatch --parsable --export=ALL,CONFIG="$config" \
        "$ROOT/scripts/extract_tool_card_geometry.sbatch"
    )
    analysis_job=$(
      sbatch --parsable --dependency="afterok:$extraction_job" \
        --export=ALL,CONFIG="$config" \
        "$ROOT/scripts/analyze_tool_card_geometry.sbatch"
    )
    echo "extraction_job=$extraction_job analysis_job=$analysis_job config=$config"
  else
    echo "sbatch --export=ALL,CONFIG=$config $ROOT/scripts/extract_tool_card_geometry.sbatch"
    echo "sbatch --dependency=afterok:<extraction_job_id> --export=ALL,CONFIG=$config $ROOT/scripts/analyze_tool_card_geometry.sbatch"
  fi
done
