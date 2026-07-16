#!/bin/bash
# Submit the preregistered 3 models x 3 datasets as one dependency chain.
# Chaining prevents two jobs from racing to create the same cross-model
# geometry shard. Use --print to inspect commands without changing Slurm state.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:---print}"
if [[ "$MODE" != "--print" && "$MODE" != "--submit" ]]; then
  echo "usage: scripts/launch_matrix.sh [--print|--submit]" >&2
  exit 2
fi
CONFIGS=(
  configs/paper1_bfcl_v4_qwen35_9b.yaml
  configs/paper1_bfcl_v4_qwen35_9b_base.yaml
  configs/paper1_bfcl_v4_gemma3_4b.yaml
  configs/paper1_sealtools_train_qwen35_9b.yaml
  configs/paper1_sealtools_train_qwen35_9b_base.yaml
  configs/paper1_sealtools_train_gemma3_4b.yaml
  configs/paper1_toolhop_qwen35_9b.yaml
  configs/paper1_toolhop_qwen35_9b_base.yaml
  configs/paper1_toolhop_gemma3_4b.yaml
)
cd "$ROOT"
if [[ "$MODE" == "--print" ]]; then
  for config in "${CONFIGS[@]}"; do
    echo "CONFIG=$config sbatch scripts/paper1_geometry.sbatch"
  done
  exit 0
fi
mkdir -p /oscar/scratch/zliu328/llm_tool_ckpt/logs
previous=""
for config in "${CONFIGS[@]}"; do
  arguments=(--parsable --export="ALL,CONFIG=$config")
  if [[ -n "$previous" ]]; then
    arguments+=(--dependency="afterok:$previous")
  fi
  job_id="$(sbatch "${arguments[@]}" scripts/paper1_geometry.sbatch)"
  echo "$config -> $job_id"
  previous="${job_id%%;*}"
done
