#!/bin/bash
set -euo pipefail

ROOT=/users/zliu328/llm_tool
SCRATCH=/oscar/scratch/zliu328/llm_tool_ckpt
PUBLISHED="$SCRATCH/published_lenses/neuronpedia-a4114d7"

CONFIG="$ROOT/configs/rq1_bfcl_qwen3_0_6b_nothink.yaml" \
LENS="$SCRATCH/lenses/qwen3_0_6b/wikitext_100/jacobian_lens.pt" \
RESULT_NAME=wikitext_100 \
  sbatch "$ROOT/scripts/read_menu_order_jacobian_trajectories.sbatch"

for size in 1.7 4 8 14; do
  config_size="${size/./_}"
  CONFIG="$ROOT/configs/rq1_bfcl_qwen3_${config_size}b_nothink.yaml" \
  LENS="$PUBLISHED/qwen3-${size}b/jlens/Salesforce-wikitext/Qwen3-${size}B_jacobian_lens.pt" \
  RESULT_NAME=published_wikitext \
    sbatch "$ROOT/scripts/read_menu_order_jacobian_trajectories.sbatch"
done

