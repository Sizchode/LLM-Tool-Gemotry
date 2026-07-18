#!/bin/bash
set -euo pipefail

ROOT=/users/zliu328/llm_tool
SCRATCH=/oscar/scratch/zliu328/llm_tool_ckpt
PUBLISHED="$SCRATCH/published_lenses/neuronpedia-a4114d7"
WIKITEXT_FIT="$ROOT/configs/jacobian_lens_fit_wikitext_100.yaml"
BFCL_FIT="$ROOT/configs/jacobian_lens_fit_bfcl.yaml"

sizes=(0_6 1_7 4 8 14)
published_sizes=(1_7 4 8 14)

for size in "${published_sizes[@]}"; do
  case "$size" in
    1_7) lens_size=1.7; model_size=1.7 ;;
    *) lens_size="$size"; model_size="$size" ;;
  esac
  config="$ROOT/configs/rq1_bfcl_qwen3_${size}b_nothink.yaml"
  lens="$PUBLISHED/qwen3-${lens_size}b/jlens/Salesforce-wikitext/Qwen3-${model_size}B_jacobian_lens.pt"
  CONFIG="$config" LENS="$lens" RESULT_NAME=published_wikitext \
    sbatch "$ROOT/scripts/read_tool_calls_with_jacobian_lens.sbatch"
done

for size in "${sizes[@]}"; do
  config="$ROOT/configs/rq1_bfcl_qwen3_${size}b_nothink.yaml"
  wiki_dir="$SCRATCH/lenses/qwen3_${size}b/wikitext_100"
  bfcl_dir="$SCRATCH/lenses/qwen3_${size}b/bfcl_original_all"

  wiki_job=$(EXPERIMENT_CONFIG="$config" FIT_CONFIG="$WIKITEXT_FIT" \
    OUTPUT_DIR="$wiki_dir" sbatch --parsable "$ROOT/scripts/fit_jacobian_lens.sbatch")
  CONFIG="$config" LENS="$wiki_dir/jacobian_lens.pt" RESULT_NAME=wikitext_100 \
    sbatch --dependency="afterok:$wiki_job" \
    "$ROOT/scripts/read_tool_calls_with_jacobian_lens.sbatch"

  bfcl_job=$(EXPERIMENT_CONFIG="$config" FIT_CONFIG="$BFCL_FIT" \
    OUTPUT_DIR="$bfcl_dir" sbatch --parsable "$ROOT/scripts/fit_jacobian_lens.sbatch")
  CONFIG="$config" LENS="$bfcl_dir/jacobian_lens.pt" RESULT_NAME=bfcl_original_all \
    sbatch --dependency="afterok:$bfcl_job" \
    "$ROOT/scripts/read_tool_calls_with_jacobian_lens.sbatch"
done

