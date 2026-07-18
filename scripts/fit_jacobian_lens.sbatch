#!/bin/bash
#SBATCH --job-name=fit_jacobian_lens
#SBATCH --partition=gpu-he
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=gpu3005
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/oscar/scratch/zliu328/llm_tool_ckpt/logs/fit_jacobian_lens_%j.out

set -euo pipefail

ROOT=/users/zliu328/llm_tool
JLENS_ROOT=/oscar/scratch/zliu328/llm_tool_ckpt/third_party/jacobian-lens
PY=/users/zliu328/.venv/bin/python
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:?EXPERIMENT_CONFIG is required}"
FIT_CONFIG="${FIT_CONFIG:?FIT_CONFIG is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$JLENS_ROOT"
$PY -m toolgeo.fit_jacobian_lens \
  --experiment-config "$EXPERIMENT_CONFIG" \
  --fit-config "$FIT_CONFIG" \
  --output-dir "$OUTPUT_DIR"

