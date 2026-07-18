#!/bin/bash
#SBATCH --job-name=jacobian_lens_bfcl
#SBATCH --partition=gpu-he
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=gpu3005
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/oscar/scratch/zliu328/llm_tool_ckpt/logs/jacobian_lens_bfcl_%j.out

set -euo pipefail

ROOT=/users/zliu328/llm_tool
JLENS_ROOT=/oscar/scratch/zliu328/llm_tool_ckpt/third_party/jacobian-lens
PY=/users/zliu328/.venv/bin/python
CONFIG="${CONFIG:?CONFIG must point to the completed Qwen3-8B BFCL run}"
LENS="${LENS:?LENS must point to the fitted lens for the configured checkpoint}"
RESULT_NAME="${RESULT_NAME:?RESULT_NAME identifies the fitting corpus}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$JLENS_ROOT"
echo "job=${SLURM_JOB_ID:-local} host=$(hostname) config=$CONFIG lens=$LENS"
nvidia-smi
$PY -m toolgeo.read_tool_calls_with_jacobian_lens \
  --config "$CONFIG" \
  --lens "$LENS" \
  --result-name "$RESULT_NAME"
