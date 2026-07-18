#!/bin/bash
#SBATCH --job-name=menu_order_jacobian
#SBATCH --partition=gpu-he
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=gpu3005
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/oscar/scratch/zliu328/llm_tool_ckpt/logs/menu_order_jacobian_%j.out

set -euo pipefail

ROOT=/users/zliu328/llm_tool
JLENS_ROOT=/oscar/scratch/zliu328/llm_tool_ckpt/third_party/jacobian-lens
PY=/users/zliu328/.venv/bin/python
CONFIG="${CONFIG:?CONFIG is required}"
LENS="${LENS:?LENS is required}"
RESULT_NAME="${RESULT_NAME:?RESULT_NAME is required}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$JLENS_ROOT"
$PY -m toolgeo.read_menu_order_jacobian_trajectories \
  --config "$CONFIG" \
  --lens "$LENS" \
  --result-name "$RESULT_NAME"

