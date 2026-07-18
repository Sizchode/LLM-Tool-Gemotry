#!/bin/bash
#SBATCH --job-name=tool_card_geometry
#SBATCH --partition=gpu-he
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=gpu3005
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:59
#SBATCH --output=/oscar/scratch/zliu328/llm_tool_ckpt/logs/tool_card_geometry_%j.out

set -euo pipefail

ROOT=/users/zliu328/llm_tool
PY=/users/zliu328/.venv/bin/python
CONFIG="${CONFIG:?CONFIG must point to a Qwen3 BFCL YAML config}"

cd "$ROOT"
export HF_HOME=/oscar/scratch/zliu328/llm_tool_ckpt/hf
export PYTHONPATH="$ROOT/src"
export TOKENIZERS_PARALLELISM=false

echo "job=${SLURM_JOB_ID:-local} host=$(hostname) config=$CONFIG"
nvidia-smi
$PY -m toolgeo.extract_tool_cards --config "$CONFIG"
