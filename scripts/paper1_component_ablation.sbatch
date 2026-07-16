#!/bin/bash
#SBATCH --job-name=toolgeo_ablate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=gpu3005
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=23:59:59
#SBATCH --output=/oscar/scratch/zliu328/llm_tool_ckpt/logs/toolgeo_ablate_%j.out

set -euo pipefail
ROOT=/users/zliu328/llm_tool
PY=/users/zliu328/.venv/bin/python
cd "$ROOT"
export HF_HOME=/oscar/scratch/zliu328/llm_tool_ckpt/hf
export PYTHONPATH="$ROOT/src"
export TOKENIZERS_PARALLELISM=false

mkdir -p outputs/seal_qwen35_4b /oscar/scratch/zliu328/llm_tool_ckpt/logs
nvidia-smi
$PY -m toolgeo validate-data --input data/raw/seal_tools_train
$PY -m toolgeo ablate-hf \
  --input data/raw/seal_tools_train \
  --model-id Qwen/Qwen3.5-4B \
  --cache-dir /oscar/scratch/zliu328/llm_tool_ckpt/hf \
  --output outputs/seal_qwen35_4b/component_ablation.json
