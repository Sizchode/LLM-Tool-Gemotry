#!/bin/bash
#SBATCH --job-name=toolgeo_bfcl
#SBATCH --partition=gpu-he
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=gpu3005
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:59
#SBATCH --output=/oscar/scratch/zliu328/llm_tool_ckpt/logs/rq1_bfcl_%j.out

set -euo pipefail

ROOT=/users/zliu328/llm_tool
PY=/users/zliu328/.venv/bin/python
CONFIG="${CONFIG:?CONFIG must point to an RQ1 BFCL YAML config}"

cd "$ROOT"
export HF_HOME=/oscar/scratch/zliu328/llm_tool_ckpt/hf
export PYTHONPATH="$ROOT/src"
export TOKENIZERS_PARALLELISM=false

echo "job=${SLURM_JOB_ID:-local} host=$(hostname) config=$CONFIG"
nvidia-smi

$PY -m toolgeo.data.bfcl --config "$CONFIG"
$PY -m toolgeo.extract_decision_states --config "$CONFIG"
$PY -m toolgeo.generate_tool_calls --config "$CONFIG"
