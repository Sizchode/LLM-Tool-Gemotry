#!/bin/bash
#SBATCH --job-name=analyze_tool_cards
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/oscar/scratch/zliu328/llm_tool_ckpt/logs/analyze_tool_cards_%j.out

set -euo pipefail

ROOT=/users/zliu328/llm_tool
PY=/users/zliu328/.venv/bin/python
CONFIG="${CONFIG:?CONFIG must point to a Qwen3 BFCL YAML config}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "job=${SLURM_JOB_ID:-local} host=$(hostname) config=$CONFIG"
$PY -m toolgeo.analyze_tool_cards --config "$CONFIG"
