#!/bin/bash
#SBATCH --job-name=toolgeo_bfcl_analysis
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/zliu328/llm_tool_ckpt/logs/analyze_bfcl_%j.out

set -euo pipefail

ROOT=/users/zliu328/llm_tool
PY=/users/zliu328/.venv/bin/python
CONFIG="${CONFIG:?CONFIG must point to an RQ1 BFCL YAML config}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "job=${SLURM_JOB_ID:-local} host=$(hostname) config=$CONFIG"
$PY -m toolgeo.analyze_bfcl --config "$CONFIG"
$PY -m toolgeo.analyze_order --config "$CONFIG"
