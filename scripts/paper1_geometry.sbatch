#!/bin/bash
#SBATCH --job-name=toolgeo_p1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=gpu3005
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=23:59:59
#SBATCH --output=/oscar/scratch/zliu328/llm_tool_ckpt/logs/toolgeo_p1_%j.out

set -euo pipefail
ROOT=/users/zliu328/llm_tool
PY=/users/zliu328/.venv/bin/python
CONFIG="${CONFIG:?CONFIG must point to one Paper-1 YAML config}"
cd "$ROOT"
export HF_HOME=/oscar/scratch/zliu328/llm_tool_ckpt/hf
export PYTHONPATH="$ROOT/src"
export TOKENIZERS_PARALLELISM=false

read -r DATA MODEL CACHE MEASUREMENTS BASELINE_MODEL BASELINE OUTPUT BRANCH_BATCH <<< "$($PY - "$CONFIG" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
print(
    c['data']['path'], c['model']['id'], c['model']['cache_dir'],
    c['measurements']['path'], c['baseline']['model_id'], c['baseline']['path'],
    c['run']['output_dir'],
    c['measurements'].get('max_branch_batch', 8),
)
PY
)"

echo "job=${SLURM_JOB_ID:-local} host=$(hostname) model=$MODEL data=$DATA"
nvidia-smi
$PY -m toolgeo validate-data --input "$DATA"

mkdir -p "$(dirname "$BASELINE")" "$MEASUREMENTS" "$OUTPUT"
if [[ ! -f "$BASELINE" ]]; then
  $PY -m toolgeo extract-baselines-hf \
    --input "$DATA" --model-id "$BASELINE_MODEL" --cache-dir "$CACHE" --output "$BASELINE"
fi
if [[ ! -f "$MEASUREMENTS/geometry_index.npz" ]]; then
  $PY -m toolgeo measure-hf \
    --input "$DATA" --model-id "$MODEL" --cache-dir "$CACHE" \
    --max-branch-batch "$BRANCH_BATCH" \
    --output "$MEASUREMENTS"
fi
$PY -m toolgeo analyze \
  --measurements "$MEASUREMENTS" --semantic-baseline "$BASELINE" \
  --output "$OUTPUT/report.json"
