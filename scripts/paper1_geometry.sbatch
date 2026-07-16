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
#SBATCH --export=ALL,PYTORCH_JIT=0

set -euo pipefail
ROOT=/users/zliu328/llm_tool
cd "$ROOT"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/oscar/scratch/zliu328/llm_tool_ckpt/hf}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PY="${PY:-/users/zliu328/.venv/bin/python}"
CONFIG="${CONFIG:?CONFIG must be supplied by scripts/launch_paper1.sh}"
mkdir -p /oscar/scratch/zliu328/llm_tool_ckpt/logs
echo "job=${SLURM_JOB_ID:-local} host=$(hostname) config=$CONFIG python=$PY"
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; else echo "nvidia-smi unavailable on this host"; fi
read -r DATA MODEL CACHE LAYER FEATURES <<< "$("$PY" - "$CONFIG" <<'PY'
import sys, yaml
c=yaml.safe_load(open(sys.argv[1]))
f=c['features']
print(c['data']['path'], f['model_id'], f['cache_dir'], f.get('layer', 18), f['path'])
PY
)"
if [[ ! -f "$FEATURES" ]]; then
  echo "Extracting HF features: model=$MODEL layer=$LAYER output=$FEATURES"
  "$PY" -m toolgeo extract-hf --input "$DATA" --model-id "$MODEL" --cache-dir "$CACHE" --layer "$LAYER" --output "$FEATURES"
fi
BEHAVIOR="$($PY - "$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))['data'].get('behavior_path', ''))
PY
)"
if [[ -n "$BEHAVIOR" && ! -f "$BEHAVIOR/decisions.jsonl" ]]; then
  echo "Running deterministic candidate-choice rollout: output=$BEHAVIOR"
  "$PY" -m toolgeo rollout-hf --input "$DATA" --model-id "$MODEL" --cache-dir "$CACHE" --output "$BEHAVIOR"
fi
"$PY" -m toolgeo run --config "$CONFIG"
PROBE_CONTEXTS="$($PY - "$CONFIG" <<'PY'
import sys, yaml
c=yaml.safe_load(open(sys.argv[1])); print(c.get('probe', {}).get('contexts_path', ''))
PY
)"
if [[ -n "$PROBE_CONTEXTS" ]]; then
  PROBE_REPORT="$($PY - "$CONFIG" <<'PY'
import sys, yaml
c=yaml.safe_load(open(sys.argv[1])); print(c['probe']['report_path'])
PY
)"
  if [[ ! -f "$PROBE_CONTEXTS" ]]; then
    PROBE_LAYERS="$($PY - "$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1])).get('probe', {}).get('layers', 'all'))
PY
)"
    "$PY" -m toolgeo extract-decision-hf --input "$DATA" --model-id "$MODEL" --cache-dir "$CACHE" --layers "$PROBE_LAYERS" --output "$PROBE_CONTEXTS"
  fi
  "$PY" -m toolgeo probe-outcome --behavior "$BEHAVIOR" --contexts "$PROBE_CONTEXTS" --output "$PROBE_REPORT" --heldout-fraction "$($PY - "$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1])).get('probe', {}).get('heldout_fraction', .25))
PY
)" --seed "$($PY - "$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))['run']['seed'])
PY
)"
fi
"$PY" -m toolgeo audit-manifest --config "$CONFIG" --output "$($PY - "$CONFIG" <<'PY'
import sys, yaml
c=yaml.safe_load(open(sys.argv[1])); print(c['run']['output_dir'] + '/audit_manifest.json')
PY
)"
