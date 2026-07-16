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
read -r DATA MODEL CACHE LAYERS FEATURES BASELINE_MODEL BASELINES MENU_REPEATS BRANCH_BATCH <<< "$("$PY" - "$CONFIG" <<'PY'
import sys, yaml
c=yaml.safe_load(open(sys.argv[1]))
f=c['features']
b=c['baselines']
print(c['data']['path'], f['model_id'], f['cache_dir'], f.get('layers', 'all'), f['path'], b['model_id'], b['path'], c.get('rollout', {}).get('menu_repeats', 3), c.get('rollout', {}).get('max_branch_batch', 8))
PY
)"
if [[ ! -f "$DATA/tools.jsonl" || ! -f "$DATA/decisions.jsonl" ]]; then
  echo "Normalized dataset is missing at $DATA. Run the matching toolgeo import-* command first." >&2
  exit 2
fi
"$PY" -m toolgeo validate-data --input "$DATA"
"$PY" -m toolgeo validate-model-template --input "$DATA" --model-id "$MODEL" --cache-dir "$CACHE"
if ! "$PY" -m toolgeo validate-features --input "$FEATURES" --data "$DATA" >/dev/null 2>&1; then
  echo "Extracting standalone-card geometry: model=$MODEL layers=$LAYERS output=$FEATURES"
  "$PY" -m toolgeo extract-hf --input "$DATA" --model-id "$MODEL" --cache-dir "$CACHE" --layers "$LAYERS" --output "$FEATURES"
fi
if [[ ! -f "$BASELINES" ]]; then
  echo "Extracting strong semantic baseline: model=$BASELINE_MODEL output=$BASELINES"
  "$PY" -m toolgeo extract-baselines-hf --input "$DATA" --model-id "$BASELINE_MODEL" --cache-dir "$CACHE" --output "$BASELINES"
fi
# Cross-model specificity controls: extract any missing control geometry on
# the identical tool inventory with the control model.
while IFS='|' read -r CTRL_NAME CTRL_MODEL CTRL_PATH; do
  [[ -z "$CTRL_NAME" ]] && continue
  if [[ -z "$CTRL_MODEL" && ! -f "$CTRL_PATH" ]]; then
    echo "control_paths entry '$CTRL_NAME' lacks model_id and $CTRL_PATH is absent" >&2
    exit 2
  fi
  if ! "$PY" -m toolgeo validate-features --input "$CTRL_PATH" --data "$DATA" >/dev/null 2>&1; then
    echo "Extracting control geometry: name=$CTRL_NAME model=$CTRL_MODEL output=$CTRL_PATH"
    "$PY" -m toolgeo extract-hf --input "$DATA" --model-id "$CTRL_MODEL" --cache-dir "$CACHE" --layers "$LAYERS" --output "$CTRL_PATH"
  fi
done <<< "$("$PY" - "$CONFIG" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
for entry in c.get('features', {}).get('control_paths', []) or []:
    print(f"{entry['name']}|{entry.get('model_id','')}|{entry['path']}")
PY
)"
BEHAVIOR="$($PY - "$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))['data'].get('behavior_path', ''))
PY
)"
if [[ -n "$BEHAVIOR" && ! -f "$BEHAVIOR/decisions.jsonl" ]]; then
  echo "Running native tool-call constrained-trie rollout: output=$BEHAVIOR"
  "$PY" -m toolgeo rollout-hf --input "$DATA" --model-id "$MODEL" --cache-dir "$CACHE" --output "$BEHAVIOR" --menu-repeats "$MENU_REPEATS" --max-branch-batch "$BRANCH_BATCH" --seed "$($PY - "$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))['run']['seed'])
PY
)"
fi
OPAQUE_BEHAVIOR="$($PY - "$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))['data'].get('opaque_behavior_path', ''))
PY
)"
if [[ -n "$OPAQUE_BEHAVIOR" && ! -f "$OPAQUE_BEHAVIOR/decisions.jsonl" ]]; then
  echo "Running opaque-name control rollout: output=$OPAQUE_BEHAVIOR"
  "$PY" -m toolgeo rollout-hf --input "$DATA" --model-id "$MODEL" --cache-dir "$CACHE" --output "$OPAQUE_BEHAVIOR" --menu-repeats "$MENU_REPEATS" --max-branch-batch "$BRANCH_BATCH" --seed "$($PY - "$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))['run']['seed'])
PY
)" --opaque-names
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
    "$PY" -m toolgeo extract-decision-hf --input "$BEHAVIOR" --model-id "$MODEL" --cache-dir "$CACHE" --layers "$PROBE_LAYERS" --output "$PROBE_CONTEXTS"
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
