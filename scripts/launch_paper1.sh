#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE=local
if [[ "${1:-}" == "--slurm" ]]; then MODE=slurm; shift
elif [[ "${1:-}" == "--analysis-only" ]]; then MODE=analysis; shift
fi
CONFIG="${1:?usage: scripts/launch_paper1.sh [--slurm|--analysis-only] CONFIG.yaml}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PY="${PY:-/users/zliu328/.venv/bin/python}"
if [[ "$MODE" == "slurm" ]]; then
  mkdir -p /oscar/scratch/zliu328/llm_tool_ckpt/logs
  export CONFIG
  exec sbatch "$ROOT/scripts/paper1_geometry.sbatch"
fi
if [[ "$MODE" == "analysis" ]]; then
  exec "$PY" -m toolgeo run --config "$CONFIG"
fi
export CONFIG
exec bash "$ROOT/scripts/paper1_geometry.sbatch"
