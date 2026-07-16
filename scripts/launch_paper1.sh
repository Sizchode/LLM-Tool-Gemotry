#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE=local
if [[ "${1:-}" == "--slurm" ]]; then MODE=slurm; shift; fi
CONFIG="${1:?usage: scripts/launch_paper1.sh [--slurm] CONFIG.yaml}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PY="${PY:-/users/zliu328/.venv/bin/python}"
if [[ "$MODE" == "slurm" ]]; then
  export CONFIG
  exec sbatch "$ROOT/scripts/paper1_geometry.sbatch"
fi
exec "$PY" -m toolgeo run --config "$CONFIG"
