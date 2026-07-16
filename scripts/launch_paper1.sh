#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${2:-${1:-}}"
if [[ -z "$CONFIG" ]]; then
  echo "usage: scripts/launch_paper1.sh [--slurm] CONFIG.yaml" >&2
  exit 2
fi
cd "$ROOT"
if [[ "${1:-}" == "--slurm" ]]; then
  mkdir -p /oscar/scratch/zliu328/llm_tool_ckpt/logs
  exec sbatch --export="ALL,CONFIG=$CONFIG" scripts/paper1_geometry.sbatch
fi
export CONFIG PYTHONPATH="$ROOT/src"
exec bash scripts/paper1_geometry.sbatch
