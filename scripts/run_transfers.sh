#!/bin/bash
# Run the full cross-dataset transfer matrix for Paper 1.
#
# Usage:
#   scripts/run_transfers.sh [CONFIG_A.yaml CONFIG_B.yaml CONFIG_C.yaml]
#
# Every ordered pair (source != target) is evaluated once. Each per-dataset
# config must already have a completed run (features, baselines, behavior),
# which scripts/launch_paper1.sh produces. Reports land in outputs/transfers/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE=run
if [[ "${1:-}" == "--print" ]]; then MODE=print; shift; fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PY="${PY:-/users/zliu328/.venv/bin/python}"
OUT_DIR="${TRANSFER_OUT:-$ROOT/outputs/transfers}"
if [[ "$MODE" == "run" ]]; then mkdir -p "$OUT_DIR"; fi
if [[ $# -eq 0 ]]; then
  set -- \
    configs/paper1_bfcl_v4_qwen35_9b.yaml \
    configs/paper1_sealtools_train_qwen35_9b.yaml \
    configs/paper1_toolhop_qwen35_9b.yaml
fi
if [[ $# -lt 2 ]]; then
  echo "usage: scripts/run_transfers.sh CONFIG_A.yaml CONFIG_B.yaml [...]" >&2
  exit 2
fi
stem() { basename "$1" .yaml; }
for SRC in "$@"; do
  for TGT in "$@"; do
    [[ "$SRC" == "$TGT" ]] && continue
    NAME="$(stem "$SRC")__to__$(stem "$TGT")"
    REPORT="$OUT_DIR/$NAME.json"
    if [[ -f "$REPORT" ]]; then
      echo "skip existing $REPORT"
      continue
    fi
    echo "transfer: $SRC -> $TGT"
    if [[ "$MODE" == "print" ]]; then
      echo "$PY -m toolgeo transfer --source-config $SRC --target-config $TGT --output $REPORT"
      continue
    fi
    "$PY" -m toolgeo transfer --source-config "$SRC" --target-config "$TGT" --output "$REPORT"
  done
done
if [[ "$MODE" == "print" ]]; then
  echo "transfer matrix preview complete: $OUT_DIR"
else
  echo "transfer matrix complete: $OUT_DIR"
fi
