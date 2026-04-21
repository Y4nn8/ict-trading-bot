#!/usr/bin/env bash
# Show training status: tmux state, nvidia-smi, tail of latest log.
#
# Usage:
#   ./status.sh            # one-shot snapshot
#   ./status.sh --follow   # live tail (Ctrl-C to stop — does not kill remote job)
set -euo pipefail
cd "$(dirname "$0")"
source ./lib.sh

FOLLOW=""
if [ "${1:-}" = "--follow" ] || [ "${1:-}" = "-f" ]; then
  FOLLOW="1"
fi

RUN_NAME=""
[ -f .last-run ] && RUN_NAME="$(cat .last-run)"

if [ -n "$FOLLOW" ]; then
  if [ -z "$RUN_NAME" ]; then
    echo "ERROR: no .last-run file — run train.sh first or pass the log path manually." >&2
    exit 1
  fi
  echo "==> Tailing logs/${RUN_NAME}.log (Ctrl-C to detach)..."
  vast_ssh "tail -f '${VAST_REMOTE_DIR}/logs/${RUN_NAME}.log'"
  exit 0
fi

echo "==> tmux sessions:"
vast_ssh "tmux ls 2>/dev/null || echo '  (none)'"

echo ""
echo "==> CPU status:"
vast_ssh "ps -eo pid,pcpu,pmem,etime,comm --sort=-pcpu | head -6"

echo ""
echo "==> GPU status:"
vast_ssh "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader || echo '  nvidia-smi unavailable'"

echo ""
if [ -n "$RUN_NAME" ]; then
  echo "==> Tail logs/${RUN_NAME}.log:"
  vast_ssh "tail -n 30 '${VAST_REMOTE_DIR}/logs/${RUN_NAME}.log' 2>/dev/null || echo '  log not found yet'"
else
  echo "==> No .last-run file — pass --follow after running train.sh"
fi
