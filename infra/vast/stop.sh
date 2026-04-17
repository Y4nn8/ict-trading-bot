#!/usr/bin/env bash
# Kill the tmux training session on the Vast.ai instance. Does NOT destroy
# the instance — do that from the Vast.ai UI to stop billing.
#
# Usage: ./stop.sh
set -euo pipefail
cd "$(dirname "$0")"
source ./lib.sh

SESSION="${1:-}"

if [ -n "$SESSION" ]; then
  echo "==> Killing tmux session '${SESSION}'..."
  vast_ssh "tmux kill-session -t '${SESSION}' 2>/dev/null && echo '  killed' || echo '  no session'"
else
  echo "==> Killing all tmux sessions..."
  for s in morpheus policy; do
    vast_ssh "tmux kill-session -t '${s}' 2>/dev/null && echo \"  ${s}: killed\" || echo \"  ${s}: no session\""
  done
fi
