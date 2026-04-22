#!/usr/bin/env bash
# Show progress of the running Optuna walk-forward on the VPS.
set -euo pipefail
cd "$(dirname "$0")/.."

IP=$(terraform output -raw server_ip)

ssh -o StrictHostKeyChecking=no root@"$IP" bash <<'REMOTE'
  set -euo pipefail
  cd /mnt/data/bot

  echo "==> Process:"
  ps -eo pid,pcpu,pmem,etime,comm --sort=-pcpu | head -5

  echo ""
  echo "==> Log tail (last 25 lines):"
  tail -25 logs/optuna_run.log 2>/dev/null || echo "  (no log yet)"

  echo ""
  echo "==> Trial CSV stats:"
  TRIALS=$(ls -1t logs/midas_robust_6mo_trials.csv 2>/dev/null | head -1)
  if [ -n "$TRIALS" ] && [ -f "$TRIALS" ]; then
    NLINES=$(wc -l < "$TRIALS")
    NTRIALS=$((NLINES - 1))
    echo "  File: $TRIALS"
    echo "  Completed trials: $NTRIALS"
    # Last window seen
    LAST_WIN=$(awk -F, 'NR > 1 {print $1}' "$TRIALS" | sort -n | tail -1)
    echo "  Last window: W$LAST_WIN"
    echo ""
    echo "  Last 5 trials:"
    tail -5 "$TRIALS" | awk -F, '{printf "    W%s T%s | test_pnl=%-8s train_pnl=%s\n", $1, $2, $6, $10}'
  else
    echo "  (no trials yet)"
  fi
REMOTE
