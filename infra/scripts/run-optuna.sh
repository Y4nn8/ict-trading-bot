#!/usr/bin/env bash
# Launch an Optuna run on the VPS via SSH.
# All extra args are passed to run_midas_wf_optuna.
#
# Usage:
#   ./run-optuna.sh --start 2025-10-01 --end 2026-04-01 \
#       --outer-trials 10 --inner-trials 20 \
#       --output config/midas_optuna_step1
#
# Options:
#   --auto-shutdown   Power off the server after the run finishes.
#                     You then run teardown.sh to destroy it.
set -euo pipefail
cd "$(dirname "$0")/.."

IP=$(terraform output -raw server_ip)

# Separate --auto-shutdown from optuna args
AUTO_SHUTDOWN=""
OPTUNA_ARGS=()
for arg in "$@"; do
  if [ "$arg" = "--auto-shutdown" ]; then
    AUTO_SHUTDOWN="true"
  else
    OPTUNA_ARGS+=("$arg")
  fi
done

SHUTDOWN_CMD=""
if [ -n "$AUTO_SHUTDOWN" ]; then
  SHUTDOWN_CMD="echo '==> Run finished, shutting down...' && sudo poweroff"
  echo "==> Auto-shutdown enabled: server will power off after the run"
fi

echo "==> Launching Optuna on $IP..."
echo "  Args: ${OPTUNA_ARGS[*]}"

# Run via nohup so it survives SSH disconnect
ssh -o StrictHostKeyChecking=no root@"$IP" bash <<REMOTE
  set -euo pipefail
  export PATH="/root/.local/bin:\$PATH"
  cd /mnt/data/bot

  # Pull latest code
  git fetch origin && git pull origin \$(git rev-parse --abbrev-ref HEAD) || true

  mkdir -p logs

  nohup bash -c '
    export PATH="/root/.local/bin:\$PATH"
    cd /mnt/data/bot
    uv run python -m scripts.run_midas_wf_optuna ${OPTUNA_ARGS[*]} \
      > logs/optuna_run.log 2>&1
    echo "EXIT_CODE=\$?" >> logs/optuna_run.log
    ${SHUTDOWN_CMD}
  ' > /dev/null 2>&1 &

  echo "==> Optuna PID: \$!"
  echo "==> Logs: ssh root@$IP tail -f /mnt/data/bot/logs/optuna_run.log"
REMOTE
