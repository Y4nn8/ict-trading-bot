#!/usr/bin/env bash
# Fetch Optuna results and logs from the VPS.
#
# Usage: ./fetch-results.sh [local_dest]
#   Default dest: ../results/vps_<timestamp>/
set -euo pipefail
cd "$(dirname "$0")/.."

IP=$(terraform output -raw server_ip)
DEST="${1:-$(dirname "$0")/../../results/vps_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$DEST"

echo "==> Fetching results from $IP..."

# Optuna CSV outputs
scp -o StrictHostKeyChecking=no -r \
  "root@$IP:/mnt/data/bot/config/midas_optuna_*" \
  "$DEST/" 2>/dev/null || echo "  No optuna CSVs found"

# Logs
scp -o StrictHostKeyChecking=no -r \
  "root@$IP:/mnt/data/bot/logs/" \
  "$DEST/logs/" 2>/dev/null || echo "  No logs found"

# Best params YAMLs
scp -o StrictHostKeyChecking=no \
  "root@$IP:/mnt/data/bot/config/midas_best_params*.yml" \
  "$DEST/" 2>/dev/null || echo "  No best params found"

# Model binaries
scp -o StrictHostKeyChecking=no \
  "root@$IP:/mnt/data/bot/config/*.bin" \
  "$DEST/" 2>/dev/null || echo "  No model binaries found"

echo "==> Results saved to: $DEST"
ls -lh "$DEST"
