#!/usr/bin/env bash
# Pull checkpoints, metrics CSV, and logs for a training run back to the host.
#
# Usage:
#   ./fetch.sh                    # fetch the run from .last-run
#   ./fetch.sh xauusd_20260415    # fetch a specific run name
set -euo pipefail
cd "$(dirname "$0")"
source ./lib.sh

RUN_NAME="${1:-}"
if [ -z "$RUN_NAME" ] && [ -f .last-run ]; then
  RUN_NAME="$(cat .last-run)"
fi
if [ -z "$RUN_NAME" ]; then
  echo "ERROR: no run name. Pass one as arg or run train.sh first." >&2
  exit 1
fi

LOCAL_RUN_DIR="$REPO_ROOT/runs/morpheus/${RUN_NAME}"
LOCAL_LOG_DIR="$REPO_ROOT/runs/morpheus/_logs"
mkdir -p "$LOCAL_RUN_DIR" "$LOCAL_LOG_DIR"

echo "==> Fetching run ${RUN_NAME}..."
rsync -az \
  -e "ssh ${SSH_OPTS[*]}" \
  "${VAST_SSH_USER}@${VAST_SSH_HOST}:${VAST_REMOTE_DIR}/runs/morpheus/${RUN_NAME}/" \
  "$LOCAL_RUN_DIR/"

echo "==> Fetching log..."
rsync -az \
  -e "ssh ${SSH_OPTS[*]}" \
  "${VAST_SSH_USER}@${VAST_SSH_HOST}:${VAST_REMOTE_DIR}/logs/${RUN_NAME}.log" \
  "$LOCAL_LOG_DIR/" 2>/dev/null || echo "  (log not found)"

echo "==> Done."
echo "  Checkpoints + metrics: $LOCAL_RUN_DIR"
echo "  Log:                   $LOCAL_LOG_DIR/${RUN_NAME}.log"
ls -lh "$LOCAL_RUN_DIR" 2>/dev/null | head -20
