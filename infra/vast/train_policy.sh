#!/usr/bin/env bash
# Launch policy training inside a tmux session on the Vast.ai instance.
# Requires a world model checkpoint already on the remote instance.
#
# Usage:
#   ./train_policy.sh --epochs 200 --batch-size 32
#
# The --checkpoint, --parquet-dir, --output-dir, --device flags are set
# here; everything else passes through.
set -euo pipefail
cd "$(dirname "$0")"
source ./lib.sh

RUN_NAME="${RUN_NAME:-policy_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="runs/policy/${RUN_NAME}"
PARQUET_DIR="data/morpheus/xauusd"
CHECKPOINT="runs/morpheus/checkpoint_epoch_18.pt"
SESSION="policy"

EXTRA_ARGS=("$@")

echo "==> Run name: ${RUN_NAME}"
echo "==> Checkpoint: ${VAST_REMOTE_DIR}/${CHECKPOINT}"
echo "==> Output:     ${VAST_REMOTE_DIR}/${OUTPUT_DIR}"
echo "==> Extra:      ${EXTRA_ARGS[*]:-<none>}"

printf -v EXTRA_STR '%q ' "${EXTRA_ARGS[@]}"

vast_ssh bash <<REMOTE
  set -euo pipefail
  export PATH="\$HOME/.local/bin:\$PATH"
  cd '${VAST_REMOTE_DIR}'

  if tmux has-session -t '${SESSION}' 2>/dev/null; then
    echo "ERROR: tmux session '${SESSION}' already exists. Kill with: tmux kill-session -t ${SESSION}" >&2
    exit 1
  fi

  mkdir -p '${OUTPUT_DIR}' logs

  tmux new-session -d -s '${SESSION}' "
    export PATH=\"\$HOME/.local/bin:\$PATH\";
    cd '${VAST_REMOTE_DIR}';
    uv run python -m scripts.train_policy \
      --checkpoint '${CHECKPOINT}' \
      --parquet-dir '${PARQUET_DIR}' \
      --output-dir '${OUTPUT_DIR}' \
      --device cuda \
      ${EXTRA_STR} \
      2>&1 | tee -a 'logs/${RUN_NAME}.log';
    echo 'EXIT_CODE='\$? >> 'logs/${RUN_NAME}.log';
    exec bash
  "

  echo "==> tmux session '${SESSION}' started."
  echo "==> Log: ${VAST_REMOTE_DIR}/logs/${RUN_NAME}.log"
REMOTE

echo "==> Policy training launched. Monitor with: ./status.sh"
echo "==> Run name saved for status/fetch: ${RUN_NAME}"
echo "${RUN_NAME}" > .last-run
