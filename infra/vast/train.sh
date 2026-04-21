#!/usr/bin/env bash
# Launch Morpheus training inside a tmux session on the Vast.ai instance.
# The session survives SSH disconnection. Extra args are forwarded to
# scripts/train_world_model.py.
#
# Usage (typical 2-year run):
#   ./train.sh --epochs 20 --batch-size 64 --seq-len 256
#
# The --parquet-dir, --output-dir, --device flags are set here; everything
# else passes through.
set -euo pipefail
cd "$(dirname "$0")"
source ./lib.sh

RUN_NAME="${RUN_NAME:-xauusd_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="runs/morpheus/${RUN_NAME}"
PARQUET_DIR="${PARQUET_DIR:-data/morpheus/xauusd}"
SESSION="morpheus"

EXTRA_ARGS=("$@")

echo "==> Run name: ${RUN_NAME}"
echo "==> Output:   ${VAST_REMOTE_DIR}/${OUTPUT_DIR}"
echo "==> Extra:    ${EXTRA_ARGS[*]:-<none>}"

# Serialize args into a single string safe for the remote heredoc.
printf -v EXTRA_STR '%q ' "${EXTRA_ARGS[@]}"

vast_ssh bash <<REMOTE
  set -euo pipefail
  export PATH="\$HOME/.local/bin:\$PATH"
  cd '${VAST_REMOTE_DIR}'

  if tmux has-session -t '${SESSION}' 2>/dev/null; then
    echo "ERROR: tmux session '${SESSION}' already exists. Attach with: tmux attach -t ${SESSION}" >&2
    exit 1
  fi

  mkdir -p '${OUTPUT_DIR}' logs

  tmux new-session -d -s '${SESSION}' "
    export PATH=\"\$HOME/.local/bin:\$PATH\";
    cd '${VAST_REMOTE_DIR}';
    uv run python -m scripts.train_world_model \
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

echo "==> Training launched. Monitor with: ./status.sh"
echo "==> Run name saved for fetch: ${RUN_NAME}"
echo "${RUN_NAME}" > .last-run
