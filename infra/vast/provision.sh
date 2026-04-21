#!/usr/bin/env bash
# Provision a fresh Vast.ai GPU instance for Morpheus training.
#
# Prereqs: you've created the instance in the Vast.ai UI (image:
# pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime or equivalent, RTX 5090),
# added your SSH key, and filled in infra/vast/.env.
#
# What it does:
#   1. Installs uv + tmux on the instance.
#   2. Rsyncs the repo (code + data/morpheus/xauusd).
#   3. Runs `uv sync` to build the env on the GPU box.
#   4. Smoke-checks torch.cuda.is_available().
#
# Usage: ./provision.sh
set -euo pipefail
cd "$(dirname "$0")"
source ./lib.sh

echo "==> Connecting to ${VAST_SSH_HOST}:${VAST_SSH_PORT}..."
vast_ssh true
echo "  OK"

echo "==> Installing uv + tmux on instance..."
vast_ssh bash <<'REMOTE'
  set -euo pipefail
  export DEBIAN_FRONTEND=noninteractive

  # uv
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  export PATH="$HOME/.local/bin:$PATH"
  uv --version

  # tmux (apt on most Vast.ai PyTorch images)
  if ! command -v tmux >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq tmux
  fi
  tmux -V

  # rsync (usually preinstalled, but just in case)
  command -v rsync >/dev/null 2>&1 || apt-get install -y -qq rsync
REMOTE

echo "==> Rsyncing repo code to ${VAST_REMOTE_DIR}..."
vast_ssh "mkdir -p '${VAST_REMOTE_DIR}'"
vast_rsync \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '.mypy_cache' \
  --exclude '.pytest_cache' \
  --exclude '.git' \
  --exclude 'runs' \
  --exclude 'results' \
  --exclude 'data' \
  --exclude 'config/*.csv' \
  --exclude 'config/*.bin' \
  "$REPO_ROOT/" "${VAST_SSH_USER}@${VAST_SSH_HOST}:${VAST_REMOTE_DIR}/"

echo "==> Rsyncing Morpheus dataset..."
vast_ssh "mkdir -p '${VAST_REMOTE_DIR}/data/morpheus'"
vast_rsync \
  "$REPO_ROOT/data/morpheus/xauusd/" \
  "${VAST_SSH_USER}@${VAST_SSH_HOST}:${VAST_REMOTE_DIR}/data/morpheus/xauusd/"

echo "==> Building Python env with uv sync..."
vast_ssh bash <<REMOTE
  set -euo pipefail
  export PATH="\$HOME/.local/bin:\$PATH"
  cd '${VAST_REMOTE_DIR}'
  uv sync --frozen || uv sync
REMOTE

echo "==> Checking CUDA availability..."
vast_ssh bash <<REMOTE
  set -euo pipefail
  export PATH="\$HOME/.local/bin:\$PATH"
  cd '${VAST_REMOTE_DIR}'
  uv run python -c "import torch; print('cuda:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
REMOTE

echo "==> Provisioning complete. Next: ./train.sh"
