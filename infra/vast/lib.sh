#!/usr/bin/env bash
# Shared helpers for Vast.ai scripts. Source this file from every script.
set -euo pipefail

# Locate repo root (two levels up from this file).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="$REPO_ROOT/infra/vast/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE not found. Copy .env.example to .env and fill in the Vast.ai SSH details." >&2
  exit 1
fi

# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a

: "${VAST_SSH_HOST:?VAST_SSH_HOST must be set in .env}"
: "${VAST_SSH_PORT:?VAST_SSH_PORT must be set in .env}"
: "${VAST_SSH_USER:=root}"
: "${VAST_REMOTE_DIR:=/workspace/ict-trading-bot}"

SSH_OPTS=(
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
  -p "$VAST_SSH_PORT"
)

vast_ssh() {
  ssh "${SSH_OPTS[@]}" "${VAST_SSH_USER}@${VAST_SSH_HOST}" "$@"
}

vast_rsync() {
  rsync -az --delete \
    -e "ssh ${SSH_OPTS[*]}" \
    "$@"
}
