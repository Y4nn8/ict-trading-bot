#!/usr/bin/env bash
# Destroy the server only — volume is preserved.
# Usage: ./teardown.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Destroying server (volume preserved)..."
terraform destroy \
  -target=hcloud_volume_attachment.data \
  -target=hcloud_server.bot \
  -auto-approve

echo "==> Server destroyed. Volume 'ict-bot-data' preserved."
echo "  To redeploy: ./scripts/deploy.sh"
echo "  To destroy everything (DANGER): terraform destroy"
