#!/usr/bin/env bash
# Deploy: create server + attach volume, wait for cloud-init.
# Usage: ./deploy.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Terraform apply..."
terraform apply -auto-approve

IP=$(terraform output -raw server_ip)
echo "==> Server IP: $IP"

echo "==> Waiting for SSH..."
for i in $(seq 1 30); do
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@"$IP" true 2>/dev/null && break
  echo "  attempt $i/30..."
  sleep 10
done

echo "==> Waiting for cloud-init to finish..."
ssh -o StrictHostKeyChecking=no root@"$IP" \
  'while [ ! -f /mnt/data/.cloud-init-done ]; do echo "  waiting..."; sleep 15; done'

echo "==> Server ready: ssh root@$IP"
echo "==> Bot code: /mnt/data/bot"
echo "==> Next: run seed-db.sh (first time) or run-optuna.sh"
