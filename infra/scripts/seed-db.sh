#!/usr/bin/env bash
# Seed the VPS database from a local pg_dump.
# First time only — data persists on the volume.
#
# Usage: ./seed-db.sh [dump_file]
#   If no dump_file, creates one from the local DB first.
set -euo pipefail
cd "$(dirname "$0")/.."

IP=$(terraform output -raw server_ip)
DUMP="${1:-/tmp/ict_bot_db.dump}"

# Step 1: create local dump if not provided
if [ ! -f "$DUMP" ]; then
  echo "==> Creating local pg_dump (custom format, compressed)..."
  PGPASSWORD=trader_secret pg_dump \
    -h localhost -U trader -d trading_bot \
    -Fc -Z 4 \
    --no-owner --no-acl \
    -f "$DUMP"
  echo "  Dump: $(du -h "$DUMP" | cut -f1)"
fi

# Step 2: upload dump
echo "==> Uploading dump to VPS..."
scp -o StrictHostKeyChecking=no "$DUMP" root@"$IP":/tmp/ict_bot_db.dump

# Step 3: restore on VPS
echo "==> Restoring on VPS (this takes a while)..."
ssh -o StrictHostKeyChecking=no root@"$IP" bash <<'REMOTE'
  set -euo pipefail

  # Wait for PG to be ready
  for i in $(seq 1 30); do
    docker exec timescaledb pg_isready -U trader -d trading_bot && break
    echo "  waiting for PG... ($i/30)"
    sleep 5
  done

  # Drop existing tables and restore
  docker exec -i timescaledb pg_restore \
    -U trader -d trading_bot \
    --clean --if-exists --no-owner --no-acl \
    < /tmp/ict_bot_db.dump || true

  # Verify
  docker exec timescaledb psql -U trader -d trading_bot \
    -c "SELECT COUNT(*) AS ticks FROM ticks"

  rm /tmp/ict_bot_db.dump
  echo "==> DB seeded successfully"
REMOTE
