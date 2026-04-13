#cloud-config

package_update: true
packages:
  - docker.io
  - docker-compose-v2
  - git
  - curl
  - pigz
  - libgomp1

runcmd:
  # --- Mount persistent volume ---
  - |
    DEVICE="/dev/disk/by-id/scsi-0HC_Volume_${volume_id}"
    MOUNT="/mnt/data"
    mkdir -p "$MOUNT"
    # Format only if not already formatted
    if ! blkid "$DEVICE" | grep -q ext4; then
      mkfs.ext4 -L ict-bot-data "$DEVICE"
    fi
    mount "$DEVICE" "$MOUNT"
    echo "$DEVICE $MOUNT ext4 defaults 0 2" >> /etc/fstab
    mkdir -p "$MOUNT"/{pgdata,bot,results,logs}

  # --- Install uv + Python 3.12 ---
  - curl -LsSf https://astral.sh/uv/install.sh | sh
  - export PATH="/root/.local/bin:$PATH"

  # --- Start Docker ---
  - systemctl enable docker
  - systemctl start docker

  # --- Clone or update repo ---
  - |
    REPO="/mnt/data/bot"
    if [ ! -d "$REPO/.git" ]; then
      git clone https://github.com/Y4nn8/ict-trading-bot.git "$REPO"
    fi
    cd "$REPO"
    git fetch origin
    git checkout ${git_branch}
    git pull origin ${git_branch} || true

  # --- Create .env ---
  - |
    cat > /mnt/data/bot/.env <<'DOTENV'
    DATABASE_URL=postgresql://trader:trader_secret@localhost:5432/trading_bot
    DOTENV

  # --- Start PostgreSQL/TimescaleDB with data on volume ---
  - |
    docker run -d \
      --name timescaledb \
      --restart unless-stopped \
      -e POSTGRES_DB=trading_bot \
      -e POSTGRES_USER=trader \
      -e POSTGRES_PASSWORD=trader_secret \
      -v /mnt/data/pgdata:/var/lib/postgresql/data \
      -v /mnt/data/bot/scripts/init_db.sql:/docker-entrypoint-initdb.d/01_init.sql \
      -p 5432:5432 \
      --health-cmd "pg_isready -U trader -d trading_bot" \
      --health-interval 10s \
      --health-timeout 5s \
      --health-retries 10 \
      timescale/timescaledb:latest-pg16

  # --- Install Python deps ---
  - |
    export PATH="/root/.local/bin:$PATH"
    cd /mnt/data/bot
    uv sync

  # --- Signal ready ---
  - touch /mnt/data/.cloud-init-done
