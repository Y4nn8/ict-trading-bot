# Vast.ai GPU Training — Morpheus

Manual-provisioned GPU instance flow for training the Morpheus world model.
The Vast.ai instance itself is created via the Vast.ai UI (no Terraform
provider); these scripts handle everything that happens after it's up.

## Prereqs

- Vast.ai account with credit (~5€ for one 8h RTX 5090 run).
- Your SSH pubkey registered on the Vast.ai account.
- `rsync`, `ssh`, `tmux` locally (you already have these).

## One-time per instance

1. In the Vast.ai UI, rent an instance:
   - **GPU**: RTX 5090 (Blackwell), 1x is enough for our model size
   - **Image**: a PyTorch image with CUDA 12.4+ (e.g. `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`)
   - **Disk**: 20–30 GB is plenty (dataset is 71 MB, checkpoints small)
2. Once the instance is running, copy the SSH host + port from the Vast.ai UI.
3. Fill in the config:
   ```bash
   cp infra/vast/.env.example infra/vast/.env
   # edit VAST_SSH_HOST and VAST_SSH_PORT
   ```
4. Make scripts executable once:
   ```bash
   chmod +x infra/vast/*.sh
   ```

## Flow

```bash
cd infra/vast

# 1. Push code + dataset + install deps + CUDA smoke-check
./provision.sh

# 2. Launch training in a tmux session (survives SSH disconnect).
#    Any args are forwarded to scripts/train_world_model.py.
./train.sh --epochs 20 --batch-size 64 --seq-len 256

# 3. Monitor (one-shot or --follow)
./status.sh
./status.sh --follow

# 4. Pull checkpoints + metrics CSV + log back locally
./fetch.sh

# 5. When done, kill the tmux job...
./stop.sh
# ...then destroy the instance in the Vast.ai UI to stop billing.
```

## Notes

- `train.sh` writes the run name to `infra/vast/.last-run`; `status.sh` and
  `fetch.sh` read it so you don't have to pass it each time.
- You can rerun `provision.sh` to push new code — rsync is incremental and
  the dataset transfer is a no-op if unchanged.
- Output layout on the instance:
  - Code: `/workspace/ict-trading-bot/`
  - Data: `/workspace/ict-trading-bot/data/morpheus/xauusd/`
  - Runs: `/workspace/ict-trading-bot/runs/morpheus/<run_name>/`
  - Logs: `/workspace/ict-trading-bot/logs/<run_name>.log`
- Locally, fetched runs land in `runs/morpheus/<run_name>/` and logs in
  `runs/morpheus/_logs/`.
