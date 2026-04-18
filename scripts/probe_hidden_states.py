"""Probe transformer hidden states for directional signal.

Loads a world model checkpoint, runs backbone on real data, and trains
a linear classifier to predict future return direction from hidden
states.  If accuracy ~50%, the hidden states carry no exploitable
directional information.

Usage:
    uv run python -m scripts.probe_hidden_states \
        --checkpoint runs/morpheus/checkpoint_epoch_18.pt \
        --parquet-dir data/morpheus/xauusd_recent \
        --h1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from src.common.logging import get_logger
from src.morpheus.dataset import MorpheusDataset, NormStats
from src.morpheus.training import TrainConfig, build_model

logger = get_logger(__name__)

RET_CLOSE_IDX = 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Probe hidden states for direction")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--parquet-dir", type=Path, required=True)
    p.add_argument("--h1", action="store_true")
    p.add_argument("--m5", action="store_true")
    p.add_argument("--eurusd-dir", type=Path, default=None)
    p.add_argument("--usdjpy-dir", type=Path, default=None)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--bucket-seconds", type=int, default=10)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--horizons", type=str, default="1,5,10,20,50")
    p.add_argument("--probe-epochs", type=int, default=50)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def resolve_device(choice: str) -> torch.device:
    """Resolve --device into a torch.device."""
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    horizons = [int(h) for h in args.horizons.split(",")]
    max_horizon = max(horizons)

    # Load world model
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    norm_stats = NormStats.from_dict(payload["norm_stats"])
    wm_cfg = payload["config"]
    train_config = TrainConfig(**{
        k: v for k, v in wm_cfg.items()
        if k in TrainConfig.__dataclass_fields__
    })
    world_model = build_model(train_config).to(device)
    world_model.load_state_dict(payload["model_state"])
    world_model.eval()
    for p in world_model.parameters():
        p.requires_grad_(False)

    d_model = train_config.d_model
    logger.info("model_loaded", d_model=d_model)

    # Load dataset with longer sequences to have future returns
    total_len = args.seq_len + max_horizon
    dataset = MorpheusDataset(
        parquet_dir=args.parquet_dir,
        seq_len=total_len,
        stride=args.stride,
        bucket_seconds=args.bucket_seconds,
        norm_stats=norm_stats,
        use_h1=args.h1,
        use_m5=args.m5,
        eurusd_dir=args.eurusd_dir,
        usdjpy_dir=args.usdjpy_dir,
    )
    n_samples = min(args.n_samples, len(dataset))
    logger.info("dataset_loaded", total_sequences=len(dataset), using=n_samples)

    # Extract hidden states and future returns
    indices = torch.randperm(len(dataset))[:n_samples]
    all_hidden = []
    all_future_rets: dict[int, list[float]] = {h: [] for h in horizons}

    batch_size = 64
    for start in range(0, n_samples, batch_size):
        batch_idx = indices[start : start + batch_size]
        seqs = torch.stack([dataset[int(i)] for i in batch_idx]).to(device)

        context = seqs[:, : args.seq_len]
        future = seqs[:, args.seq_len :]

        with torch.no_grad():
            hidden = world_model._run_backbone(context)  # type: ignore[operator]
            h_last = hidden[:, -1]  # (batch, d_model)

        all_hidden.append(h_last.cpu())

        # Future returns: denormalize ret_close and sum over horizon
        future_ret_close = future[:, :, RET_CLOSE_IDX].cpu().numpy()
        ret_std = float(norm_stats.std[RET_CLOSE_IDX])
        ret_mean = float(norm_stats.mean[RET_CLOSE_IDX])
        raw_rets = future_ret_close * ret_std + ret_mean

        for h in horizons:
            cum_ret = raw_rets[:, :h].sum(axis=1)
            all_future_rets[h].extend(cum_ret.tolist())

    hidden_all = torch.cat(all_hidden, dim=0)  # (n_samples, d_model)
    logger.info("hidden_states_extracted", shape=list(hidden_all.shape))

    # Train linear probes for each horizon
    for h in horizons:
        future_ret = np.array(all_future_rets[h])
        labels = (future_ret > 0).astype(np.float32)
        balance = labels.mean()

        # Split 80/20
        n_train = int(0.8 * n_samples)
        x_train = hidden_all[:n_train]
        y_train = torch.from_numpy(labels[:n_train])
        x_test = hidden_all[n_train:]
        y_test = torch.from_numpy(labels[n_train:])

        # Linear probe
        probe = nn.Linear(d_model, 1)
        opt = Adam(probe.parameters(), lr=1e-3)

        for _epoch in range(args.probe_epochs):
            logits = probe(x_train).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y_train)
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()

        # Evaluate
        with torch.no_grad():
            test_logits = probe(x_test).squeeze(-1)
            test_preds = (test_logits > 0).float()
            accuracy = (test_preds == y_test).float().mean().item()

        logger.info(
            "probe_result",
            horizon=h,
            accuracy=f"{accuracy:.4f}",
            baseline=f"{max(balance, 1 - balance):.4f}",
            n_train=n_train,
            n_test=n_samples - n_train,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
