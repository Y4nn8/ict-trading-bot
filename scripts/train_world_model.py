"""Train the Morpheus world model on exported Parquet candles.

Loads a MorpheusDataset from a directory of monthly Parquet files,
chronologically splits it into train/val, and runs the full training
loop with checkpointing and CSV metrics logging.

Usage:
    uv run python -m scripts.train_world_model \
        --parquet-dir data/morpheus/xauusd \
        --output-dir runs/morpheus_xauusd_001 \
        --epochs 10 --batch-size 32 --seq-len 256

    # Resume from a previous checkpoint:
    uv run python -m scripts.train_world_model \
        --parquet-dir data/morpheus/xauusd \
        --output-dir runs/morpheus_xauusd_001 \
        --resume runs/morpheus_xauusd_001/checkpoint_epoch_5.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from src.common.logging import get_logger
from src.morpheus.dataset import MorpheusDataset
from src.morpheus.training import TrainConfig, chronological_split, train

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Train the Morpheus world model")

    # Data
    p.add_argument("--parquet-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--bucket-seconds", type=int, default=10)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--h1", action="store_true", help="Add H1 features")
    p.add_argument("--m5", action="store_true", help="Add M5 features")

    # Training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--val-gap", type=int, default=256)
    p.add_argument("--checkpoint-interval", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument(
        "--shuffle-split", action="store_true",
        help="Random split instead of chronological",
    )
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    p.add_argument("--amp", action="store_true", help="Enable mixed-precision (float16)")
    p.add_argument("--log-interval", type=int, default=10, help="Log every N train steps (0=off)")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (0=main thread)")

    # Model selection
    p.add_argument(
        "--model", choices=["rssm", "transformer"], default="rssm",
        help="World model architecture",
    )

    # RSSM-specific
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--det-dim", type=int, default=256)
    p.add_argument("--stoch-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--free-nats", type=float, default=1.0)

    # Transformer-specific
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-seq-len", type=int, default=1024)

    # Auxiliary directional head
    p.add_argument(
        "--aux-horizon", type=int, default=0,
        help="Directional prediction horizon (0=off)",
    )
    p.add_argument("--aux-weight", type=float, default=0.1, help="Weight for directional loss")

    return p.parse_args(argv)


def resolve_device(choice: str) -> torch.device:
    """Resolve ``--device`` into a torch.device."""
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    device = resolve_device(args.device)

    logger.info(
        "training_start",
        parquet_dir=str(args.parquet_dir),
        output_dir=str(args.output_dir),
        device=str(device),
        epochs=args.epochs,
    )

    dataset = MorpheusDataset(
        parquet_dir=args.parquet_dir,
        seq_len=args.seq_len,
        stride=args.stride,
        bucket_seconds=args.bucket_seconds,
        use_h1=args.h1,
        use_m5=args.m5,
    )
    logger.info(
        "dataset_loaded",
        total_sequences=len(dataset),
        obs_dim=dataset.obs_dim,
    )

    if args.shuffle_split:
        from torch.utils.data import random_split

        val_size = round(len(dataset) * args.val_fraction)
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
        logger.info(
            "dataset_split",
            mode="shuffle",
            train_size=len(train_set),
            val_size=len(val_set),
        )
    else:
        train_set, val_set = chronological_split(
            dataset, val_fraction=args.val_fraction, gap=args.val_gap,
        )
        logger.info(
            "dataset_split",
            mode="chronological",
            train_size=len(train_set),
            val_size=len(val_set),
            gap=args.val_gap,
        )

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_clip=args.grad_clip,
        val_fraction=args.val_fraction,
        val_gap=args.val_gap,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
        model_type=args.model,
        obs_dim=dataset.obs_dim,
        embed_dim=args.embed_dim,
        det_dim=args.det_dim,
        stoch_dim=args.stoch_dim,
        hidden_dim=args.hidden_dim,
        kl_weight=args.kl_weight,
        free_nats=args.free_nats,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        aux_horizon=args.aux_horizon,
        aux_weight=args.aux_weight,
        compile=args.compile,
        amp=args.amp,
        num_workers=args.num_workers,
    )

    history = train(
        train_set=train_set,
        val_set=val_set,
        norm_stats=dataset.norm_stats,
        config=config,
        output_dir=args.output_dir,
        device=device,
        resume_from=args.resume,
    )

    final = history[-1] if history else None
    if final is not None:
        logger.info(
            "training_complete",
            epochs=final.epoch,
            train_loss=final.train_loss,
            val_loss=final.val_loss,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
