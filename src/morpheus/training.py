"""Training loop and checkpoint utilities for the Morpheus world model.

Separated from the CLI script (scripts/train_world_model.py) so the
core loop stays unit-testable.  Provides:

  - TrainConfig: dataclass of hyperparameters
  - chronological_split: purged train/val split on a MorpheusDataset
  - save_checkpoint / load_checkpoint
  - train_epoch / evaluate: per-epoch primitives
  - train: full training loop with checkpointing
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.amp import GradScaler, autocast  # type: ignore[attr-defined]
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from src.common.logging import get_logger
from src.morpheus.world_model import WorldModel

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from src.morpheus.dataset import MorpheusDataset, NormStats

logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for a training run."""

    epochs: int = 10
    batch_size: int = 32
    lr: float = 3e-4
    grad_clip: float = 1.0
    val_fraction: float = 0.1
    val_gap: int = 256
    log_interval: int = 50
    checkpoint_interval: int = 1
    seed: int = 0
    # Model hyperparameters propagated into the checkpoint for reproducibility.
    obs_dim: int = 10
    embed_dim: int = 64
    det_dim: int = 256
    stoch_dim: int = 64
    hidden_dim: int = 128
    kl_weight: float = 1.0
    free_nats: float = 1.0
    compile: bool = False
    amp: bool = False
    num_workers: int = 0


@dataclass
class EpochMetrics:
    """Per-epoch aggregated metrics."""

    epoch: int
    train_loss: float
    train_recon: float
    train_kl: float
    val_loss: float = field(default=float("nan"))
    val_recon: float = field(default=float("nan"))
    val_kl: float = field(default=float("nan"))


def chronological_split(
    dataset: MorpheusDataset,
    val_fraction: float,
    gap: int,
) -> tuple[Subset[torch.Tensor], Subset[torch.Tensor]]:
    """Split a dataset chronologically with a purge gap between train and val.

    Sequences in a MorpheusDataset are ordered by segment, and segments
    are time-sorted, so the index order is already chronological.  We
    insert a ``gap`` of dropped sequences between train and val to
    prevent a train sample from overlapping the first val sample.

    Args:
        dataset: The dataset to split.
        val_fraction: Fraction of the dataset reserved for validation.
        gap: Number of sequences to drop between train and val.

    Returns:
        (train_subset, val_subset).
    """
    if not 0.0 < val_fraction < 1.0:
        msg = f"val_fraction must be in (0, 1), got {val_fraction}"
        raise ValueError(msg)
    if gap < 0:
        msg = f"gap must be non-negative, got {gap}"
        raise ValueError(msg)

    n = len(dataset)
    val_size = round(n * val_fraction)
    train_end = n - val_size - gap
    if train_end <= 0 or val_size <= 0:
        msg = (
            f"Dataset too small for split: n={n}, val_fraction={val_fraction}, "
            f"gap={gap}"
        )
        raise ValueError(msg)

    train_indices = list(range(train_end))
    val_indices = list(range(train_end + gap, n))
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def save_checkpoint(
    path: Path,
    *,
    model: WorldModel,
    optimizer: Adam,
    epoch: int,
    config: TrainConfig,
    norm_stats: NormStats | None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a training checkpoint atomically.

    Writes to a temp file then renames so an interrupted write cannot
    leave a corrupt checkpoint on disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "config": asdict(config),
        "norm_stats": norm_stats.to_dict() if norm_stats is not None else None,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(
    path: Path,
    *,
    model: WorldModel,
    optimizer: Adam | None = None,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Restore model (and optionally optimizer) state from a checkpoint.

    Returns the raw payload so callers can read ``epoch``, ``config``,
    ``norm_stats`` etc. without re-reading the file.
    """
    payload: dict[str, Any] = torch.load(
        path, map_location=map_location, weights_only=False,
    )
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def _mean(values: Iterable[float]) -> float:
    vs = list(values)
    return sum(vs) / len(vs) if vs else float("nan")


def train_epoch(
    model: WorldModel,
    loader: DataLoader[torch.Tensor],
    optimizer: Adam,
    *,
    grad_clip: float,
    device: torch.device,
    scaler: GradScaler | None = None,
    use_amp: bool = False,
    epoch: int = 0,
    log_interval: int = 500,
) -> tuple[float, float, float]:
    """Run one training epoch and return (avg_loss, avg_recon, avg_kl)."""
    model.train()
    losses: list[float] = []
    recons: list[float] = []
    kls: list[float] = []
    total_steps = len(loader)
    t0 = time.monotonic()

    for step, batch in enumerate(loader):
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=use_amp):
            out = model(batch)

        if scaler is not None:
            scaler.scale(out.loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            out.loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        losses.append(out.loss.item())
        recons.append(out.recon_loss.item())
        kls.append(out.kl_loss.item())

        if log_interval > 0 and (step + 1) % log_interval == 0:
            elapsed = time.monotonic() - t0
            steps_per_sec = (step + 1) / elapsed
            eta_sec = (total_steps - step - 1) / steps_per_sec if steps_per_sec > 0 else 0
            logger.info(
                "train_step",
                epoch=epoch + 1,
                step=step + 1,
                total=total_steps,
                loss=f"{_mean(losses[-log_interval:]):.4f}",
                steps_per_sec=f"{steps_per_sec:.1f}",
                eta_min=f"{eta_sec / 60:.1f}",
            )

    return _mean(losses), _mean(recons), _mean(kls)


@torch.no_grad()
def evaluate(
    model: WorldModel,
    loader: DataLoader[torch.Tensor],
    *,
    device: torch.device,
    use_amp: bool = False,
) -> tuple[float, float, float]:
    """Run one evaluation pass and return (avg_loss, avg_recon, avg_kl)."""
    model.eval()
    losses: list[float] = []
    recons: list[float] = []
    kls: list[float] = []

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with autocast("cuda", enabled=use_amp):
            out = model(batch)
        losses.append(out.loss.item())
        recons.append(out.recon_loss.item())
        kls.append(out.kl_loss.item())

    return _mean(losses), _mean(recons), _mean(kls)


def append_metrics_csv(path: Path, metrics: EpochMetrics) -> None:
    """Append one epoch's metrics to a CSV (creating the header if needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epoch",
        "train_loss",
        "train_recon",
        "train_kl",
        "val_loss",
        "val_recon",
        "val_kl",
    ]
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(asdict(metrics))


def train(
    *,
    train_set: Subset[torch.Tensor] | MorpheusDataset,
    val_set: Subset[torch.Tensor] | MorpheusDataset | None,
    norm_stats: NormStats | None,
    config: TrainConfig,
    output_dir: Path,
    device: torch.device,
    resume_from: Path | None = None,
) -> list[EpochMetrics]:
    """Full training loop with checkpointing and CSV metrics logging.

    Args:
        train_set: Training dataset (or Subset thereof).
        val_set: Optional validation dataset.
        norm_stats: Normalization stats saved alongside checkpoints.
        config: Hyperparameters.
        output_dir: Directory for checkpoints and metrics.csv.
        device: Target device (cpu/cuda).
        resume_from: Optional path to a checkpoint to resume from.

    Returns:
        List of per-epoch metrics.
    """
    torch.manual_seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = WorldModel(
        obs_dim=config.obs_dim,
        embed_dim=config.embed_dim,
        det_dim=config.det_dim,
        stoch_dim=config.stoch_dim,
        hidden_dim=config.hidden_dim,
        kl_weight=config.kl_weight,
        free_nats=config.free_nats,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=config.lr)

    start_epoch = 0
    if resume_from is not None:
        payload = load_checkpoint(
            resume_from, model=model, optimizer=optimizer, map_location=device,
        )
        start_epoch = int(payload.get("epoch", 0))

    if config.compile and device.type == "cuda":
        logger.info("compiling_model")
        model = torch.compile(model)  # type: ignore[assignment]

    use_amp = config.amp and device.type == "cuda"
    scaler = GradScaler("cuda") if use_amp else None
    if use_amp:
        logger.info("amp_enabled")

    if len(train_set) < config.batch_size:
        msg = (
            f"Training set size ({len(train_set)}) is smaller than "
            f"batch_size ({config.batch_size}); DataLoader with drop_last=True "
            "would yield zero batches."
        )
        raise ValueError(msg)
    pin = device.type == "cuda" and config.num_workers > 0
    train_loader: DataLoader[torch.Tensor] = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=pin,
        persistent_workers=config.num_workers > 0,
    )
    val_loader: DataLoader[torch.Tensor] | None = None
    if val_set is not None and len(val_set) >= config.batch_size:
        val_loader = DataLoader(
            val_set,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.num_workers,
            pin_memory=pin,
            persistent_workers=config.num_workers > 0,
        )

    metrics_path = output_dir / "metrics.csv"
    history: list[EpochMetrics] = []

    for epoch in range(start_epoch, config.epochs):
        tr_loss, tr_recon, tr_kl = train_epoch(
            model, train_loader, optimizer,
            grad_clip=config.grad_clip, device=device,
            scaler=scaler, use_amp=use_amp,
            epoch=epoch, log_interval=config.log_interval,
        )
        val_loss = val_recon = val_kl = float("nan")
        if val_loader is not None:
            val_loss, val_recon, val_kl = evaluate(
                model, val_loader, device=device, use_amp=use_amp,
            )

        m = EpochMetrics(
            epoch=epoch + 1,
            train_loss=tr_loss, train_recon=tr_recon, train_kl=tr_kl,
            val_loss=val_loss, val_recon=val_recon, val_kl=val_kl,
        )
        history.append(m)
        append_metrics_csv(metrics_path, m)

        logger.info(
            "epoch_complete",
            epoch=epoch + 1,
            train_loss=f"{tr_loss:.4f}",
            train_recon=f"{tr_recon:.4f}",
            train_kl=f"{tr_kl:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_recon=f"{val_recon:.4f}",
            val_kl=f"{val_kl:.4f}",
        )

        if (epoch + 1) % config.checkpoint_interval == 0:
            save_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch + 1}.pt",
                model=model, optimizer=optimizer,
                epoch=epoch + 1, config=config, norm_stats=norm_stats,
            )

    save_checkpoint(
        output_dir / "checkpoint_final.pt",
        model=model, optimizer=optimizer,
        epoch=config.epochs, config=config, norm_stats=norm_stats,
    )
    return history
