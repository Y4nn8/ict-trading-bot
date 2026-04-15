"""Validate a trained Morpheus world model on held-out Parquet data.

Loads a checkpoint, samples starting points from a validation Parquet
directory, generates imagined trajectories of ``--horizon`` steps from
a ``--context-len`` context window, and reports statistics comparing
imagined to real trajectories.

Usage:
    uv run python -m scripts.validate_world_model \
        --checkpoint runs/morpheus_xauusd_001/checkpoint_final.pt \
        --parquet-dir data/morpheus/xauusd_val \
        --context-len 256 --horizon 64 --n-samples 256 \
        --output-csv runs/morpheus_xauusd_001/validation.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.common.logging import get_logger
from src.morpheus.dataset import (
    GAP_THRESHOLD_FACTOR,
    OBS_COLUMNS,
    NormStats,
    compute_observations,
    load_parquet_dir,
)
from src.morpheus.validation import ValidationReport, validate_trajectories
from src.morpheus.world_model import WorldModel

if TYPE_CHECKING:
    import polars as pl
    from numpy.typing import NDArray

logger = get_logger(__name__)


def _segments_with_time(
    df: pl.DataFrame, bucket_seconds: int, min_len: int,
) -> list[tuple[NDArray[np.float32], NDArray[np.int32]]]:
    """Return list of (obs_array, hours_array) for segments >= min_len.

    Same gap detection as :func:`find_segments`, but keeps the
    corresponding hour-of-day alongside each segment's observations.
    """
    if df.height == 0:
        return []

    threshold_us = bucket_seconds * GAP_THRESHOLD_FACTOR * 1_000_000
    time_diffs = df["time"].diff().dt.total_microseconds().fill_null(0).to_numpy()
    gap_indices = np.where(time_diffs > threshold_us)[0]
    split_points = [0, *gap_indices.tolist(), df.height]

    obs_data = df.select(OBS_COLUMNS).to_numpy().astype(np.float32)
    hours = df["time"].dt.hour().to_numpy().astype(np.int32)

    out: list[tuple[NDArray[np.float32], NDArray[np.int32]]] = []
    for i in range(len(split_points) - 1):
        start, end = split_points[i], split_points[i + 1]
        if end - start >= min_len:
            out.append((obs_data[start:end], hours[start:end]))
    return out


def sample_windows(
    parquet_dir: Path,
    *,
    context_len: int,
    horizon: int,
    bucket_seconds: int,
    n_samples: int,
    norm_stats: NormStats,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, NDArray[np.float32], NDArray[np.int32]]:
    """Sample ``n_samples`` (context, real_future, future_hours) tuples.

    Observations are z-score normalized using the checkpoint's stats.
    """
    df = load_parquet_dir(parquet_dir)
    obs_df = compute_observations(df)
    window_len = context_len + horizon
    segments = _segments_with_time(obs_df, bucket_seconds, window_len)
    if not segments:
        msg = (
            f"No segments >= {window_len} in {parquet_dir}. "
            "Reduce --context-len/--horizon or provide more data."
        )
        raise ValueError(msg)

    # Uniform sampling over all valid starting points without materializing
    # every (segment, offset) pair — O(n_segments) memory instead of O(total_rows).
    start_counts = np.fromiter(
        (len(obs) - window_len + 1 for obs, _ in segments),
        dtype=np.int64,
        count=len(segments),
    )
    total_starts = int(start_counts.sum())
    if total_starts <= 0:
        msg = "No valid starting points found"
        raise ValueError(msg)

    cumulative_starts = np.cumsum(start_counts)
    picks = rng.choice(
        total_starts, size=n_samples, replace=total_starts < n_samples,
    )

    context_batch = np.empty((n_samples, context_len, len(OBS_COLUMNS)), dtype=np.float32)
    future_batch = np.empty((n_samples, horizon, len(OBS_COLUMNS)), dtype=np.float32)
    hours_batch = np.empty((n_samples, horizon), dtype=np.int32)

    for i, pick in enumerate(picks):
        global_start = int(pick)
        seg_idx = int(np.searchsorted(cumulative_starts, global_start, side="right"))
        prev_cum = 0 if seg_idx == 0 else int(cumulative_starts[seg_idx - 1])
        offset = global_start - prev_cum
        obs, hours = segments[seg_idx]
        window = obs[offset : offset + window_len]
        context_batch[i] = window[:context_len]
        future_batch[i] = window[context_len:]
        hours_batch[i] = hours[offset + context_len : offset + window_len]

    context_norm = (context_batch - norm_stats.mean) / norm_stats.std
    future_norm = (future_batch - norm_stats.mean) / norm_stats.std
    return (
        torch.from_numpy(np.ascontiguousarray(context_norm)),
        future_norm,
        hours_batch,
    )


@torch.no_grad()
def imagine_batch(
    model: WorldModel,
    context: torch.Tensor,
    horizon: int,
    *,
    device: torch.device,
    batch_size: int = 32,
) -> NDArray[np.float32]:
    """Run ``model.imagine`` in mini-batches and return a numpy array."""
    model.eval()
    outputs: list[NDArray[np.float32]] = []
    for start in range(0, context.shape[0], batch_size):
        chunk = context[start : start + batch_size].to(device)
        pred = model.imagine(chunk, horizon=horizon)
        outputs.append(pred.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Validate a Morpheus world model")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--parquet-dir", type=Path, required=True)
    p.add_argument("--context-len", type=int, default=256)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--n-samples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-lag", type=int, default=20)
    p.add_argument("--bucket-seconds", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--output-csv", type=Path, default=None)
    # Which observation dimension to treat as "return" for KS/ACF metrics.
    # Default: ret_close (index 3 in OBS_COLUMNS).
    p.add_argument("--return-dim", type=int, default=OBS_COLUMNS.index("ret_close"))
    return p.parse_args(argv)


def resolve_device(choice: str) -> torch.device:
    """Resolve ``--device`` into a torch.device."""
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def write_report_csv(path: Path, report: ValidationReport) -> None:
    """Flatten the report to a single-row CSV for easy comparison across runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, float] = {
        "ks_statistic": report.ks_statistic,
        "ks_pvalue": report.ks_pvalue,
        "acf_max_abs_diff": report.acf_max_abs_diff,
        "acf_sq_max_abs_diff": report.acf_sq_max_abs_diff,
        "hourly_vol_max_abs_diff": report.hourly_vol_max_abs_diff,
    }
    for lag, (real, imag) in enumerate(zip(report.acf_real, report.acf_imag, strict=True), start=1):
        row[f"acf_real_lag_{lag}"] = float(real)
        row[f"acf_imag_lag_{lag}"] = float(imag)
    for lag, (real, imag) in enumerate(
        zip(report.acf_sq_real, report.acf_sq_imag, strict=True), start=1,
    ):
        row[f"acf_sq_real_lag_{lag}"] = float(real)
        row[f"acf_sq_imag_lag_{lag}"] = float(imag)
    for h in range(24):
        row[f"hv_real_h{h:02d}"] = float(report.hourly_vol_real[h])
        row[f"hv_imag_h{h:02d}"] = float(report.hourly_vol_imag[h])

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    device = resolve_device(args.device)
    rng = np.random.default_rng(args.seed)

    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = payload["config"]
    if payload.get("norm_stats") is None:
        msg = f"Checkpoint {args.checkpoint} has no norm_stats"
        raise ValueError(msg)
    norm_stats = NormStats.from_dict(payload["norm_stats"])

    model = WorldModel(
        obs_dim=cfg["obs_dim"],
        embed_dim=cfg["embed_dim"],
        det_dim=cfg["det_dim"],
        stoch_dim=cfg["stoch_dim"],
        hidden_dim=cfg["hidden_dim"],
        kl_weight=cfg["kl_weight"],
        free_nats=cfg["free_nats"],
    ).to(device)
    model.load_state_dict(payload["model_state"])

    context, real_future, hours = sample_windows(
        args.parquet_dir,
        context_len=args.context_len,
        horizon=args.horizon,
        bucket_seconds=args.bucket_seconds,
        n_samples=args.n_samples,
        norm_stats=norm_stats,
        rng=rng,
    )

    imag_future = imagine_batch(
        model, context, horizon=args.horizon,
        device=device, batch_size=args.batch_size,
    )

    real_returns = real_future[:, :, args.return_dim].astype(np.float64)
    imag_returns = imag_future[:, :, args.return_dim].astype(np.float64)

    report = validate_trajectories(
        real_returns, imag_returns, hours, max_lag=args.max_lag,
    )

    logger.info(
        "validation_report",
        ks_statistic=report.ks_statistic,
        ks_pvalue=report.ks_pvalue,
        acf_max_abs_diff=report.acf_max_abs_diff,
        acf_sq_max_abs_diff=report.acf_sq_max_abs_diff,
        hourly_vol_max_abs_diff=report.hourly_vol_max_abs_diff,
    )

    if args.output_csv is not None:
        write_report_csv(args.output_csv, report)
        logger.info("report_saved", path=str(args.output_csv))


if __name__ == "__main__":
    main(sys.argv[1:])
