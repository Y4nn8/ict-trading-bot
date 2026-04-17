"""PyTorch Dataset for Morpheus world model training.

Reads Parquet candle files, computes the 10-dim observation vector
(relative returns + tick_count + spread + time encodings), splits
into contiguous segments (no gaps across weekends/holidays), and
serves fixed-length sequences with optional z-score normalization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

BASE_OBS_COLUMNS = [
    "ret_open",
    "ret_high",
    "ret_low",
    "ret_close",
    "tick_count",
    "spread",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

H1_OBS_COLUMNS = [
    "h1_ret_open",
    "h1_ret_high",
    "h1_ret_low",
    "h1_ret_close",
    "h1_tick_count",
    "h1_spread",
]

OBS_COLUMNS = BASE_OBS_COLUMNS
OBS_DIM = len(OBS_COLUMNS)

OBS_COLUMNS_H1 = BASE_OBS_COLUMNS + H1_OBS_COLUMNS
OBS_DIM_H1 = len(OBS_COLUMNS_H1)

GAP_THRESHOLD_FACTOR = 6


@dataclass(frozen=True)
class NormStats:
    """Per-feature mean and std for z-score normalization."""

    mean: NDArray[np.float32]
    std: NDArray[np.float32]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for checkpoint saving."""
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NormStats:
        """Deserialize from dict."""
        return cls(
            mean=np.array(d["mean"], dtype=np.float32),
            std=np.array(d["std"], dtype=np.float32),
        )


@dataclass
class SegmentIndex:
    """Index of contiguous segments and their sequence offsets."""

    segments: list[NDArray[np.float32]] = field(default_factory=list)
    offsets: list[tuple[int, int]] = field(default_factory=list)

    @property
    def total_sequences(self) -> int:
        """Total number of sequences across all segments."""
        return len(self.offsets)


def load_parquet_dir(parquet_dir: Path) -> pl.DataFrame:
    """Load and concatenate all Parquet files from a directory, sorted by time."""
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        msg = f"No .parquet files found in {parquet_dir}"
        raise FileNotFoundError(msg)
    dfs = [
        pl.read_parquet(f).cast({"tick_count": pl.Int64}, strict=False)
        for f in files
    ]
    return pl.concat(dfs).sort("time")


def compute_observations(df: pl.DataFrame) -> pl.DataFrame:
    """Compute the 10-dim observation vector from raw candle data.

    Args:
        df: DataFrame with columns [time, open, high, low, close, tick_count, spread].

    Returns:
        DataFrame with original columns plus observation columns.
        The first row is dropped (no previous close for returns).
    """
    prev_close = df["close"].shift(1)

    time_col = df["time"]
    if time_col.dtype == pl.Datetime:
        hour_rad = (
            (
                pl.col("time").dt.hour().cast(pl.Float64)
                + pl.col("time").dt.minute().cast(pl.Float64) / 60.0
            )
            * (2.0 * math.pi / 24.0)
        )
        dow_rad = pl.col("time").dt.weekday().cast(pl.Float64) * (
            2.0 * math.pi / 7.0
        )
    else:
        hour_rad = pl.lit(0.0)
        dow_rad = pl.lit(0.0)

    return df.with_columns(
        ((pl.col("open") - prev_close) / prev_close).alias("ret_open"),
        ((pl.col("high") - prev_close) / prev_close).alias("ret_high"),
        ((pl.col("low") - prev_close) / prev_close).alias("ret_low"),
        ((pl.col("close") - prev_close) / prev_close).alias("ret_close"),
        hour_rad.sin().alias("hour_sin"),
        hour_rad.cos().alias("hour_cos"),
        dow_rad.sin().alias("dow_sin"),
        dow_rad.cos().alias("dow_cos"),
    ).slice(1)


def compute_h1_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute H1-scale features and join back onto 10s rows.

    For each 10s candle, adds features from the last *completed* H1 candle
    (shifted by one hour to avoid look-ahead).  H1 candles are aggregated
    from the raw 10s data using floor-to-hour bucketing.

    Args:
        df: DataFrame with columns [time, open, high, low, close,
            tick_count, spread] — must already be sorted by time.

    Returns:
        DataFrame with additional h1_* columns.  Rows where no prior
        completed H1 candle exists are filled with 0.0.
    """
    h1 = (
        df.group_by_dynamic("time", every="1h")
        .agg(
            pl.col("open").first().alias("h1_open"),
            pl.col("high").max().alias("h1_high"),
            pl.col("low").min().alias("h1_low"),
            pl.col("close").last().alias("h1_close"),
            pl.col("tick_count").sum().alias("h1_tick_count_raw"),
            pl.col("spread").mean().alias("h1_spread_raw"),
        )
        .sort("time")
    )

    h1_prev_close = h1["h1_close"].shift(1)
    h1 = h1.with_columns(
        ((pl.col("h1_open") - h1_prev_close) / h1_prev_close).alias(
            "h1_ret_open"
        ),
        ((pl.col("h1_high") - h1_prev_close) / h1_prev_close).alias(
            "h1_ret_high"
        ),
        ((pl.col("h1_low") - h1_prev_close) / h1_prev_close).alias(
            "h1_ret_low"
        ),
        ((pl.col("h1_close") - h1_prev_close) / h1_prev_close).alias(
            "h1_ret_close"
        ),
        pl.col("h1_tick_count_raw").alias("h1_tick_count"),
        pl.col("h1_spread_raw").alias("h1_spread"),
    ).select("time", *H1_OBS_COLUMNS)

    h1_shifted = h1.with_columns(
        (pl.col("time") + pl.duration(hours=1)).alias("time"),
    )

    result = df.join_asof(
        h1_shifted, on="time", strategy="backward",
    )

    fill_cols = {c: pl.col(c).fill_null(0.0) for c in H1_OBS_COLUMNS}
    return result.with_columns(**fill_cols)


def find_segments(
    df: pl.DataFrame,
    bucket_seconds: int,
    seq_len: int,
    obs_columns: list[str] | None = None,
) -> list[NDArray[np.float32]]:
    """Split observations into contiguous segments (no time gaps).

    A gap is detected when the time difference between consecutive
    candles exceeds bucket_seconds * GAP_THRESHOLD_FACTOR.

    Args:
        df: DataFrame with observation columns and 'time'.
        bucket_seconds: Expected candle interval.
        seq_len: Minimum segment length (shorter segments are dropped).
        obs_columns: Column names to extract. Defaults to OBS_COLUMNS.

    Returns:
        List of numpy arrays, each of shape (segment_len, obs_dim).
    """
    if obs_columns is None:
        obs_columns = OBS_COLUMNS

    times = df["time"]
    if len(times) == 0:
        return []

    obs_data = df.select(obs_columns).to_numpy().astype(np.float32)
    threshold_us = bucket_seconds * GAP_THRESHOLD_FACTOR * 1_000_000

    if times.dtype == pl.Datetime:
        time_diffs = times.diff().dt.total_microseconds().fill_null(0).to_numpy()
    else:
        time_diffs = np.full(len(times), bucket_seconds * 1_000_000, dtype=np.int64)
        time_diffs[0] = 0

    gap_indices = np.where(time_diffs > threshold_us)[0]
    split_points = [0, *gap_indices.tolist(), len(obs_data)]

    segments: list[NDArray[np.float32]] = []
    for i in range(len(split_points) - 1):
        start, end = split_points[i], split_points[i + 1]
        if end - start >= seq_len:
            segments.append(obs_data[start:end])

    return segments


def build_segment_index(
    segments: list[NDArray[np.float32]],
    seq_len: int,
    stride: int = 1,
) -> SegmentIndex:
    """Build an index mapping global sequence index to (segment, offset).

    Args:
        segments: List of contiguous observation arrays.
        seq_len: Sequence length for each sample.
        stride: Step between consecutive sequence starts.

    Returns:
        SegmentIndex with precomputed offsets.
    """
    if seq_len <= 0:
        msg = f"seq_len must be positive, got {seq_len}"
        raise ValueError(msg)
    if stride <= 0:
        msg = f"stride must be positive, got {stride}"
        raise ValueError(msg)
    index = SegmentIndex(segments=segments)
    for seg_idx, seg in enumerate(segments):
        n_seqs = (len(seg) - seq_len) // stride + 1
        for s in range(n_seqs):
            index.offsets.append((seg_idx, s * stride))
    return index


def compute_norm_stats(segments: list[NDArray[np.float32]]) -> NormStats:
    """Compute per-feature mean and std across all segments."""
    if not segments:
        msg = "Cannot compute normalization stats from empty segment list"
        raise ValueError(msg)
    all_data = np.concatenate(segments, axis=0)
    mean = all_data.mean(axis=0).astype(np.float32)
    std = all_data.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return NormStats(mean=mean, std=std)


class MorpheusDataset(Dataset[torch.Tensor]):
    """PyTorch Dataset serving fixed-length observation sequences.

    Args:
        parquet_dir: Directory containing monthly Parquet files.
        seq_len: Number of candles per sequence.
        stride: Step between consecutive sequence starts.
        bucket_seconds: Candle duration (for gap detection).
        norm_stats: Pre-computed normalization stats. If None, computed from data.
        normalize: Whether to apply z-score normalization.
    """

    def __init__(
        self,
        parquet_dir: Path,
        seq_len: int = 256,
        stride: int = 1,
        bucket_seconds: int = 10,
        norm_stats: NormStats | None = None,
        normalize: bool = True,
        use_h1: bool = False,
    ) -> None:
        df = load_parquet_dir(parquet_dir)
        if use_h1:
            df = compute_h1_features(df)
        obs_df = compute_observations(df)
        self._obs_columns = OBS_COLUMNS_H1 if use_h1 else OBS_COLUMNS
        segments = find_segments(
            obs_df, bucket_seconds, seq_len,
            obs_columns=self._obs_columns,
        )

        if not segments:
            msg = "No valid segments found (all segments shorter than seq_len)"
            raise ValueError(msg)

        self._index = build_segment_index(segments, seq_len, stride)
        self._seq_len = seq_len
        self._normalize = normalize
        self._stats: NormStats | None = None

        if normalize:
            self._stats = norm_stats or compute_norm_stats(segments)

    @property
    def norm_stats(self) -> NormStats | None:
        """Return normalization stats (for saving to checkpoint)."""
        return self._stats

    @property
    def obs_dim(self) -> int:
        """Observation dimensionality."""
        return len(self._obs_columns)

    def __len__(self) -> int:
        return self._index.total_sequences

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return observation sequence of shape (seq_len, OBS_DIM)."""
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            msg = f"Index {idx} out of range [0, {len(self)})"
            raise IndexError(msg)

        seg_idx, offset = self._index.offsets[idx]
        seq = self._index.segments[seg_idx][offset : offset + self._seq_len]

        if self._normalize and self._stats is not None:
            seq = (seq - self._stats.mean) / self._stats.std

        return torch.from_numpy(np.ascontiguousarray(seq))
