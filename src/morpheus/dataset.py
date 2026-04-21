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

M5_OBS_COLUMNS = [
    "m5_ret_open",
    "m5_ret_high",
    "m5_ret_low",
    "m5_ret_close",
    "m5_tick_count",
    "m5_spread",
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

OBS_COLUMNS_M5 = BASE_OBS_COLUMNS + M5_OBS_COLUMNS
OBS_COLUMNS_M5_H1 = BASE_OBS_COLUMNS + M5_OBS_COLUMNS + H1_OBS_COLUMNS

CROSS_OBS_COLUMNS = [
    "eurusd_ret_open",
    "eurusd_ret_high",
    "eurusd_ret_low",
    "eurusd_ret_close",
    "eurusd_tick_count",
    "eurusd_spread",
    "usdjpy_ret_open",
    "usdjpy_ret_high",
    "usdjpy_ret_low",
    "usdjpy_ret_close",
    "usdjpy_tick_count",
    "usdjpy_spread",
]

DERIVED_OBS_COLUMNS = [
    "gold_vol_5",
    "gold_vol_20",
    "gold_vol_60",
    "gold_mom_5",
    "gold_mom_20",
    "gold_mom_60",
    "dxy_ret_close",
    "dxy_vol_20",
    "dxy_mom_20",
]

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


def compute_m5_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute M5-scale features and join back onto base rows.

    Same logic as compute_h1_features but with 5-minute aggregation.
    Features come from the last *completed* M5 candle (shifted by 5min).

    Args:
        df: DataFrame with columns [time, open, high, low, close,
            tick_count, spread] — must already be sorted by time.

    Returns:
        DataFrame with additional m5_* columns.
    """
    m5 = (
        df.group_by_dynamic("time", every="5m")
        .agg(
            pl.col("open").first().alias("m5_open"),
            pl.col("high").max().alias("m5_high"),
            pl.col("low").min().alias("m5_low"),
            pl.col("close").last().alias("m5_close"),
            pl.col("tick_count").sum().alias("m5_tick_count_raw"),
            pl.col("spread").mean().alias("m5_spread_raw"),
        )
        .sort("time")
    )

    m5_prev_close = m5["m5_close"].shift(1)
    m5 = m5.with_columns(
        ((pl.col("m5_open") - m5_prev_close) / m5_prev_close).alias(
            "m5_ret_open",
        ),
        ((pl.col("m5_high") - m5_prev_close) / m5_prev_close).alias(
            "m5_ret_high",
        ),
        ((pl.col("m5_low") - m5_prev_close) / m5_prev_close).alias(
            "m5_ret_low",
        ),
        ((pl.col("m5_close") - m5_prev_close) / m5_prev_close).alias(
            "m5_ret_close",
        ),
        pl.col("m5_tick_count_raw").alias("m5_tick_count"),
        pl.col("m5_spread_raw").alias("m5_spread"),
    ).select("time", *M5_OBS_COLUMNS)

    m5_shifted = m5.with_columns(
        (pl.col("time") + pl.duration(minutes=5)).alias("time"),
    )

    result = df.join_asof(
        m5_shifted, on="time", strategy="backward",
    )

    fill_cols = {c: pl.col(c).fill_null(0.0) for c in M5_OBS_COLUMNS}
    return result.with_columns(**fill_cols)


def compute_cross_features(
    df: pl.DataFrame,
    eurusd_dir: Path | None = None,
    usdjpy_dir: Path | None = None,
) -> pl.DataFrame:
    """Add cross-asset OHLC returns + tick_count + spread via time join.

    For each cross asset, computes 6 features: ret_open, ret_high,
    ret_low, ret_close, tick_count, spread. Joins backward on timestamps.

    Args:
        df: Main DataFrame with a 'time' column, sorted.
        eurusd_dir: Directory with EUR/USD M1 parquets.
        usdjpy_dir: Directory with USD/JPY M1 parquets.

    Returns:
        DataFrame with 12 additional cross-asset columns.
    """
    result = df
    for name, directory in [("eurusd", eurusd_dir), ("usdjpy", usdjpy_dir)]:
        feature_cols = [
            f"{name}_ret_open",
            f"{name}_ret_high",
            f"{name}_ret_low",
            f"{name}_ret_close",
            f"{name}_tick_count",
            f"{name}_spread",
        ]
        if directory is None:
            result = result.with_columns(
                [pl.lit(0.0).alias(c) for c in feature_cols],
            )
            continue

        cross = load_parquet_dir(directory)
        prev_close = cross["close"].shift(1)
        cross = cross.with_columns(
            ((pl.col("open") - prev_close) / prev_close).alias(feature_cols[0]),
            ((pl.col("high") - prev_close) / prev_close).alias(feature_cols[1]),
            ((pl.col("low") - prev_close) / prev_close).alias(feature_cols[2]),
            ((pl.col("close") - prev_close) / prev_close).alias(feature_cols[3]),
            pl.col("tick_count").cast(pl.Float64).alias(feature_cols[4]),
            pl.col("spread").alias(feature_cols[5]),
        ).select("time", *feature_cols).sort("time")

        result = result.join_asof(cross, on="time", strategy="backward")
        result = result.with_columns(
            [pl.col(c).fill_null(0.0) for c in feature_cols],
        )

    return result


def compute_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add derived features: rolling vol, momentum, partial USD index.

    Requires the main 'close' column and optionally eurusd/usdjpy
    columns (added by compute_cross_features).

    The "dxy_*" columns are a **partial** USD index — the real DXY has
    6 components (EUR 57.6%, JPY 13.6%, GBP 11.9%, CAD 9.1%, SEK 4.2%,
    CHF 3.6%). We only use EUR and JPY (~72% of the basket), so this is
    a USD-strength proxy, not the true DXY. Weight signs match ICE
    convention: stronger USD ⇒ EUR down (-) and USD/JPY up (+).

    Args:
        df: DataFrame with 'close' and optionally cross-asset columns.

    Returns:
        DataFrame with 9 additional derived columns.
    """
    prev_close = df["close"].shift(1)
    gold_ret = (pl.col("close") - prev_close) / prev_close

    has_cross = "eurusd_ret_close" in df.columns and "usdjpy_ret_close" in df.columns

    result = df.with_columns(
        # Rolling volatility (std of returns)
        gold_ret.rolling_std(window_size=5).fill_null(0.0).alias("gold_vol_5"),
        gold_ret.rolling_std(window_size=20).fill_null(0.0).alias("gold_vol_20"),
        gold_ret.rolling_std(window_size=60).fill_null(0.0).alias("gold_vol_60"),
        # Rolling momentum (sum of returns)
        gold_ret.rolling_sum(window_size=5).fill_null(0.0).alias("gold_mom_5"),
        gold_ret.rolling_sum(window_size=20).fill_null(0.0).alias("gold_mom_20"),
        gold_ret.rolling_sum(window_size=60).fill_null(0.0).alias("gold_mom_60"),
    )

    if has_cross:
        # DXY proxy: stronger USD = EUR down + JPY up (when quoted as USD/JPY)
        dxy_ret = (
            -0.58 * pl.col("eurusd_ret_close")
            + 0.14 * pl.col("usdjpy_ret_close")
        )
        result = result.with_columns(dxy_ret.alias("dxy_ret_close"))
        result = result.with_columns(
            pl.col("dxy_ret_close").rolling_std(window_size=20)
            .fill_null(0.0).alias("dxy_vol_20"),
            pl.col("dxy_ret_close").rolling_sum(window_size=20)
            .fill_null(0.0).alias("dxy_mom_20"),
        )
    else:
        result = result.with_columns(
            pl.lit(0.0).alias("dxy_ret_close"),
            pl.lit(0.0).alias("dxy_vol_20"),
            pl.lit(0.0).alias("dxy_mom_20"),
        )

    return result


def _compute_split_points(
    df: pl.DataFrame,
    bucket_seconds: int,
) -> list[int]:
    """Find gap-based split points in a time-sorted DataFrame."""
    times = df["time"]
    if len(times) == 0:
        return [0, 0]
    threshold_us = bucket_seconds * GAP_THRESHOLD_FACTOR * 1_000_000
    if times.dtype == pl.Datetime:
        time_diffs = times.diff().dt.total_microseconds().fill_null(0).to_numpy()
    else:
        time_diffs = np.full(len(times), bucket_seconds * 1_000_000, dtype=np.int64)
        time_diffs[0] = 0
    gap_indices = np.where(time_diffs > threshold_us)[0]
    return [0, *gap_indices.tolist(), len(times)]


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
    if len(df) == 0:
        return []

    obs_data = df.select(obs_columns).to_numpy().astype(np.float32)
    split_points = _compute_split_points(df, bucket_seconds)

    segments: list[NDArray[np.float32]] = []
    for i in range(len(split_points) - 1):
        start, end = split_points[i], split_points[i + 1]
        if end - start >= seq_len:
            segments.append(obs_data[start:end])

    return segments


def find_segments_with_close(
    df: pl.DataFrame,
    bucket_seconds: int,
    seq_len: int,
    obs_columns: list[str] | None = None,
) -> tuple[list[NDArray[np.float32]], list[NDArray[np.float32]]]:
    """Like find_segments, but also returns close prices per segment.

    Args:
        df: DataFrame with observation columns, 'time', and 'close'.
        bucket_seconds: Expected candle interval.
        seq_len: Minimum segment length.
        obs_columns: Column names to extract. Defaults to OBS_COLUMNS.

    Returns:
        (obs_segments, close_segments) where each close segment has
        shape (segment_len,).
    """
    if obs_columns is None:
        obs_columns = OBS_COLUMNS
    if len(df) == 0:
        return [], []

    obs_data = df.select(obs_columns).to_numpy().astype(np.float32)
    close_data = df["close"].to_numpy().astype(np.float32)
    split_points = _compute_split_points(df, bucket_seconds)

    obs_segs: list[NDArray[np.float32]] = []
    close_segs: list[NDArray[np.float32]] = []
    for i in range(len(split_points) - 1):
        start, end = split_points[i], split_points[i + 1]
        if end - start >= seq_len:
            obs_segs.append(obs_data[start:end])
            close_segs.append(close_data[start:end])

    return obs_segs, close_segs


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
        use_m5: bool = False,
        eurusd_dir: Path | None = None,
        usdjpy_dir: Path | None = None,
        use_derived: bool = False,
    ) -> None:
        df = load_parquet_dir(parquet_dir)
        if use_m5:
            df = compute_m5_features(df)
        if use_h1:
            df = compute_h1_features(df)

        use_cross = eurusd_dir is not None or usdjpy_dir is not None
        if use_cross:
            df = compute_cross_features(df, eurusd_dir, usdjpy_dir)

        if use_derived:
            df = compute_derived_features(df)

        obs_df = compute_observations(df)

        base_cols = BASE_OBS_COLUMNS.copy()
        if use_m5:
            base_cols = base_cols + M5_OBS_COLUMNS
        if use_h1:
            base_cols = base_cols + H1_OBS_COLUMNS
        if use_cross:
            base_cols = base_cols + CROSS_OBS_COLUMNS
        if use_derived:
            base_cols = base_cols + DERIVED_OBS_COLUMNS
        self._obs_columns = base_cols
        segments, close_segments = find_segments_with_close(
            obs_df, bucket_seconds, seq_len,
            obs_columns=self._obs_columns,
        )

        if not segments:
            msg = "No valid segments found (all segments shorter than seq_len)"
            raise ValueError(msg)

        self._index = build_segment_index(segments, seq_len, stride)
        self._close_segments = close_segments
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

    def get_close(self, idx: int) -> float:
        """Return close price of the last candle in sequence ``idx``."""
        if idx < 0:
            idx += len(self)
        seg_idx, offset = self._index.offsets[idx]
        return float(self._close_segments[seg_idx][offset + self._seq_len - 1])

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
