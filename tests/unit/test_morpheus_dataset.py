"""Tests for Morpheus dataset and export helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
import torch

from scripts.export_candles_parquet import month_ranges
from src.morpheus.dataset import (
    OBS_COLUMNS,
    OBS_DIM,
    MorpheusDataset,
    NormStats,
    build_segment_index,
    compute_norm_stats,
    compute_observations,
    find_segments,
    load_parquet_dir,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_candle_df(
    n: int, start: datetime | None = None, gap_at: int | None = None,
) -> pl.DataFrame:
    """Create a synthetic candle DataFrame with n rows."""
    if start is None:
        start = datetime(2025, 1, 6, 8, 0, 0, tzinfo=UTC)  # Monday 08:00 UTC

    times = []
    t = start
    for i in range(n):
        if gap_at is not None and i == gap_at:
            t += timedelta(hours=48)  # weekend gap
        times.append(t)
        t += timedelta(seconds=10)

    base_price = 2000.0
    rng = np.random.default_rng(42)
    closes = base_price + np.cumsum(rng.normal(0, 0.5, n))

    return pl.DataFrame({
        "time": times,
        "open": closes + rng.normal(0, 0.1, n),
        "high": closes + np.abs(rng.normal(0, 0.3, n)),
        "low": closes - np.abs(rng.normal(0, 0.3, n)),
        "close": closes.tolist(),
        "tick_count": rng.integers(5, 50, n).tolist(),
        "spread": (0.3 + rng.normal(0, 0.05, n)).tolist(),
    })


@pytest.fixture
def candle_df() -> pl.DataFrame:
    return _make_candle_df(100)


@pytest.fixture
def candle_df_with_gap() -> pl.DataFrame:
    return _make_candle_df(100, gap_at=50)


@pytest.fixture
def parquet_dir(tmp_path: Path) -> Path:
    d = tmp_path / "candles"
    d.mkdir()
    df1 = _make_candle_df(60, start=datetime(2025, 1, 6, 8, 0, 0, tzinfo=UTC))
    df2 = _make_candle_df(60, start=datetime(2025, 2, 3, 8, 0, 0, tzinfo=UTC))
    df1.write_parquet(d / "2025-01.parquet")
    df2.write_parquet(d / "2025-02.parquet")
    return d


# ---------------------------------------------------------------------------
# month_ranges tests
# ---------------------------------------------------------------------------

class TestMonthRanges:
    def test_single_month(self) -> None:
        start = datetime(2025, 3, 1, tzinfo=UTC)
        end = datetime(2025, 3, 15, tzinfo=UTC)
        ranges = month_ranges(start, end)
        assert len(ranges) == 1
        assert ranges[0][2] == "2025-03"
        assert ranges[0][0] == start
        assert ranges[0][1] == end

    def test_multi_month(self) -> None:
        start = datetime(2025, 1, 15, tzinfo=UTC)
        end = datetime(2025, 4, 10, tzinfo=UTC)
        ranges = month_ranges(start, end)
        assert len(ranges) == 4
        labels = [r[2] for r in ranges]
        assert labels == ["2025-01", "2025-02", "2025-03", "2025-04"]
        assert ranges[0][0] == start
        assert ranges[-1][1] == end

    def test_year_boundary(self) -> None:
        start = datetime(2024, 11, 1, tzinfo=UTC)
        end = datetime(2025, 2, 1, tzinfo=UTC)
        ranges = month_ranges(start, end)
        labels = [r[2] for r in ranges]
        assert labels == ["2024-11", "2024-12", "2025-01"]

    def test_empty_range(self) -> None:
        start = datetime(2025, 3, 1, tzinfo=UTC)
        assert month_ranges(start, start) == []


# ---------------------------------------------------------------------------
# compute_observations tests
# ---------------------------------------------------------------------------

class TestComputeObservations:
    def test_shape(self, candle_df: pl.DataFrame) -> None:
        obs = compute_observations(candle_df)
        assert len(obs) == len(candle_df) - 1  # first row dropped
        for col in OBS_COLUMNS:
            assert col in obs.columns

    def test_returns_are_relative(self, candle_df: pl.DataFrame) -> None:
        obs = compute_observations(candle_df)
        ret_close = obs["ret_close"].to_numpy()
        closes = candle_df["close"].to_numpy()
        expected = (closes[1:] - closes[:-1]) / closes[:-1]
        np.testing.assert_allclose(ret_close, expected, rtol=1e-6)

    def test_time_encodings_bounded(self, candle_df: pl.DataFrame) -> None:
        obs = compute_observations(candle_df)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            vals = obs[col].to_numpy()
            assert np.all(vals >= -1.0 - 1e-9)
            assert np.all(vals <= 1.0 + 1e-9)

    def test_no_nans(self, candle_df: pl.DataFrame) -> None:
        obs = compute_observations(candle_df)
        data = obs.select(OBS_COLUMNS).to_numpy()
        assert not np.any(np.isnan(data))


# ---------------------------------------------------------------------------
# find_segments tests
# ---------------------------------------------------------------------------

class TestFindSegments:
    def test_no_gap_single_segment(self, candle_df: pl.DataFrame) -> None:
        obs = compute_observations(candle_df)
        segments = find_segments(obs, bucket_seconds=10, seq_len=10)
        assert len(segments) == 1
        assert segments[0].shape == (len(obs), OBS_DIM)

    def test_gap_splits_segments(self, candle_df_with_gap: pl.DataFrame) -> None:
        obs = compute_observations(candle_df_with_gap)
        segments = find_segments(obs, bucket_seconds=10, seq_len=10)
        assert len(segments) == 2

    def test_short_segments_dropped(self) -> None:
        df = _make_candle_df(20, gap_at=10)
        obs = compute_observations(df)
        segments = find_segments(obs, bucket_seconds=10, seq_len=15)
        assert len(segments) == 0

    def test_segment_dtype(self, candle_df: pl.DataFrame) -> None:
        obs = compute_observations(candle_df)
        segments = find_segments(obs, bucket_seconds=10, seq_len=10)
        assert segments[0].dtype == np.float32


# ---------------------------------------------------------------------------
# build_segment_index tests
# ---------------------------------------------------------------------------

class TestBuildSegmentIndex:
    def test_stride_1(self) -> None:
        seg = np.zeros((50, OBS_DIM), dtype=np.float32)
        index = build_segment_index([seg], seq_len=10, stride=1)
        assert index.total_sequences == 41  # 50 - 10 + 1

    def test_stride_equals_seq_len(self) -> None:
        seg = np.zeros((100, OBS_DIM), dtype=np.float32)
        index = build_segment_index([seg], seq_len=10, stride=10)
        assert index.total_sequences == 10  # 100 / 10

    def test_multiple_segments(self) -> None:
        seg1 = np.zeros((30, OBS_DIM), dtype=np.float32)
        seg2 = np.zeros((20, OBS_DIM), dtype=np.float32)
        index = build_segment_index([seg1, seg2], seq_len=10, stride=1)
        expected = (30 - 10 + 1) + (20 - 10 + 1)
        assert index.total_sequences == expected

    def test_offsets_correct(self) -> None:
        seg = np.zeros((15, OBS_DIM), dtype=np.float32)
        index = build_segment_index([seg], seq_len=10, stride=5)
        assert index.offsets == [(0, 0), (0, 5)]

    def test_invalid_stride_raises(self) -> None:
        seg = np.zeros((20, OBS_DIM), dtype=np.float32)
        with pytest.raises(ValueError, match="stride must be positive"):
            build_segment_index([seg], seq_len=10, stride=0)

    def test_invalid_seq_len_raises(self) -> None:
        seg = np.zeros((20, OBS_DIM), dtype=np.float32)
        with pytest.raises(ValueError, match="seq_len must be positive"):
            build_segment_index([seg], seq_len=0, stride=1)


# ---------------------------------------------------------------------------
# NormStats tests
# ---------------------------------------------------------------------------

class TestNormStats:
    def test_compute(self) -> None:
        rng = np.random.default_rng(42)
        seg = rng.normal(5.0, 2.0, (100, OBS_DIM)).astype(np.float32)
        stats = compute_norm_stats([seg])
        np.testing.assert_allclose(stats.mean, seg.mean(axis=0), rtol=1e-5)
        np.testing.assert_allclose(stats.std, seg.std(axis=0), rtol=1e-5)

    def test_zero_std_replaced(self) -> None:
        seg = np.ones((50, OBS_DIM), dtype=np.float32)
        stats = compute_norm_stats([seg])
        assert np.all(stats.std >= 1.0)

    def test_empty_segments_raises(self) -> None:
        with pytest.raises(ValueError, match="empty segment list"):
            compute_norm_stats([])

    def test_roundtrip(self) -> None:
        stats = NormStats(
            mean=np.array([1.0, 2.0], dtype=np.float32),
            std=np.array([0.5, 1.5], dtype=np.float32),
        )
        restored = NormStats.from_dict(stats.to_dict())
        np.testing.assert_array_equal(stats.mean, restored.mean)
        np.testing.assert_array_equal(stats.std, restored.std)


# ---------------------------------------------------------------------------
# MorpheusDataset integration tests
# ---------------------------------------------------------------------------

class TestMorpheusDataset:
    def test_basic(self, parquet_dir: Path) -> None:
        ds = MorpheusDataset(parquet_dir, seq_len=10, stride=1)
        assert len(ds) > 0
        assert ds.obs_dim == OBS_DIM

    def test_getitem_shape(self, parquet_dir: Path) -> None:
        ds = MorpheusDataset(parquet_dir, seq_len=16, stride=1)
        sample = ds[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (16, OBS_DIM)
        assert sample.dtype == torch.float32

    def test_normalized_stats_applied(self, parquet_dir: Path) -> None:
        ds = MorpheusDataset(parquet_dir, seq_len=10, stride=1, normalize=True)
        assert ds.norm_stats is not None
        sample = ds[0]
        assert sample.abs().mean() < 10.0  # rough sanity check

    def test_no_normalization(self, parquet_dir: Path) -> None:
        ds = MorpheusDataset(parquet_dir, seq_len=10, stride=1, normalize=False)
        assert ds.norm_stats is None

    def test_external_norm_stats(self, parquet_dir: Path) -> None:
        stats = NormStats(
            mean=np.zeros(OBS_DIM, dtype=np.float32),
            std=np.ones(OBS_DIM, dtype=np.float32),
        )
        ds = MorpheusDataset(parquet_dir, seq_len=10, stride=1, norm_stats=stats)
        assert ds.norm_stats is stats

    def test_index_out_of_range(self, parquet_dir: Path) -> None:
        ds = MorpheusDataset(parquet_dir, seq_len=10, stride=1)
        with pytest.raises(IndexError):
            ds[len(ds)]
        with pytest.raises(IndexError):
            ds[-(len(ds) + 1)]

    def test_negative_index(self, parquet_dir: Path) -> None:
        ds = MorpheusDataset(parquet_dir, seq_len=10, stride=1)
        last = ds[-1]
        assert last.shape == (10, OBS_DIM)
        torch.testing.assert_close(last, ds[len(ds) - 1])

    def test_stride_reduces_length(self, parquet_dir: Path) -> None:
        ds1 = MorpheusDataset(parquet_dir, seq_len=10, stride=1)
        ds10 = MorpheusDataset(parquet_dir, seq_len=10, stride=10)
        assert len(ds10) < len(ds1)

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            MorpheusDataset(empty, seq_len=10)

    def test_sequences_no_nan(self, parquet_dir: Path) -> None:
        ds = MorpheusDataset(parquet_dir, seq_len=10, stride=10)
        for i in range(min(len(ds), 20)):
            assert not torch.any(torch.isnan(ds[i]))


# ---------------------------------------------------------------------------
# load_parquet_dir tests
# ---------------------------------------------------------------------------

class TestLoadParquetDir:
    def test_loads_and_sorts(self, parquet_dir: Path) -> None:
        df = load_parquet_dir(parquet_dir)
        assert len(df) == 120  # 60 + 60
        times = df["time"].to_list()
        assert times == sorted(times)

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "no_files"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            load_parquet_dir(empty)


# ---------------------------------------------------------------------------
# Enriched dataset (M5, H1, cross-asset, derived)
# ---------------------------------------------------------------------------

class TestEnrichedDataset:
    def _write_cross(
        self, tmp_path: Path, name: str, start: datetime, n: int,
    ) -> Path:
        d = tmp_path / name
        d.mkdir()
        df = _make_candle_df(n, start=start)
        df.write_parquet(d / "2025-01.parquet")
        return d

    def test_m5_features(self, parquet_dir: Path) -> None:
        ds = MorpheusDataset(
            parquet_dir=parquet_dir, seq_len=16, stride=1, use_m5=True,
        )
        assert ds.obs_dim == OBS_DIM + 6
        assert not torch.any(torch.isnan(ds[0]))

    def test_m5_and_h1(self, parquet_dir: Path) -> None:
        ds = MorpheusDataset(
            parquet_dir=parquet_dir, seq_len=16, stride=1,
            use_m5=True, use_h1=True,
        )
        assert ds.obs_dim == OBS_DIM + 6 + 6

    def test_cross_assets(self, parquet_dir: Path, tmp_path: Path) -> None:
        eur_dir = self._write_cross(
            tmp_path, "eurusd", datetime(2025, 1, 6, 8, 0, tzinfo=UTC), 60,
        )
        jpy_dir = self._write_cross(
            tmp_path, "usdjpy", datetime(2025, 1, 6, 8, 0, tzinfo=UTC), 60,
        )
        ds = MorpheusDataset(
            parquet_dir=parquet_dir, seq_len=16, stride=1,
            eurusd_dir=eur_dir, usdjpy_dir=jpy_dir,
        )
        assert ds.obs_dim == OBS_DIM + 12
        assert not torch.any(torch.isnan(ds[0]))

    def test_derived_without_cross(self, parquet_dir: Path) -> None:
        ds = MorpheusDataset(
            parquet_dir=parquet_dir, seq_len=16, stride=1,
            use_derived=True,
        )
        assert ds.obs_dim == OBS_DIM + 9
        # dxy columns fallback to 0 when no cross assets
        assert not torch.any(torch.isnan(ds[0]))

    def test_full_stack(self, parquet_dir: Path, tmp_path: Path) -> None:
        eur_dir = self._write_cross(
            tmp_path, "eurusd_full", datetime(2025, 1, 6, 8, 0, tzinfo=UTC), 60,
        )
        jpy_dir = self._write_cross(
            tmp_path, "usdjpy_full", datetime(2025, 1, 6, 8, 0, tzinfo=UTC), 60,
        )
        ds = MorpheusDataset(
            parquet_dir=parquet_dir, seq_len=16, stride=1,
            use_m5=True, use_h1=True,
            eurusd_dir=eur_dir, usdjpy_dir=jpy_dir,
            use_derived=True,
        )
        assert ds.obs_dim == OBS_DIM + 6 + 6 + 12 + 9
        assert not torch.any(torch.isnan(ds[0]))

    def test_get_close_matches_raw(self, parquet_dir: Path) -> None:
        ds = MorpheusDataset(parquet_dir=parquet_dir, seq_len=16, stride=1)
        price = ds.get_close(0)
        assert price > 0
        assert not np.isnan(price)
