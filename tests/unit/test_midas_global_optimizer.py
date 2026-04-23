"""Tests for global (panel) Midas optimizer."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.midas.global_optimizer import (
    AGGREGATE_KEYS,
    Window,
    compute_aggregates,
    generate_disjoint_windows,
)


class TestGenerateDisjointWindows:
    """Windows must be back-to-back disjoint and chronologically ordered."""

    def test_basic_layout(self) -> None:
        start = datetime(2025, 1, 1, tzinfo=UTC)
        end = datetime(2025, 2, 20, tzinfo=UTC)  # 50 days
        ws = generate_disjoint_windows(
            start, end,
            train_days=14, test_days=1, val_days=1,
        )
        assert len(ws) == 3  # 50 / 16 = 3.125 → 3 full windows

        # First window must start at data_start
        assert ws[0].train_start == start
        # Each window is train+test+val long
        span = ws[0].val_end - ws[0].train_start
        assert span.days == 16

    def test_disjoint(self) -> None:
        ws = generate_disjoint_windows(
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 1, tzinfo=UTC),
            train_days=14, test_days=1, val_days=1, n_windows=5,
        )
        assert len(ws) == 5
        # Next window's train_start must be >= previous window's val_end
        for i in range(1, len(ws)):
            assert ws[i].train_start >= ws[i - 1].val_end

    def test_internal_adjacency(self) -> None:
        """Within a window, train_end == test_start, test_end == val_start."""
        ws = generate_disjoint_windows(
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 3, 1, tzinfo=UTC),
            train_days=14, test_days=2, val_days=1,
        )
        assert ws
        for w in ws:
            assert w.train_end == w.test_start
            assert w.test_end == w.val_start
            assert (w.train_end - w.train_start).days == 14
            assert (w.test_end - w.test_start).days == 2
            assert (w.val_end - w.val_start).days == 1

    def test_n_windows_limit(self) -> None:
        ws = generate_disjoint_windows(
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2030, 1, 1, tzinfo=UTC),
            train_days=14, test_days=1, val_days=1, n_windows=3,
        )
        assert len(ws) == 3

    def test_insufficient_range_returns_empty(self) -> None:
        ws = generate_disjoint_windows(
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 10, tzinfo=UTC),
            train_days=14, test_days=1, val_days=1,
        )
        assert ws == []

    def test_step_shorter_than_test_val_rejected(self) -> None:
        """step must be >= test_days + val_days to keep test/val disjoint."""
        with pytest.raises(ValueError, match="step_days"):
            generate_disjoint_windows(
                datetime(2025, 1, 1, tzinfo=UTC),
                datetime(2026, 1, 1, tzinfo=UTC),
                train_days=14, test_days=2, val_days=2, step_days=3,
            )

    def test_overlapping_trains_allowed(self) -> None:
        """step < train+test+val → trains overlap but tests/vals stay disjoint."""
        ws = generate_disjoint_windows(
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 6, 1, tzinfo=UTC),
            train_days=14, test_days=1, val_days=1, step_days=2,
        )
        assert len(ws) > 5
        # Trains overlap: W1.train_start < W0.train_end
        assert ws[1].train_start < ws[0].train_end
        # Tests stay disjoint: W1.test_start >= W0.test_end
        for i in range(1, len(ws)):
            assert ws[i].test_start >= ws[i - 1].test_end
            assert ws[i].val_start >= ws[i - 1].val_end

    def test_custom_step_larger_than_span(self) -> None:
        """step > span → gaps between windows are allowed."""
        ws = generate_disjoint_windows(
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 4, 1, tzinfo=UTC),
            train_days=14, test_days=1, val_days=1, step_days=30,
        )
        assert len(ws) == 3
        gap = (ws[1].train_start - ws[0].val_end).days
        assert gap == 14


class TestComputeAggregates:
    """Aggregates must all be present and correct for known inputs."""

    def test_all_keys_present(self) -> None:
        aggs = compute_aggregates([1.0, 2.0, 3.0])
        assert set(aggs.keys()) == set(AGGREGATE_KEYS)

    def test_empty_returns_zeros(self) -> None:
        aggs = compute_aggregates([])
        for k in AGGREGATE_KEYS:
            assert aggs[k] == 0.0

    def test_mean_median_sum(self) -> None:
        aggs = compute_aggregates([1.0, 2.0, 3.0, 4.0])
        assert aggs["mean_pnl"] == pytest.approx(2.5)
        assert aggs["median_pnl"] == pytest.approx(2.5)
        assert aggs["sum_pnl"] == pytest.approx(10.0)

    def test_pct_positive(self) -> None:
        aggs = compute_aggregates([-1.0, 0.0, 1.0, 2.0])
        # 2 out of 4 are strictly positive
        assert aggs["pct_positive"] == pytest.approx(0.5)

    def test_min_max(self) -> None:
        aggs = compute_aggregates([-5.0, 3.0, 10.0, -1.0])
        assert aggs["min_pnl"] == pytest.approx(-5.0)
        assert aggs["max_pnl"] == pytest.approx(10.0)

    def test_sharpe_positive_stable(self) -> None:
        """All positive, low dispersion → strong positive Sharpe."""
        aggs = compute_aggregates([100.0, 105.0, 95.0, 102.0])
        assert aggs["sharpe"] > 5.0

    def test_sharpe_zero_on_constant(self) -> None:
        """stdev = 0 → sharpe = 0 (avoid division by zero)."""
        aggs = compute_aggregates([3.0, 3.0, 3.0])
        assert aggs["sharpe"] == 0.0

    def test_sharpe_negative_on_losses(self) -> None:
        aggs = compute_aggregates([-5.0, -3.0, -4.0, -2.0])
        assert aggs["sharpe"] < 0.0

    def test_single_window_sharpe_zero(self) -> None:
        aggs = compute_aggregates([42.0])
        assert aggs["sharpe"] == 0.0

    def test_n_windows_traded_from_trades_count(self) -> None:
        """When trade counts are given, only windows with n_trades > 0
        contribute to n_windows_traded (not PnL > 0)."""
        aggs = compute_aggregates(
            window_pnls=[10.0, 0.0, -5.0, 0.0],
            window_n_trades=[3, 1, 2, 0],
        )
        # 3 windows have n_trades > 0 even though one has pnl=0
        assert aggs["n_windows_traded"] == 3

    def test_n_windows_traded_falls_back_to_pnl(self) -> None:
        """Without trade counts, non-zero PnL counts as traded."""
        aggs = compute_aggregates([10.0, 0.0, -5.0, 0.0])
        assert aggs["n_windows_traded"] == 2


class TestWindow:
    """Window dataclass should be immutable + hashable."""

    def test_window_frozen(self) -> None:
        w = Window(
            train_start=datetime(2025, 1, 1, tzinfo=UTC),
            train_end=datetime(2025, 1, 15, tzinfo=UTC),
            test_start=datetime(2025, 1, 15, tzinfo=UTC),
            test_end=datetime(2025, 1, 16, tzinfo=UTC),
            val_start=datetime(2025, 1, 16, tzinfo=UTC),
            val_end=datetime(2025, 1, 17, tzinfo=UTC),
        )
        with pytest.raises(AttributeError):
            w.train_start = datetime(2024, 1, 1, tzinfo=UTC)  # type: ignore[misc]
