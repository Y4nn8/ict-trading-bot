"""Tests for Midas CandleBuilder."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.midas.candle_builder import CandleBuilder
from src.midas.types import Tick


@pytest.fixture
def builder() -> CandleBuilder:
    """Create a CandleBuilder with 10s buckets."""
    return CandleBuilder(bucket_seconds=10)


def _tick(seconds: int, bid: float, ask: float) -> Tick:
    """Create a tick at a given second offset from a fixed base time."""
    base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    return Tick(time=base + timedelta(seconds=seconds), bid=bid, ask=ask)


class TestCandleBuilder:
    """Tests for CandleBuilder."""

    def test_first_tick_no_candle(self, builder: CandleBuilder) -> None:
        result = builder.process_tick(_tick(0, 100.0, 100.5))
        assert result is None
        assert builder.partial is not None
        assert builder.partial.tick_count == 1

    def test_same_bucket_updates_partial(self, builder: CandleBuilder) -> None:
        builder.process_tick(_tick(0, 100.0, 100.5))
        builder.process_tick(_tick(3, 101.0, 101.5))
        builder.process_tick(_tick(7, 99.0, 99.5))

        p = builder.partial
        assert p is not None
        assert p.tick_count == 3
        assert p.open == pytest.approx(100.25)  # mid of first tick
        assert p.high == pytest.approx(101.25)  # mid of highest
        assert p.low == pytest.approx(99.25)  # mid of lowest
        assert p.close == pytest.approx(99.25)  # mid of last tick

    def test_new_bucket_closes_candle(self, builder: CandleBuilder) -> None:
        builder.process_tick(_tick(0, 100.0, 100.5))
        builder.process_tick(_tick(5, 101.0, 101.5))

        # Tick in next bucket (10s)
        closed = builder.process_tick(_tick(10, 102.0, 102.5))

        assert closed is not None
        assert closed["open"] == pytest.approx(100.25)
        assert closed["high"] == pytest.approx(101.25)
        assert closed["low"] == pytest.approx(100.25)
        assert closed["close"] == pytest.approx(101.25)
        assert closed["tick_count"] == 2

        # New partial started
        assert builder.partial is not None
        assert builder.partial.tick_count == 1
        assert builder.candle_index == 1

    def test_multiple_candles(self, builder: CandleBuilder) -> None:
        closed_candles = []
        # 30 ticks over 3 buckets (0-9s, 10-19s, 20-29s)
        for i in range(30):
            result = builder.process_tick(_tick(i, 100.0 + i, 100.5 + i))
            if result is not None:
                closed_candles.append(result)

        # Should have 2 closed candles (bucket 0 and 1), bucket 2 still open
        assert len(closed_candles) == 2
        assert builder.candle_index == 2
        assert builder.partial is not None
        assert builder.partial.tick_count == 10

    def test_flush_closes_partial(self, builder: CandleBuilder) -> None:
        builder.process_tick(_tick(0, 100.0, 100.5))
        builder.process_tick(_tick(5, 101.0, 101.5))

        flushed = builder.flush()
        assert flushed is not None
        assert flushed["tick_count"] == 2
        assert builder.partial is None

    def test_flush_empty_returns_none(self, builder: CandleBuilder) -> None:
        assert builder.flush() is None

    def test_gap_in_ticks(self, builder: CandleBuilder) -> None:
        builder.process_tick(_tick(0, 100.0, 100.5))
        # Jump 60 seconds (skip 5 buckets)
        closed = builder.process_tick(_tick(60, 105.0, 105.5))

        assert closed is not None
        assert closed["tick_count"] == 1  # only the first tick
        assert builder.partial is not None
        assert builder.partial.tick_count == 1

    def test_reset(self, builder: CandleBuilder) -> None:
        builder.process_tick(_tick(0, 100.0, 100.5))
        builder.reset()
        assert builder.partial is None
        assert builder.candle_index == 0

    def test_elapsed_seconds(self, builder: CandleBuilder) -> None:
        builder.process_tick(_tick(0, 100.0, 100.5))
        builder.process_tick(_tick(7, 100.0, 100.5))

        assert builder.partial is not None
        assert builder.partial.elapsed_seconds == pytest.approx(7.0)

    def test_bucket_alignment(self) -> None:
        """Ticks at 03s and 13s should land in different 10s buckets."""
        builder = CandleBuilder(bucket_seconds=10)
        base = datetime(2025, 1, 1, 12, 0, 3, tzinfo=UTC)  # 03s
        t1 = Tick(time=base, bid=100.0, ask=100.5)
        t2 = Tick(time=base + timedelta(seconds=10), bid=101.0, ask=101.5)

        builder.process_tick(t1)
        closed = builder.process_tick(t2)

        assert closed is not None
        # First bucket starts at 00s (truncated)
        assert closed["time"].second == 0
