"""Tests for swing high/low detection."""

from __future__ import annotations

import polars as pl
import pytest

from src.structure.swings import (
    SwingState,
    SwingType,
    detect_swings_incremental,
    detect_swings_vectorized,
)
from tests.fixtures.annotated_candles import (
    EXPECTED_SWING_HIGHS,
    EXPECTED_SWING_LOWS,
    SWING_FIXTURE,
)


class TestSwingsVectorized:
    """Tests for vectorized swing detection."""

    def test_detects_known_swing_high(self) -> None:
        result = detect_swings_vectorized(SWING_FIXTURE)
        highs = result.filter(pl.col("swing_type") == SwingType.HIGH)
        assert len(highs) >= 1
        first_sh = highs.row(0, named=True)
        assert first_sh["index"] == EXPECTED_SWING_HIGHS[0]["index"]
        assert first_sh["price"] == pytest.approx(EXPECTED_SWING_HIGHS[0]["price"])

    def test_detects_known_swing_low(self) -> None:
        result = detect_swings_vectorized(SWING_FIXTURE)
        lows = result.filter(pl.col("swing_type") == SwingType.LOW)
        assert len(lows) >= 1
        # Find the swing low at index 9
        sl = lows.filter(pl.col("index") == EXPECTED_SWING_LOWS[0]["index"])
        assert len(sl) == 1
        assert sl["price"][0] == pytest.approx(EXPECTED_SWING_LOWS[0]["price"])

    def test_returns_empty_for_insufficient_data(self) -> None:
        short_df = SWING_FIXTURE.head(3)
        result = detect_swings_vectorized(short_df)
        assert result.is_empty()

    def test_custom_bar_count(self) -> None:
        result = detect_swings_vectorized(SWING_FIXTURE, left_bars=1, right_bars=1)
        assert len(result) > 0


class TestSwingsIncremental:
    """Tests for incremental swing detection."""

    def test_produces_same_results_as_vectorized(self) -> None:
        """Parametrized consistency test: incremental must match vectorized."""
        vec_result = detect_swings_vectorized(SWING_FIXTURE)

        state = SwingState()
        incr_swings = []
        for row in SWING_FIXTURE.to_dicts():
            new = detect_swings_incremental(row, state)
            incr_swings.extend(new)

        # Compare results
        vec_set = {(r["index"], r["swing_type"], round(r["price"], 6))
                   for r in vec_result.to_dicts()}
        incr_set = {(s.index, s.swing_type, round(s.price, 6))
                    for s in incr_swings}

        assert vec_set == incr_set

    def test_empty_before_enough_candles(self) -> None:
        state = SwingState()
        candle = SWING_FIXTURE.row(0, named=True)
        result = detect_swings_incremental(candle, state)
        assert result == []
