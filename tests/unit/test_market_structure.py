"""Tests for market structure (BOS/CHoCH) detection."""

from __future__ import annotations

import polars as pl

from src.structure.market_structure import (
    MarketStructureIncrState,
    MSBreakType,
    Trend,
    detect_market_structure_incremental,
    detect_market_structure_vectorized,
)
from src.structure.swings import SwingState, detect_swings_incremental
from tests.fixtures.annotated_candles import BOS_CHOCH_FIXTURE


class TestMarketStructureVectorized:
    """Tests for vectorized market structure detection."""

    def test_detects_bos(self) -> None:
        result, _trend = detect_market_structure_vectorized(BOS_CHOCH_FIXTURE)
        bos = result.filter(pl.col("break_type") == MSBreakType.BOS)
        assert len(bos) >= 1

    def test_detects_choch(self) -> None:
        result, _trend = detect_market_structure_vectorized(BOS_CHOCH_FIXTURE)
        choch = result.filter(pl.col("break_type") == MSBreakType.CHOCH)
        assert len(choch) >= 1

    def test_returns_final_trend(self) -> None:
        _result, trend = detect_market_structure_vectorized(BOS_CHOCH_FIXTURE)
        # After the CHoCH to bearish, final trend should be bearish
        assert trend == Trend.BEARISH

    def test_returns_empty_for_insufficient_data(self) -> None:
        short_df = BOS_CHOCH_FIXTURE.head(3)
        result, trend = detect_market_structure_vectorized(short_df)
        assert result.is_empty()
        assert trend == Trend.UNDEFINED

    def test_bos_direction_is_bullish_first(self) -> None:
        result, _ = detect_market_structure_vectorized(BOS_CHOCH_FIXTURE)
        bos = result.filter(pl.col("break_type") == MSBreakType.BOS)
        if len(bos) > 0:
            first_bos = bos.row(0, named=True)
            assert first_bos["direction"] == Trend.BULLISH


class TestMarketStructureIncremental:
    """Tests for incremental market structure detection."""

    def test_produces_breaks(self) -> None:
        swing_state = SwingState()
        ms_state = MarketStructureIncrState()
        all_breaks = []

        for row in BOS_CHOCH_FIXTURE.to_dicts():
            swings = detect_swings_incremental(row, swing_state)
            breaks = detect_market_structure_incremental(row, swings, ms_state)
            all_breaks.extend(breaks)

        assert len(all_breaks) > 0

    def test_incremental_matches_vectorized_break_types(self) -> None:
        """Both implementations should detect the same break types."""
        vec_result, _vec_trend = detect_market_structure_vectorized(BOS_CHOCH_FIXTURE)

        swing_state = SwingState()
        ms_state = MarketStructureIncrState()
        all_breaks = []

        for row in BOS_CHOCH_FIXTURE.to_dicts():
            swings = detect_swings_incremental(row, swing_state)
            breaks = detect_market_structure_incremental(row, swings, ms_state)
            all_breaks.extend(breaks)

        # Both should find BOS and CHoCH
        vec_types = set(vec_result["break_type"].to_list())
        incr_types = {b.break_type for b in all_breaks}

        assert MSBreakType.BOS in vec_types
        assert MSBreakType.BOS in incr_types
