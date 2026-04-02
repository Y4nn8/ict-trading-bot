"""Tests for FVG detection."""

from __future__ import annotations

import polars as pl
import pytest

from src.structure.fvg import (
    FVGState,
    FVGType,
    detect_fvg_incremental,
    detect_fvg_vectorized,
)
from tests.fixtures.annotated_candles import (
    EXPECTED_BEARISH_FVG,
    EXPECTED_BULLISH_FVG,
    FVG_FIXTURE,
)


class TestFVGVectorized:
    """Tests for vectorized FVG detection."""

    def test_detects_bullish_fvg(self) -> None:
        result = detect_fvg_vectorized(FVG_FIXTURE)
        bullish = result.filter(pl.col("fvg_type") == FVGType.BULLISH)
        assert len(bullish) >= 1
        # Find the specific intended FVG
        target = bullish.filter(pl.col("index") == EXPECTED_BULLISH_FVG["index"])
        assert len(target) == 1
        fvg = target.row(0, named=True)
        assert fvg["top"] == pytest.approx(EXPECTED_BULLISH_FVG["top"])
        assert fvg["bottom"] == pytest.approx(EXPECTED_BULLISH_FVG["bottom"])

    def test_detects_bearish_fvg(self) -> None:
        result = detect_fvg_vectorized(FVG_FIXTURE)
        bearish = result.filter(pl.col("fvg_type") == FVGType.BEARISH)
        assert len(bearish) >= 1
        fvg = bearish.row(0, named=True)
        assert fvg["index"] == EXPECTED_BEARISH_FVG["index"]
        assert fvg["top"] == pytest.approx(EXPECTED_BEARISH_FVG["top"])
        assert fvg["bottom"] == pytest.approx(EXPECTED_BEARISH_FVG["bottom"])

    def test_returns_empty_for_insufficient_data(self) -> None:
        short_df = FVG_FIXTURE.head(2)
        result = detect_fvg_vectorized(short_df)
        assert result.is_empty()

    def test_midpoint_calculation(self) -> None:
        result = detect_fvg_vectorized(FVG_FIXTURE)
        bullish = result.filter(pl.col("fvg_type") == FVGType.BULLISH)
        fvg = bullish.row(0, named=True)
        expected_mid = (fvg["top"] + fvg["bottom"]) / 2
        assert fvg["midpoint"] == pytest.approx(expected_mid)


class TestFVGIncremental:
    """Tests for incremental FVG detection."""

    def test_produces_same_results_as_vectorized(self) -> None:
        vec_result = detect_fvg_vectorized(FVG_FIXTURE)

        state = FVGState()
        incr_fvgs = []
        for row in FVG_FIXTURE.to_dicts():
            new = detect_fvg_incremental(row, state)
            incr_fvgs.extend(new)

        vec_set = {(r["index"], r["fvg_type"], round(r["top"], 6), round(r["bottom"], 6))
                   for r in vec_result.to_dicts()}
        incr_set = {(f.index, f.fvg_type, round(f.top, 6), round(f.bottom, 6))
                    for f in incr_fvgs}

        assert vec_set == incr_set
