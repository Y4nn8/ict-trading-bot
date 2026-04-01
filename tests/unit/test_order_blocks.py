"""Tests for Order Block detection."""

from __future__ import annotations

import polars as pl
import pytest

from src.structure.order_blocks import (
    OBState,
    OBType,
    detect_order_blocks_incremental,
    detect_order_blocks_vectorized,
)
from tests.fixtures.annotated_candles import EXPECTED_BULLISH_OB, OB_FIXTURE


class TestOrderBlocksVectorized:
    """Tests for vectorized OB detection."""

    def test_detects_bullish_ob(self) -> None:
        result = detect_order_blocks_vectorized(OB_FIXTURE)
        bullish = result.filter(pl.col("ob_type") == OBType.BULLISH)
        assert len(bullish) >= 1
        ob = bullish.row(0, named=True)
        assert ob["index"] == EXPECTED_BULLISH_OB["index"]
        assert ob["top"] == pytest.approx(EXPECTED_BULLISH_OB["top"])
        assert ob["bottom"] == pytest.approx(EXPECTED_BULLISH_OB["bottom"])

    def test_returns_empty_for_insufficient_data(self) -> None:
        short_df = OB_FIXTURE.head(5)
        result = detect_order_blocks_vectorized(short_df)
        assert result.is_empty()


class TestOrderBlocksIncremental:
    """Tests for incremental OB detection."""

    def test_produces_same_results_as_vectorized(self) -> None:
        vec_result = detect_order_blocks_vectorized(OB_FIXTURE)

        state = OBState()
        incr_obs = []
        for row in OB_FIXTURE.to_dicts():
            new = detect_order_blocks_incremental(row, state)
            incr_obs.extend(new)

        vec_set = {(r["index"], r["ob_type"]) for r in vec_result.to_dicts()}
        incr_set = {(ob.index, ob.ob_type) for ob in incr_obs}

        assert vec_set == incr_set
