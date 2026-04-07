"""Tests for remaining detectors: liquidity, premium/discount, sessions, displacement, state."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from src.structure.displacement import (
    DisplacementState,
    detect_displacement_incremental,
    detect_displacement_vectorized,
)
from src.structure.liquidity import (
    LiquidityState,
    detect_liquidity_incremental,
    detect_liquidity_vectorized,
)
from src.structure.premium_discount import (
    PriceZone,
    classify_price_zone,
    compute_pd_levels,
    detect_pd_zones_vectorized,
    is_in_ote,
)
from src.structure.sessions import (
    Killzone,
    Session,
    add_session_columns_vectorized,
    get_killzone,
    get_session,
)
from src.structure.state import MarketStructureState
from tests.fixtures.annotated_candles import OB_FIXTURE, SWING_FIXTURE


class TestLiquidity:
    """Tests for liquidity pool detection."""

    @pytest.fixture
    def liq_fixture(self) -> pl.DataFrame:
        """Create data with equal highs."""
        n = 60
        base = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)
        times = [base + timedelta(minutes=i * 5) for i in range(n)]
        highs = [1.090 if i % 10 == 0 else 1.085 + (i % 5) * 0.001 for i in range(n)]
        lows = [h - 0.003 for h in highs]
        return pl.DataFrame({
            "time": times,
            "high": highs,
            "low": lows,
            "open": [h - 0.001 for h in highs],
            "close": [h - 0.002 for h in highs],
            "volume": [100.0] * n,
        })

    @pytest.fixture
    def swings_fixture(self, liq_fixture: pl.DataFrame) -> pl.DataFrame:
        """Create swings with equal highs at 1.090."""
        from src.structure.swings import detect_swings_vectorized

        return detect_swings_vectorized(liq_fixture, left_bars=2, right_bars=2)

    def test_vectorized_returns_dataframe(
        self, liq_fixture: pl.DataFrame, swings_fixture: pl.DataFrame,
    ) -> None:
        result = detect_liquidity_vectorized(liq_fixture, swings=swings_fixture)
        assert isinstance(result, pl.DataFrame)

    def test_incremental_returns_list(self) -> None:
        state = LiquidityState(lookback=5, min_touches=2)
        swing = {
            "time": datetime(2024, 1, 15, 10, 0, tzinfo=UTC),
            "price": 1.090,
            "swing_type": "swing_high",
            "index": 0,
        }
        result = detect_liquidity_incremental(swing, state)
        assert isinstance(result, list)


class TestPremiumDiscount:
    """Tests for premium/discount zone detection."""

    def test_compute_levels(self) -> None:
        levels = compute_pd_levels(swing_high=1.100, swing_low=1.080)
        assert levels.equilibrium == pytest.approx(1.090)
        assert levels.ote_high == pytest.approx(1.0924)
        assert levels.ote_low == pytest.approx(1.0958)

    def test_classify_premium(self) -> None:
        levels = compute_pd_levels(1.100, 1.080)
        assert classify_price_zone(1.095, levels) == PriceZone.PREMIUM

    def test_classify_discount(self) -> None:
        levels = compute_pd_levels(1.100, 1.080)
        assert classify_price_zone(1.085, levels) == PriceZone.DISCOUNT

    def test_classify_equilibrium(self) -> None:
        levels = compute_pd_levels(1.100, 1.080)
        assert classify_price_zone(1.090, levels) == PriceZone.EQUILIBRIUM

    def test_is_in_ote(self) -> None:
        levels = compute_pd_levels(1.100, 1.080)
        assert is_in_ote(1.094, levels)
        assert not is_in_ote(1.085, levels)

    def test_vectorized_zones(self) -> None:
        df = pl.DataFrame({
            "time": [datetime(2024, 1, 15, 10, 0, tzinfo=UTC)],
            "close": [1.095],
        })
        result = detect_pd_zones_vectorized(df, swing_high=1.100, swing_low=1.080)
        assert result["zone"][0] == PriceZone.PREMIUM


class TestSessions:
    """Tests for session/killzone classification."""

    def test_asian_session(self) -> None:
        assert get_session(2) == Session.ASIAN

    def test_london_session(self) -> None:
        assert get_session(8) == Session.LONDON

    def test_ny_session(self) -> None:
        assert get_session(14) == Session.NEW_YORK

    def test_off_hours(self) -> None:
        assert get_session(22) == Session.OFF_HOURS

    def test_london_killzone(self) -> None:
        assert get_killzone(8) == Killzone.LONDON_OPEN

    def test_ny_killzone(self) -> None:
        assert get_killzone(13) == Killzone.NEW_YORK_OPEN

    def test_no_killzone(self) -> None:
        assert get_killzone(22) == Killzone.NONE

    def test_vectorized_session_columns(self) -> None:
        df = pl.DataFrame({
            "time": [
                datetime(2024, 1, 15, 2, 0, tzinfo=UTC),   # Asian
                datetime(2024, 1, 15, 8, 0, tzinfo=UTC),   # London
                datetime(2024, 1, 15, 14, 0, tzinfo=UTC),  # NY
            ],
            "close": [1.08, 1.09, 1.10],
        })
        result = add_session_columns_vectorized(df)
        assert "session" in result.columns
        assert "killzone" in result.columns
        assert "in_killzone" in result.columns


class TestDisplacement:
    """Tests for displacement detection."""

    def test_vectorized_detects_displacement(self) -> None:
        result = detect_displacement_vectorized(OB_FIXTURE)
        assert len(result) > 0

    def test_vectorized_returns_empty_for_short_data(self) -> None:
        short = OB_FIXTURE.head(5)
        result = detect_displacement_vectorized(short)
        assert result.is_empty()

    def test_incremental_matches_vectorized(self) -> None:
        vec_result = detect_displacement_vectorized(OB_FIXTURE)

        state = DisplacementState()
        incr_disps = []
        for row in OB_FIXTURE.to_dicts():
            new = detect_displacement_incremental(row, state)
            incr_disps.extend(new)

        vec_indices = set(vec_result["index"].to_list())
        incr_indices = {d.index for d in incr_disps}
        assert vec_indices == incr_indices


class TestMarketStructureState:
    """Tests for the multi-TF state manager."""

    def test_process_candles(self) -> None:
        mss = MarketStructureState(
            instruments=["EUR/USD"],
            timeframes=["M5"],
        )
        for row in SWING_FIXTURE.to_dicts():
            events = mss.process_candle("EUR/USD", "M5", row)
            assert events is not None

    def test_get_trend(self) -> None:
        mss = MarketStructureState(
            instruments=["EUR/USD"],
            timeframes=["M5"],
        )
        # Before any candles, trend is undefined
        from src.structure.market_structure import Trend

        assert mss.get_trend("EUR/USD", "M5") == Trend.UNDEFINED

    def test_auto_register_unknown_pair(self) -> None:
        mss = MarketStructureState(instruments=[], timeframes=[])
        candle = SWING_FIXTURE.row(0, named=True)
        events = mss.process_candle("UNKNOWN", "H1", candle)
        assert events is not None
        assert mss.get_states("UNKNOWN", "H1") is not None

    def test_multi_instrument(self) -> None:
        mss = MarketStructureState(
            instruments=["EUR/USD", "GBP/USD"],
            timeframes=["M5", "H1"],
        )
        candle = SWING_FIXTURE.row(0, named=True)
        mss.process_candle("EUR/USD", "M5", candle)
        mss.process_candle("GBP/USD", "M5", candle)

        # States should be independent
        eur_states = mss.get_states("EUR/USD", "M5")
        gbp_states = mss.get_states("GBP/USD", "M5")
        assert eur_states is not None
        assert gbp_states is not None
        assert eur_states is not gbp_states
