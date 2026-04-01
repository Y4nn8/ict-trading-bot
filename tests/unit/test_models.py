"""Tests for Pydantic data models."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.common.models import (
    Candle,
    Direction,
    ImpactLevel,
    NewsEvent,
    Timeframe,
    Trade,
)


class TestTimeframe:
    """Tests for Timeframe enum."""

    def test_values(self) -> None:
        assert Timeframe.M5 == "M5"
        assert Timeframe.H1 == "H1"
        assert Timeframe.H4 == "H4"
        assert Timeframe.D1 == "D1"

    def test_from_string(self) -> None:
        assert Timeframe("M5") is Timeframe.M5


class TestCandle:
    """Tests for Candle model validation."""

    def test_valid_candle(self) -> None:
        candle = Candle(
            time=datetime(2024, 1, 15, 10, 0, tzinfo=UTC),
            instrument="EUR/USD",
            timeframe=Timeframe.M5,
            open=1.0800,
            high=1.0810,
            low=1.0795,
            close=1.0805,
            volume=150.0,
        )
        assert candle.open == 1.0800
        assert candle.spread is None

    def test_high_below_low_raises(self) -> None:
        with pytest.raises(ValidationError, match=r"high.*must be >= low"):
            Candle(
                time=datetime(2024, 1, 15, 10, 0, tzinfo=UTC),
                instrument="EUR/USD",
                timeframe=Timeframe.M5,
                open=1.0800,
                high=1.0790,
                low=1.0795,
                close=1.0805,
                volume=100.0,
            )

    def test_high_below_close_raises(self) -> None:
        with pytest.raises(ValidationError, match=r"high.*must be >= max"):
            Candle(
                time=datetime(2024, 1, 15, 10, 0, tzinfo=UTC),
                instrument="EUR/USD",
                timeframe=Timeframe.M5,
                open=1.0800,
                high=1.0804,
                low=1.0795,
                close=1.0805,
                volume=100.0,
            )

    def test_low_above_open_raises(self) -> None:
        with pytest.raises(ValidationError, match=r"low.*must be <= min"):
            Candle(
                time=datetime(2024, 1, 15, 10, 0, tzinfo=UTC),
                instrument="EUR/USD",
                timeframe=Timeframe.M5,
                open=1.0800,
                high=1.0810,
                low=1.0801,
                close=1.0805,
                volume=100.0,
            )

    def test_candle_with_spread(self) -> None:
        candle = Candle(
            time=datetime(2024, 1, 15, 10, 0, tzinfo=UTC),
            instrument="EUR/USD",
            timeframe=Timeframe.M5,
            open=1.0800,
            high=1.0810,
            low=1.0795,
            close=1.0805,
            volume=100.0,
            spread=0.8,
        )
        assert candle.spread == 0.8

    def test_default_volume(self) -> None:
        candle = Candle(
            time=datetime(2024, 1, 15, 10, 0, tzinfo=UTC),
            instrument="NAS100",
            timeframe=Timeframe.H1,
            open=17000.0,
            high=17050.0,
            low=16990.0,
            close=17030.0,
        )
        assert candle.volume == 0.0


class TestTrade:
    """Tests for Trade model."""

    def test_minimal_trade(self) -> None:
        trade = Trade(
            opened_at=datetime(2024, 1, 15, 10, 0, tzinfo=UTC),
            instrument="EUR/USD",
            direction=Direction.LONG,
        )
        assert trade.id is not None
        assert trade.closed_at is None
        assert trade.pnl is None
        assert trade.is_backtest is False

    def test_complete_trade(self) -> None:
        trade = Trade(
            opened_at=datetime(2024, 1, 15, 10, 0, tzinfo=UTC),
            closed_at=datetime(2024, 1, 15, 14, 0, tzinfo=UTC),
            instrument="EUR/USD",
            direction=Direction.SHORT,
            entry_price=1.0800,
            exit_price=1.0750,
            stop_loss=1.0830,
            take_profit=1.0720,
            size=0.1,
            pnl=50.0,
            pnl_percent=1.5,
            r_multiple=1.67,
            confluence_score=0.75,
            is_backtest=True,
        )
        assert trade.direction == Direction.SHORT
        assert trade.pnl == 50.0


class TestNewsEvent:
    """Tests for NewsEvent model."""

    def test_news_event(self) -> None:
        event = NewsEvent(
            time=datetime(2024, 1, 15, 13, 30, tzinfo=UTC),
            source="finnhub",
            event_type="nfp",
            title="Non-Farm Payrolls",
            currency="USD",
            actual="216K",
            forecast="200K",
            previous="187K",
            impact_level=ImpactLevel.HIGH,
        )
        assert event.impact_level == ImpactLevel.HIGH
        assert event.instruments == []
