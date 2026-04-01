"""Shared test fixtures."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from src.common.config import AppConfig, DatabaseConfig, load_config
from src.common.models import Candle, Timeframe


@pytest.fixture
def app_config() -> AppConfig:
    """Load the default application config."""
    return load_config()


@pytest.fixture
def db_config() -> DatabaseConfig:
    """Create a test database config."""
    return DatabaseConfig(
        url="postgresql://test:test@localhost:5432/test_trading_bot",
        min_connections=1,
        max_connections=5,
    )


@pytest.fixture
def sample_candles() -> list[Candle]:
    """Create sample candle data for testing."""
    base_time = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)
    return [
        Candle(
            time=datetime(
                base_time.year, base_time.month, base_time.day,
                base_time.hour, base_time.minute + i * 5,
                tzinfo=UTC,
            ),
            instrument="EUR/USD",
            timeframe=Timeframe.M5,
            open=1.0800 + i * 0.0001,
            high=1.0805 + i * 0.0001,
            low=1.0795 + i * 0.0001,
            close=1.0802 + i * 0.0001,
            volume=100.0 + i * 10,
            spread=0.8,
        )
        for i in range(12)  # 1 hour of M5 candles
    ]


@pytest.fixture
def sample_candles_df(sample_candles: list[Candle]) -> pl.DataFrame:
    """Create sample candle DataFrame for testing."""
    return pl.DataFrame([
        {
            "time": c.time,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
            "spread": c.spread,
        }
        for c in sample_candles
    ])
