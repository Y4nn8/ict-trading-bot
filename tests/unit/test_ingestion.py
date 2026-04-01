"""Tests for MarketDataIngester."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import polars as pl
import pytest

from src.common.config import InstrumentConfig
from src.market_data.ingestion import MarketDataIngester


@pytest.fixture
def instrument() -> InstrumentConfig:
    return InstrumentConfig(
        name="EUR/USD",
        epic="CS.D.EURUSD.CFD.IP",
        asset_class="forex",
        leverage=30,
    )


@pytest.fixture
def mock_ig_client() -> MagicMock:
    client = MagicMock()
    client.fetch_historical_candles.return_value = pl.DataFrame(
        schema={
            "time": pl.Datetime("us", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        }
    )
    return client


@pytest.fixture
def mock_storage() -> AsyncMock:
    storage = AsyncMock()
    storage.get_latest_candle_time = AsyncMock(return_value=None)
    storage.upsert_candles = AsyncMock(return_value=0)
    storage.execute_raw = AsyncMock(return_value="CALL")
    return storage


@pytest.fixture
def ingester(mock_ig_client: MagicMock, mock_storage: AsyncMock) -> MarketDataIngester:
    return MarketDataIngester(mock_ig_client, mock_storage)


class TestMarketDataIngester:
    """Tests for ingestion pipeline."""

    async def test_ingest_empty_data(
        self, ingester: MarketDataIngester, instrument: InstrumentConfig
    ) -> None:
        result = await ingester.ingest_historical(instrument, days=1)
        assert result == 0

    async def test_ingest_with_data(
        self,
        ingester: MarketDataIngester,
        instrument: InstrumentConfig,
        mock_ig_client: MagicMock,
        mock_storage: AsyncMock,
    ) -> None:
        # Return some data for the first chunk
        mock_ig_client.fetch_historical_candles.return_value = pl.DataFrame({
            "time": [datetime(2024, 1, 15, 10, 0, tzinfo=UTC)],
            "open": [1.08],
            "high": [1.082],
            "low": [1.079],
            "close": [1.081],
            "volume": [100.0],
        })
        mock_storage.upsert_candles.return_value = 1

        result = await ingester.ingest_historical(instrument, days=3)
        assert result > 0
        assert mock_storage.upsert_candles.called

    async def test_ingest_resumes_from_latest(
        self,
        ingester: MarketDataIngester,
        instrument: InstrumentConfig,
        mock_storage: AsyncMock,
    ) -> None:
        # Simulate existing data up to now - should return 0 (up to date)
        mock_storage.get_latest_candle_time.return_value = (
            datetime.now(tz=UTC) + timedelta(minutes=1)
        )
        result = await ingester.ingest_historical(instrument, days=1)
        assert result == 0

    async def test_refresh_aggregates(self, ingester: MarketDataIngester) -> None:
        await ingester.refresh_aggregates()
        # Should attempt to refresh 3 views (h1, h4, d1)
        assert ingester._storage.execute_raw.call_count == 3  # type: ignore[union-attr]
