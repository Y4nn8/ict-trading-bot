"""Tests for CandleStorage (mocked DB)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import polars as pl
import pytest

from src.common.db import Database
from src.common.exceptions import DatabaseError
from src.market_data.storage import CandleStorage, _get_table_for_timeframe


class TestGetTableForTimeframe:
    """Tests for the timeframe-to-table mapping."""

    def test_m5_returns_candles(self) -> None:
        assert _get_table_for_timeframe("M5") == "candles"

    def test_h1_returns_view(self) -> None:
        assert _get_table_for_timeframe("H1") == "candles_h1"

    def test_h4_returns_view(self) -> None:
        assert _get_table_for_timeframe("H4") == "candles_h4"

    def test_d1_returns_view(self) -> None:
        assert _get_table_for_timeframe("D1") == "candles_d1"

    def test_unknown_raises(self) -> None:
        with pytest.raises(DatabaseError, match="Unknown timeframe"):
            _get_table_for_timeframe("M15")


class TestCandleStorage:
    """Tests for CandleStorage with mocked database."""

    @pytest.fixture
    def mock_db(self) -> Database:
        db = AsyncMock(spec=Database)
        db.execute = AsyncMock(return_value="INSERT 0 1")
        db.fetch = AsyncMock(return_value=[])
        db.fetchval = AsyncMock(return_value=None)
        return db  # type: ignore[return-value]

    @pytest.fixture
    def storage(self, mock_db: Database) -> CandleStorage:
        return CandleStorage(mock_db)

    async def test_upsert_empty_df_returns_zero(self, storage: CandleStorage) -> None:
        df = pl.DataFrame(
            schema={
                "time": pl.Datetime("us", "UTC"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )
        result = await storage.upsert_candles("EUR/USD", "M5", df)
        assert result == 0

    async def test_upsert_calls_execute_per_row(
        self, storage: CandleStorage, mock_db: Database
    ) -> None:
        df = pl.DataFrame({
            "time": [
                datetime(2024, 1, 15, 10, 0, tzinfo=UTC),
                datetime(2024, 1, 15, 10, 5, tzinfo=UTC),
            ],
            "open": [1.08, 1.081],
            "high": [1.082, 1.083],
            "low": [1.079, 1.080],
            "close": [1.081, 1.082],
            "volume": [100.0, 110.0],
        })
        result = await storage.upsert_candles("EUR/USD", "M5", df)
        assert result == 2
        assert mock_db.execute.call_count == 2  # type: ignore[union-attr]

    async def test_fetch_candles_returns_empty_df(
        self, storage: CandleStorage
    ) -> None:
        result = await storage.fetch_candles("EUR/USD", "M5")
        assert isinstance(result, pl.DataFrame)
        assert result.is_empty()

    async def test_get_latest_candle_time_returns_none(
        self, storage: CandleStorage
    ) -> None:
        result = await storage.get_latest_candle_time("EUR/USD", "M5")
        assert result is None

    async def test_get_candle_count_returns_zero(
        self, storage: CandleStorage
    ) -> None:
        result = await storage.get_candle_count("EUR/USD", "M5")
        assert result == 0
