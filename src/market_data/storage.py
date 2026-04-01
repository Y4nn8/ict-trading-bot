"""TimescaleDB storage for market data (candles)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from src.common.exceptions import DatabaseError
from src.common.logging import get_logger

if TYPE_CHECKING:
    from datetime import datetime

    from src.common.db import Database

logger = get_logger(__name__)


class CandleStorage:
    """Manages candle data in TimescaleDB.

    Args:
        db: Database connection manager.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    async def upsert_candles(
        self,
        instrument: str,
        timeframe: str,
        df: pl.DataFrame,
    ) -> int:
        """Insert or update candles into the database.

        Uses ON CONFLICT to upsert. Expects DataFrame columns:
        time, open, high, low, close, volume.

        Args:
            instrument: Instrument identifier.
            timeframe: Timeframe string (M5, H1, etc.).
            df: Polars DataFrame with candle data.

        Returns:
            Number of rows upserted.

        Raises:
            DatabaseError: If the database operation fails.
        """
        if df.is_empty():
            return 0

        query = """
            INSERT INTO candles
                (time, instrument, timeframe, open, high, low, close, volume, spread)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (time, instrument, timeframe)
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                spread = EXCLUDED.spread
        """

        rows = df.to_dicts()
        count = 0

        try:
            for row in rows:
                await self._db.execute(
                    query,
                    row["time"],
                    instrument,
                    timeframe,
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row.get("volume", 0)),
                    float(row["spread"]) if row.get("spread") is not None else None,
                )
                count += 1

            await logger.ainfo(
                "candles_upserted",
                instrument=instrument,
                timeframe=timeframe,
                count=count,
            )
            return count

        except DatabaseError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to upsert candles: {e}") from e

    async def fetch_candles(
        self,
        instrument: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> pl.DataFrame:
        """Fetch candles from the database.

        For base timeframe (M5), reads from the candles table.
        For aggregated timeframes (H1, H4, D1), reads from continuous aggregates.

        Args:
            instrument: Instrument identifier.
            timeframe: Timeframe string.
            start: Optional start datetime filter.
            end: Optional end datetime filter.
            limit: Optional max number of rows.

        Returns:
            Polars DataFrame with candle data.
        """
        # Use continuous aggregate views for higher timeframes
        table = _get_table_for_timeframe(timeframe)

        conditions = ["instrument = $1"]
        params: list[object] = [instrument]
        param_idx = 2

        if timeframe == "M5":
            conditions.append("timeframe = 'M5'")

        if start is not None:
            conditions.append(f"time >= ${param_idx}")
            params.append(start)
            param_idx += 1

        if end is not None:
            conditions.append(f"time < ${param_idx}")
            params.append(end)
            param_idx += 1

        where_clause = " AND ".join(conditions)
        limit_clause = f" LIMIT {limit}" if limit else ""

        query = f"""
            SELECT time, open, high, low, close, volume, spread
            FROM {table}
            WHERE {where_clause}
            ORDER BY time ASC
            {limit_clause}
        """

        try:
            records = await self._db.fetch(query, *params)
        except DatabaseError:
            raise
        except Exception as e:
            raise DatabaseError(f"Failed to fetch candles: {e}") from e

        if not records:
            return pl.DataFrame(
                schema={
                    "time": pl.Datetime("us", "UTC"),
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Float64,
                    "spread": pl.Float64,
                }
            )

        return pl.DataFrame(
            [dict(r) for r in records],
            schema={
                "time": pl.Datetime("us", "UTC"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "spread": pl.Float64,
            },
        )

    async def get_latest_candle_time(
        self,
        instrument: str,
        timeframe: str,
    ) -> datetime | None:
        """Get the most recent candle timestamp for an instrument/timeframe.

        Args:
            instrument: Instrument identifier.
            timeframe: Timeframe string.

        Returns:
            The latest candle datetime, or None if no data exists.
        """
        table = _get_table_for_timeframe(timeframe)
        conditions = ["instrument = $1"]
        params: list[object] = [instrument]

        if timeframe == "M5":
            conditions.append("timeframe = 'M5'")

        where_clause = " AND ".join(conditions)
        query = f"SELECT MAX(time) FROM {table} WHERE {where_clause}"

        result = await self._db.fetchval(query, *params)
        if result is None:
            return None
        return result  # type: ignore[no-any-return]

    async def get_candle_count(
        self,
        instrument: str,
        timeframe: str,
    ) -> int:
        """Get the number of candles stored for an instrument/timeframe.

        Args:
            instrument: Instrument identifier.
            timeframe: Timeframe string.

        Returns:
            Count of stored candles.
        """
        table = _get_table_for_timeframe(timeframe)
        conditions = ["instrument = $1"]
        params: list[object] = [instrument]

        if timeframe == "M5":
            conditions.append("timeframe = 'M5'")

        where_clause = " AND ".join(conditions)
        query = f"SELECT COUNT(*) FROM {table} WHERE {where_clause}"

        result = await self._db.fetchval(query, *params)
        return int(result) if result is not None else 0


def _get_table_for_timeframe(timeframe: str) -> str:
    """Map timeframe to the appropriate table or view name."""
    mapping = {
        "M5": "candles",
        "H1": "candles_h1",
        "H4": "candles_h4",
        "D1": "candles_d1",
    }
    table = mapping.get(timeframe)
    if table is None:
        raise DatabaseError(f"Unknown timeframe: {timeframe}")
    return table
