"""TimescaleDB storage for market data (candles)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from src.common.exceptions import DatabaseError
from src.common.logging import get_logger
from src.common.models import Timeframe

if TYPE_CHECKING:
    from datetime import datetime

    from src.common.db import Database

logger = get_logger(__name__)

CANDLE_SCHEMA: dict[str, Any] = {
    "time": pl.Datetime("us", "UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "spread": pl.Float64,
}

_TIMEFRAME_TABLE_MAP: dict[str, str] = {
    Timeframe.M5: "candles",
    Timeframe.H1: "candles_h1",
    Timeframe.H4: "candles_h4",
    Timeframe.D1: "candles_d1",
}


def _get_table_for_timeframe(timeframe: str) -> str:
    """Map timeframe to the appropriate table or view name."""
    table = _TIMEFRAME_TABLE_MAP.get(timeframe)
    if table is None:
        raise DatabaseError(f"Unknown timeframe: {timeframe}")
    return table


def _build_candle_where(
    instrument: str,
    timeframe: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> tuple[str, list[object]]:
    """Build WHERE clause and params for candle queries."""
    conditions = ["instrument = $1"]
    params: list[object] = [instrument]
    param_idx = 2

    if timeframe == Timeframe.M5:
        conditions.append("timeframe = 'M5'")

    if start is not None:
        conditions.append(f"time >= ${param_idx}")
        params.append(start)
        param_idx += 1

    if end is not None:
        conditions.append(f"time < ${param_idx}")
        params.append(end)
        param_idx += 1

    return " AND ".join(conditions), params


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
        """Insert or update candles into the database (batched).

        Args:
            instrument: Instrument identifier.
            timeframe: Timeframe string (M5, H1, etc.).
            df: Polars DataFrame with candle data.

        Returns:
            Number of rows upserted.
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
        args = [
            (
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
            for row in rows
        ]

        await self._db.executemany(query, args)

        count = len(args)
        await logger.ainfo(
            "candles_upserted",
            instrument=instrument,
            timeframe=timeframe,
            count=count,
        )
        return count

    async def fetch_candles(
        self,
        instrument: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> pl.DataFrame:
        """Fetch candles from the database.

        For M5, reads from the candles table directly.
        For H1/H4/D1, merges data from two sources:
        - The candles table (directly imported data, e.g. from Yahoo)
        - Continuous aggregate views (computed from M5 data)
        Deduplicates by time, preferring directly imported data.

        Args:
            instrument: Instrument identifier.
            timeframe: Timeframe string.
            start: Optional start datetime filter.
            end: Optional end datetime filter.
            limit: Optional max number of rows.

        Returns:
            Polars DataFrame with candle data.
        """
        limit_clause = f" LIMIT {limit}" if limit is not None else ""

        if timeframe == Timeframe.M5:
            where_clause, params = _build_candle_where(
                instrument, timeframe, start, end
            )
            query = f"""
                SELECT time, open, high, low, close, volume, spread
                FROM candles
                WHERE {where_clause}
                ORDER BY time ASC
                {limit_clause}
            """
        else:
            # Merge directly imported candles with continuous aggregate view.
            # Use DISTINCT ON to deduplicate by time, preferring imported data
            # (source=1) over aggregated data (source=2).
            agg_view = _get_table_for_timeframe(timeframe)
            where_direct, params_direct = _build_candle_where(
                instrument, timeframe, start, end
            )
            where_agg, params_agg = _build_candle_where(
                instrument, timeframe, start, end
            )
            # Offset param indices for the second query
            offset = len(params_direct)
            where_agg_reindexed = where_agg
            for i in range(len(params_agg), 0, -1):
                where_agg_reindexed = where_agg_reindexed.replace(
                    f"${i}", f"${i + offset}"
                )

            query = f"""
                SELECT DISTINCT ON (time) time, open, high, low, close, volume, spread
                FROM (
                    SELECT time, open, high, low, close, volume, spread, 1 AS source
                    FROM candles
                    WHERE {where_direct} AND timeframe = '{timeframe}'
                    UNION ALL
                    SELECT time, open, high, low, close, volume, spread, 2 AS source
                    FROM {agg_view}
                    WHERE {where_agg_reindexed}
                ) AS combined
                ORDER BY time ASC, source ASC
                {limit_clause}
            """
            params = [*params_direct, *params_agg]

        records = await self._db.fetch(query, *params)

        if not records:
            return pl.DataFrame(schema=CANDLE_SCHEMA)

        return pl.DataFrame([dict(r) for r in records], schema=CANDLE_SCHEMA)

    async def fetch_candles_raw(self, query: str, *args: object) -> pl.DataFrame:
        """Execute a raw SQL query and return results as a candle DataFrame."""
        records = await self._db.fetch(query, *args)
        if not records:
            return pl.DataFrame(schema=CANDLE_SCHEMA)
        return pl.DataFrame([dict(r) for r in records], schema=CANDLE_SCHEMA)

    async def get_latest_candle_time(
        self,
        instrument: str,
        timeframe: str,
    ) -> datetime | None:
        """Get the most recent candle timestamp for an instrument/timeframe.

        Checks both the candles table (direct imports) and continuous
        aggregate views, returning the most recent of the two.
        """
        # Check direct imports
        where_clause, params = _build_candle_where(instrument, timeframe)
        query = f"""
            SELECT MAX(time) FROM candles
            WHERE {where_clause} AND timeframe = ${ len(params) + 1}
        """
        direct_max = await self._db.fetchval(query, *params, timeframe)

        if timeframe == Timeframe.M5:
            return direct_max  # type: ignore[no-any-return]

        # Also check continuous aggregate view
        agg_table = _get_table_for_timeframe(timeframe)
        where_agg, params_agg = _build_candle_where(instrument, timeframe)
        query_agg = f"SELECT MAX(time) FROM {agg_table} WHERE {where_agg}"
        agg_max = await self._db.fetchval(query_agg, *params_agg)

        if direct_max is None:
            return agg_max  # type: ignore[no-any-return]
        if agg_max is None:
            return direct_max  # type: ignore[no-any-return]
        return max(direct_max, agg_max)  # type: ignore[no-any-return]

    async def get_candle_count(
        self,
        instrument: str,
        timeframe: str,
    ) -> int:
        """Get the number of candles stored for an instrument/timeframe."""
        table = _get_table_for_timeframe(timeframe)
        where_clause, params = _build_candle_where(instrument, timeframe)
        query = f"SELECT COUNT(*) FROM {table} WHERE {where_clause}"

        result = await self._db.fetchval(query, *params)
        return int(result) if result is not None else 0

    async def execute_raw(self, query: str, *args: object) -> str:
        """Execute a raw SQL statement through the database connection."""
        return await self._db.execute(query, *args)
