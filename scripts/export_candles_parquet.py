"""Export ticks from TimescaleDB into 10s candles as Parquet files.

Aggregates raw ticks via SQL time_bucket into OHLC candles with
tick_count and spread, then writes one Parquet file per month.

Usage:
    uv run python -m scripts.export_candles_parquet \
        --instrument XAUUSD --start 2024-04-01 --end 2026-04-01 \
        --output-dir data/morpheus/xauusd

    # Custom bucket size:
    uv run python -m scripts.export_candles_parquet \
        --instrument XAUUSD --start 2024-04-01 --end 2026-04-01 \
        --bucket-seconds 30 --output-dir data/morpheus/xauusd_30s
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from src.common.config import load_config
from src.common.db import Database
from src.common.logging import get_logger

logger = get_logger(__name__)

DUKASCOPY_INSTRUMENTS: dict[str, str] = {
    "XAUUSD": "xauusd",
    "EUR/USD": "eurusd",
    "GBP/USD": "gbpusd",
}

CANDLE_QUERY = """
SELECT
    time_bucket($1::interval, time) AS bucket,
    (first(bid, time) + first(ask, time)) / 2.0 AS open,
    (max(bid) + max(ask)) / 2.0 AS high,
    (min(bid) + min(ask)) / 2.0 AS low,
    (last(bid, time) + last(ask, time)) / 2.0 AS close,
    count(*)::int AS tick_count,
    last(ask, time) - last(bid, time) AS spread
FROM ticks
WHERE instrument = $2 AND time >= $3 AND time < $4
GROUP BY bucket
ORDER BY bucket
"""


async def fetch_candles(
    db: Database,
    instrument: str,
    start: datetime,
    end: datetime,
    bucket_seconds: int,
) -> pl.DataFrame:
    """Fetch aggregated candles from ticks table via time_bucket."""
    db_instrument = DUKASCOPY_INSTRUMENTS.get(instrument, instrument.lower())
    interval = f"{bucket_seconds} seconds"

    rows = await db.fetch(CANDLE_QUERY, interval, db_instrument, start, end)
    if not rows:
        return pl.DataFrame(
            schema={
                "time": pl.Datetime("us", "UTC"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "tick_count": pl.Int32,
                "spread": pl.Float64,
            }
        )

    return pl.DataFrame(
        {
            "time": [r["bucket"] for r in rows],
            "open": [float(r["open"]) for r in rows],
            "high": [float(r["high"]) for r in rows],
            "low": [float(r["low"]) for r in rows],
            "close": [float(r["close"]) for r in rows],
            "tick_count": [int(r["tick_count"]) for r in rows],
            "spread": [float(r["spread"]) for r in rows],
        }
    )


def month_ranges(
    start: datetime, end: datetime
) -> list[tuple[datetime, datetime, str]]:
    """Split a date range into (month_start, month_end, label) tuples."""
    ranges: list[tuple[datetime, datetime, str]] = []
    cursor = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC)
    while cursor < end:
        year, month = cursor.year, cursor.month
        if month == 12:
            next_month = cursor.replace(year=year + 1, month=1)
        else:
            next_month = cursor.replace(month=month + 1)

        m_start = max(cursor, start)
        m_end = min(next_month, end)
        label = f"{year:04d}-{month:02d}"
        ranges.append((m_start, m_end, label))
        cursor = next_month
    return ranges


async def export(
    instrument: str,
    start: datetime,
    end: datetime,
    bucket_seconds: int,
    output_dir: Path,
) -> int:
    """Export candles to Parquet files partitioned by month."""
    config = load_config()
    db = Database(config.database)
    await db.connect()

    output_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0

    try:
        for m_start, m_end, label in month_ranges(start, end):
            df = await fetch_candles(db, instrument, m_start, m_end, bucket_seconds)
            if df.is_empty():
                await logger.ainfo("month_empty", month=label)
                continue

            path = output_dir / f"{label}.parquet"
            df.write_parquet(path)
            total_rows += len(df)
            await logger.ainfo(
                "month_exported", month=label, rows=len(df), path=str(path)
            )
    finally:
        await db.disconnect()

    await logger.ainfo(
        "export_complete",
        instrument=instrument,
        bucket_seconds=bucket_seconds,
        total_rows=total_rows,
        files=len(list(output_dir.glob("*.parquet"))),
    )
    return total_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Export ticks to candle Parquet files")
    parser.add_argument("--instrument", required=True, help="e.g. XAUUSD, EUR/USD")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--bucket-seconds", type=int, default=10, help="Candle duration (default: 10)",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for Parquet files")
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)

    total = await export(
        instrument=args.instrument,
        start=start,
        end=end,
        bucket_seconds=args.bucket_seconds,
        output_dir=Path(args.output_dir),
    )
    print(f"Exported {total} candles to {args.output_dir}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
