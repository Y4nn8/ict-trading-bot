"""Download FXCM data and seed directly into TimescaleDB.

Combines download + CSV import in one step. No intermediate files needed.

Usage:
    uv run python -m scripts.seed_from_fxcm --instruments EUR/USD,DAX40 --weeks 26
    uv run python -m scripts.seed_from_fxcm --all --weeks 12
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import io
from datetime import UTC, datetime, timedelta

import aiohttp
import polars as pl
from dotenv import load_dotenv

from src.common.config import load_config
from src.common.db import Database
from src.common.logging import get_logger, setup_logging
from src.market_data.storage import CandleStorage

load_dotenv()

logger = get_logger(__name__)

FXCM_BASE_URL = "https://tickdata.fxcorporate.com"

OUR_TO_FXCM: dict[str, str] = {
    "EUR/USD": "EURUSD",
    "GBP/USD": "GBPUSD",
    "XAUUSD": "XAUUSD",
    "SPX500": "SPX500",
    "DOW30": "US30",
    "DAX40": "GER40",
    "NIKKEI225": "JPN225",
}


def _parse_fxcm_csv(csv_text: str) -> pl.DataFrame:
    """Parse FXCM CSV text into a candle DataFrame."""
    df = pl.read_csv(io.StringIO(csv_text), try_parse_dates=True)
    df = df.rename({c: c.lower().strip() for c in df.columns})

    # Find date column
    date_col = None
    for candidate in ["datetime", "date", "time"]:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        # FXCM sometimes uses the first column without a clear name
        date_col = df.columns[0]

    if df[date_col].dtype == pl.Utf8:
        df = df.with_columns(
            pl.col(date_col).str.to_datetime().dt.replace_time_zone("UTC").alias("time")
        )
    else:
        df = df.with_columns(pl.col(date_col).alias("time"))

    # Use bid prices if available, otherwise direct OHLC
    if "bidopen" in df.columns:
        return df.select(
            pl.col("time"),
            pl.col("bidopen").cast(pl.Float64).alias("open"),
            pl.col("bidhigh").cast(pl.Float64).alias("high"),
            pl.col("bidlow").cast(pl.Float64).alias("low"),
            pl.col("bidclose").cast(pl.Float64).alias("close"),
            (
                pl.col("tickqty").cast(pl.Float64).alias("volume")
                if "tickqty" in df.columns
                else pl.lit(0.0).alias("volume")
            ),
        ).sort("time")

    return df.select(
        pl.col("time"),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.lit(0.0).alias("volume"),
    ).sort("time")


async def seed_instrument(
    session: aiohttp.ClientSession,
    storage: CandleStorage,
    instrument: str,
    fxcm_symbol: str,
    weeks: int,
) -> int:
    """Download and seed one instrument.

    Args:
        session: HTTP session.
        storage: DB storage.
        instrument: Our instrument name.
        fxcm_symbol: FXCM symbol.
        weeks: Number of weeks.

    Returns:
        Total candles stored.
    """
    now = datetime.now(tz=UTC)
    total = 0

    for w in range(weeks):
        target_date = now - timedelta(weeks=w)
        year = target_date.isocalendar()[0]
        week = target_date.isocalendar()[1]

        url = f"{FXCM_BASE_URL}/{fxcm_symbol}/{year}/{week}.csv.gz"

        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    await logger.ainfo(
                        "fxcm_week_unavailable",
                        instrument=instrument,
                        year=year,
                        week=week,
                    )
                    continue

                compressed = await resp.read()
                csv_text = gzip.decompress(compressed).decode("utf-8")

            df = _parse_fxcm_csv(csv_text)
            if df.is_empty():
                continue

            count = await storage.upsert_candles(instrument, "M5", df)
            total += count

            await logger.ainfo(
                "fxcm_week_seeded",
                instrument=instrument,
                year=year,
                week=week,
                candles=count,
            )

        except Exception as e:
            await logger.awarning(
                "fxcm_week_error",
                instrument=instrument,
                year=year,
                week=week,
                error=str(e),
            )

    return total


async def seed(instruments: list[str], weeks: int) -> None:
    """Main seed process."""
    config = load_config()
    setup_logging(config.logging.level, config.logging.json_format)

    db = Database(config.database)
    await db.connect()
    storage = CandleStorage(db)

    try:
        async with aiohttp.ClientSession() as session:
            for instrument in instruments:
                fxcm_sym = OUR_TO_FXCM.get(instrument)
                if not fxcm_sym:
                    await logger.awarning(
                        "no_fxcm_mapping", instrument=instrument
                    )
                    continue

                await logger.ainfo(
                    "seeding_from_fxcm",
                    instrument=instrument,
                    fxcm_symbol=fxcm_sym,
                    weeks=weeks,
                )
                total = await seed_instrument(
                    session, storage, instrument, fxcm_sym, weeks
                )
                await logger.ainfo(
                    "instrument_seeded",
                    instrument=instrument,
                    total_candles=total,
                )
    finally:
        await db.disconnect()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download FXCM data and seed into TimescaleDB"
    )
    parser.add_argument(
        "--instruments",
        type=str,
        default=None,
        help="Comma-separated instrument names (default: all)",
    )
    parser.add_argument("--all", action="store_true", help="Seed all instruments")
    parser.add_argument(
        "--weeks", type=int, default=26, help="Weeks of history (default: 26)"
    )

    args = parser.parse_args()

    if args.all or not args.instruments:
        instruments = list(OUR_TO_FXCM.keys())
    else:
        instruments = [i.strip() for i in args.instruments.split(",")]

    asyncio.run(seed(instruments, args.weeks))


if __name__ == "__main__":
    main()
