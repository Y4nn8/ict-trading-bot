"""Seed historical data from Yahoo Finance into TimescaleDB.

Yahoo Finance provides free M5 data (max 60 days) and H1 data (max 730 days).
No API key required.

Usage:
    uv run python -m scripts.seed_from_yahoo --instruments EUR/USD,DAX40 --interval 5m
    uv run python -m scripts.seed_from_yahoo --all --interval 1h --period 6mo
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC

import polars as pl
import yfinance as yf
from dotenv import load_dotenv

from src.common.config import load_config
from src.common.db import Database
from src.common.logging import get_logger, setup_logging
from src.market_data.storage import CandleStorage

load_dotenv()

logger = get_logger(__name__)

# Map our instrument names to Yahoo Finance tickers
OUR_TO_YAHOO: dict[str, str] = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "XAUUSD": "GC=F",
    "SPX500": "^GSPC",
    "DOW30": "^DJI",
    "DAX40": "^GDAXI",
    "NIKKEI225": "^N225",
}

# Map Yahoo interval strings to our timeframe names
_INTERVAL_TO_TF: dict[str, str] = {
    "5m": "M5",
    "1h": "H1",
    "4h": "H4",
    "1d": "D1",
}

# Yahoo limits per interval
_INTERVAL_MAX_PERIOD: dict[str, str] = {
    "5m": "60d",
    "1h": "730d",
    "4h": "730d",
    "1d": "max",
}


def _download_yahoo(ticker: str, interval: str, period: str | None) -> pl.DataFrame:
    """Download data from Yahoo Finance and convert to Polars.

    Args:
        ticker: Yahoo ticker symbol.
        interval: Candle interval (5m, 1h, 4h, 1d).
        period: Data period (e.g. 60d, 6mo, 1y).

    Returns:
        Polars DataFrame with time, open, high, low, close, volume.
    """
    actual_period = period or _INTERVAL_MAX_PERIOD.get(interval, "60d")

    data = yf.download(ticker, period=actual_period, interval=interval, progress=False)

    if data.empty:
        return pl.DataFrame(schema={
            "time": pl.Datetime("us", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        })

    # yfinance returns multi-level columns with ticker name
    # Flatten them
    if hasattr(data.columns, "levels") and len(data.columns.levels) > 1:
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    # Find datetime column (could be "Datetime" or "Date")
    date_col = "Datetime" if "Datetime" in data.columns else "Date"

    records = []
    for _, row in data.iterrows():
        dt = row[date_col]
        if hasattr(dt, "to_pydatetime"):
            dt = dt.to_pydatetime()
        if dt.tzinfo is None:
            from datetime import timezone
            dt = dt.replace(tzinfo=timezone.utc)

        records.append({
            "time": dt,
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row.get("Volume", 0)),
        })

    return pl.DataFrame(records).sort("time")


async def seed(
    instruments: list[str],
    interval: str,
    period: str | None,
) -> None:
    """Download from Yahoo and seed into TimescaleDB."""
    config = load_config()
    setup_logging(config.logging.level, config.logging.json_format)

    db = Database(config.database)
    await db.connect()
    storage = CandleStorage(db)

    timeframe = _INTERVAL_TO_TF.get(interval, "M5")

    try:
        for instrument in instruments:
            ticker = OUR_TO_YAHOO.get(instrument)
            if not ticker:
                await logger.awarning("no_yahoo_mapping", instrument=instrument)
                continue

            await logger.ainfo(
                "downloading_yahoo",
                instrument=instrument,
                ticker=ticker,
                interval=interval,
            )

            df = _download_yahoo(ticker, interval, period)

            if df.is_empty():
                await logger.awarning("no_yahoo_data", instrument=instrument)
                continue

            count = await storage.upsert_candles(instrument, timeframe, df)
            await logger.ainfo(
                "yahoo_seeded",
                instrument=instrument,
                timeframe=timeframe,
                candles=count,
                start=str(df["time"][0]),
                end=str(df["time"][-1]),
            )

    finally:
        await db.disconnect()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Seed market data from Yahoo Finance"
    )
    parser.add_argument(
        "--instruments",
        type=str,
        default=None,
        help="Comma-separated instrument names (default: all)",
    )
    parser.add_argument("--all", action="store_true", help="Seed all instruments")
    parser.add_argument(
        "--interval",
        type=str,
        default="5m",
        choices=["5m", "1h", "4h", "1d"],
        help="Candle interval (default: 5m, max 60 days)",
    )
    parser.add_argument(
        "--period",
        type=str,
        default=None,
        help="Period override (e.g. 30d, 6mo, 1y). Default: max for interval.",
    )

    args = parser.parse_args()

    if args.all or not args.instruments:
        instruments = list(OUR_TO_YAHOO.keys())
    else:
        instruments = [i.strip() for i in args.instruments.split(",")]

    asyncio.run(seed(instruments, args.interval, args.period))


if __name__ == "__main__":
    main()
