"""Seed historical M5 data from Twelve Data into TimescaleDB.

Twelve Data free tier: 800 requests/day, 5000 datapoints/request.
No API key required for demo, but register for free key for higher limits.

Usage:
    uv run python -m scripts.seed_from_twelvedata --weeks 26
    uv run python -m scripts.seed_from_twelvedata --weeks 26 --interval 1min
"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections import deque
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

TWELVEDATA_URL = "https://api.twelvedata.com/time_series"

# Map our instrument names to Twelve Data symbols
OUR_TO_TWELVEDATA: dict[str, str] = {
    "EUR/USD": "EUR/USD",
    "GBP/USD": "GBP/USD",
    "XAUUSD": "XAU/USD",
    "SPX500": "SPX",
    "DOW30": "DJI",
    "DAX40": "DAX",
    "NIKKEI225": "NIKKEI/JPY",
}

_INTERVAL_TO_TF: dict[str, str] = {
    "5min": "M5",
    "1min": "M1",
    "15min": "M15",
    "1h": "H1",
}

_MAX_POINTS_PER_REQUEST = 5000
_MAX_REQUESTS_PER_MINUTE = 8  # Free tier: 8/min


async def _rate_limit(request_times: deque[float]) -> None:
    """Rate limit to stay within Twelve Data free tier."""
    now = time.monotonic()
    while request_times and now - request_times[0] > 60:
        request_times.popleft()
    if len(request_times) >= _MAX_REQUESTS_PER_MINUTE:
        wait = 60 - (now - request_times[0]) + 0.5
        if wait > 0:
            await logger.ainfo("twelvedata_rate_limit", wait_seconds=round(wait, 1))
            await asyncio.sleep(wait)
    request_times.append(time.monotonic())


def _parse_response(data: dict[str, object]) -> pl.DataFrame:
    """Parse Twelve Data response into a Polars DataFrame."""
    values = data.get("values")
    if not values or not isinstance(values, list):
        return pl.DataFrame(schema={
            "time": pl.Datetime("us", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        })

    records = []
    for v in values:
        try:
            dt = datetime.strptime(str(v["datetime"]), "%Y-%m-%d %H:%M:%S")
            dt = dt.replace(tzinfo=UTC)
            records.append({
                "time": dt,
                "open": float(v["open"]),
                "high": float(v["high"]),
                "low": float(v["low"]),
                "close": float(v["close"]),
                "volume": float(v.get("volume", 0)),
            })
        except (ValueError, KeyError, TypeError):
            continue

    if not records:
        return pl.DataFrame(schema={
            "time": pl.Datetime("us", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        })

    return pl.DataFrame(records).sort("time")


async def seed_instrument(
    session: aiohttp.ClientSession,
    storage: CandleStorage,
    instrument: str,
    td_symbol: str,
    interval: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    api_key: str,
    request_times: deque[float],
) -> int:
    """Seed one instrument from Twelve Data."""
    total = 0
    chunk_start = start

    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=8), end)
        await _rate_limit(request_times)

        params = {
            "symbol": td_symbol,
            "interval": interval,
            "start_date": chunk_start.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
            "outputsize": str(_MAX_POINTS_PER_REQUEST),
            "apikey": api_key,
            "timezone": "UTC",
        }

        try:
            async with session.get(TWELVEDATA_URL, params=params) as resp:
                if resp.status != 200:
                    await logger.awarning(
                        "twelvedata_request_failed",
                        status=resp.status,
                        symbol=td_symbol,
                    )
                    chunk_start = chunk_end
                    continue

                data = await resp.json()

            if "code" in data and data["code"] != 200:
                await logger.awarning(
                    "twelvedata_api_error",
                    symbol=td_symbol,
                    message=data.get("message", ""),
                )
                chunk_start = chunk_end
                continue

            df = _parse_response(data)
            if not df.is_empty():
                count = await storage.upsert_candles(instrument, timeframe, df)
                total += count

            await logger.ainfo(
                "twelvedata_chunk",
                instrument=instrument,
                start=chunk_start.strftime("%Y-%m-%d"),
                candles=len(df),
                total=total,
            )

        except Exception as e:
            await logger.awarning(
                "twelvedata_error",
                instrument=instrument,
                error=str(e),
            )

        chunk_start = chunk_end

    return total


async def seed(
    instruments: list[str],
    weeks: int,
    interval: str,
    api_key: str,
) -> None:
    """Seed all instruments from Twelve Data."""
    config = load_config()
    setup_logging(config.logging.level, json_format=False)

    db = Database(config.database)
    await db.connect()
    storage = CandleStorage(db)

    timeframe = _INTERVAL_TO_TF.get(interval, "M5")
    end = datetime.now(tz=UTC)
    start = end - timedelta(weeks=weeks)
    request_times: deque[float] = deque()

    try:
        async with aiohttp.ClientSession() as session:
            for instrument in instruments:
                td_symbol = OUR_TO_TWELVEDATA.get(instrument)
                if not td_symbol:
                    await logger.awarning(
                        "no_twelvedata_mapping", instrument=instrument
                    )
                    continue

                await logger.ainfo(
                    "seeding_twelvedata",
                    instrument=instrument,
                    symbol=td_symbol,
                    interval=interval,
                    weeks=weeks,
                )

                total = await seed_instrument(
                    session, storage, instrument, td_symbol,
                    interval, timeframe, start, end, api_key, request_times,
                )

                await logger.ainfo(
                    "instrument_complete",
                    instrument=instrument,
                    total_candles=total,
                )

    finally:
        await db.disconnect()


def main() -> None:
    """CLI entry point."""
    import os

    parser = argparse.ArgumentParser(
        description="Seed market data from Twelve Data (free)"
    )
    parser.add_argument(
        "--instruments", type=str, default=None,
        help="Comma-separated instrument names (default: all)",
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--weeks", type=int, default=26, help="Weeks of history (default: 26)",
    )
    parser.add_argument(
        "--interval", type=str, default="5min",
        choices=["1min", "5min", "15min", "1h"],
        help="Candle interval (default: 5min)",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Twelve Data API key (default: TWELVEDATA_API_KEY env or 'demo')",
    )

    args = parser.parse_args()

    if args.all or not args.instruments:
        instruments = list(OUR_TO_TWELVEDATA.keys())
    else:
        instruments = [i.strip() for i in args.instruments.split(",")]

    api_key = args.api_key or os.environ.get("TWELVEDATA_API_KEY", "demo")

    asyncio.run(seed(instruments, args.weeks, args.interval, api_key))


if __name__ == "__main__":
    main()
