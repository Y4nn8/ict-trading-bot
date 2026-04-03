"""Seed historical OHLCV data from IG Markets into TimescaleDB.

Usage:
    uv run python -m scripts.seed_historical_data [--days 180] [--instruments EUR/USD,NAS100]
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from dotenv import load_dotenv

from src.common.config import load_config

load_dotenv()
from src.common.db import Database
from src.common.logging import get_logger, setup_logging
from src.market_data.ig_client import IGClient
from src.market_data.ingestion import MarketDataIngester
from src.market_data.storage import CandleStorage

logger = get_logger(__name__)


async def seed(days: int, instrument_names: list[str] | None) -> None:
    """Run the historical data seeding process.

    Args:
        days: Number of days of history to fetch.
        instrument_names: Optional list of instrument names to seed. If None, seeds all.
    """
    config = load_config()
    setup_logging(config.logging.level, config.logging.json_format)

    # Filter instruments if specified
    instruments = config.instruments
    if instrument_names:
        name_set = set(instrument_names)
        instruments = [i for i in instruments if i.name in name_set]
        if not instruments:
            await logger.aerror(
                "no_matching_instruments", requested=instrument_names
            )
            sys.exit(1)

    # Connect to DB and broker
    db = Database(config.database)
    await db.connect()

    ig_client = IGClient(config.broker)
    ig_client.connect()

    storage = CandleStorage(db)
    ingester = MarketDataIngester(ig_client, storage)

    try:
        for instrument in instruments:
            await logger.ainfo(
                "seeding_instrument",
                instrument=instrument.name,
                epic=instrument.epic,
                days=days,
            )
            count = await ingester.ingest_historical(
                instrument=instrument,
                days=days,
                timeframe=config.market_data.base_timeframe,
            )
            await logger.ainfo(
                "instrument_seeded",
                instrument=instrument.name,
                candles_stored=count,
            )

        # Refresh aggregates after bulk insert
        await logger.ainfo("refreshing_aggregates")
        await ingester.refresh_aggregates()
        await logger.ainfo("seeding_complete")

    finally:
        ig_client.disconnect()
        await db.disconnect()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Seed historical market data from IG Markets")
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Number of days of history to fetch (default: 180)",
    )
    parser.add_argument(
        "--instruments",
        type=str,
        default=None,
        help="Comma-separated list of instrument names (default: all configured)",
    )

    args = parser.parse_args()
    instrument_list = args.instruments.split(",") if args.instruments else None

    asyncio.run(seed(days=args.days, instrument_names=instrument_list))


if __name__ == "__main__":
    main()
