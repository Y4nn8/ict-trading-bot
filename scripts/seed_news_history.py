"""Seed historical news data from configured sources into the database.

Usage:
    uv run python -m scripts.seed_news_history [--days 180] [--source finnhub]
"""

from __future__ import annotations

import argparse
import asyncio

from src.common.config import load_config
from src.common.db import Database
from src.common.logging import get_logger, setup_logging
from src.news.calendar.finnhub import FinnhubCalendarSource
from src.news.store import NewsStore

logger = get_logger(__name__)


async def seed(days: int, source_name: str) -> None:
    """Run the news seeding process.

    Args:
        days: Number of days of history.
        source_name: Source to use (finnhub, ff_api).
    """
    from datetime import UTC, datetime, timedelta

    config = load_config()
    setup_logging(config.logging.level, config.logging.json_format)

    db = Database(config.database)
    await db.connect()
    store = NewsStore(db)

    end = datetime.now(tz=UTC)
    start = end - timedelta(days=days)

    try:
        if source_name == "finnhub":
            source = FinnhubCalendarSource(api_key="")  # Set via config/env
            await source.connect()
            events = await source.get_events(start, end)
            await store.save_events(events)
            await source.disconnect()
        else:
            await logger.aerror("unknown_source", source=source_name)
    finally:
        await db.disconnect()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Seed historical news data")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--source", type=str, default="finnhub")
    args = parser.parse_args()
    asyncio.run(seed(days=args.days, source_name=args.source))


if __name__ == "__main__":
    main()
