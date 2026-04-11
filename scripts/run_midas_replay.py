"""Run Midas tick replay: extract features from historical ticks.

Streams ticks from TimescaleDB, builds 10s candles, runs all feature
extractors, and writes the feature matrix to Parquet.

Usage:
    uv run python -m scripts.run_midas_replay \
        --instrument XAUUSD \
        --start 2025-01-01 --end 2025-02-01 \
        --output features.parquet
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from src.common.config import load_config
from src.common.db import Database
from src.common.logging import setup_logging
from src.midas.replay_engine import (
    ReplayConfig,
    ReplayEngine,
    build_default_registry,
)


async def main(args: argparse.Namespace) -> None:
    """Run the replay."""
    config = load_config()
    setup_logging(config.logging.level, json_format=False)

    db = Database(config.database)
    await db.connect()

    try:
        # Count available ticks
        count = await db.fetchval(
            "SELECT COUNT(*) FROM ticks "
            "WHERE instrument = $1 AND time >= $2 AND time < $3",
            args.instrument,
            args.start,
            args.end,
        )
        print(
            f"Ticks available: {count:,} "
            f"({args.instrument}, {args.start.date()} → {args.end.date()})",
        )

        if count == 0:
            print("No ticks found. Exiting.")
            return

        registry = build_default_registry(instrument=args.instrument)
        # Apply default params
        registry.configure_all({})

        replay_config = ReplayConfig(
            instrument=args.instrument,
            start=args.start,
            end=args.end,
            output_path=Path(args.output) if args.output else None,
            sample_rate=args.sample_rate,
        )

        engine = ReplayEngine(db, registry, replay_config)
        result = await engine.run()

        print("\nReplay complete:")
        print(f"  Ticks processed: {result.total_ticks:,}")
        print(f"  Candles built:   {result.total_candles:,}")
        print(f"  Feature rows:    {result.feature_rows:,}")
        if result.output_path:
            size_mb = result.output_path.stat().st_size / 1024 / 1024
            print(f"  Output:          {result.output_path} ({size_mb:.1f} MB)")

    finally:
        await db.disconnect()


def cli() -> None:
    """CLI entry point."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Midas tick replay — extract features to Parquet",
    )
    parser.add_argument("--instrument", type=str, default="XAUUSD")
    parser.add_argument(
        "--start", type=str, required=True, help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", type=str, required=True, help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output Parquet file path (omit for dry run)",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=1,
        help="Extract features every N ticks (default: every tick)",
    )
    args = parser.parse_args()
    args.start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
    args.end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)

    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
