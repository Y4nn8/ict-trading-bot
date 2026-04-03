"""Seed historical data from CSV files into TimescaleDB.

Usage:
    uv run python -m scripts.seed_from_csv --file data/EURUSD_M5.csv --instrument EUR/USD
    uv run python -m scripts.seed_from_csv --dir data/ --format fxcm

File naming convention for --dir mode: {INSTRUMENT}_M5.csv
  e.g. EURUSD_M5.csv, XAUUSD_M5.csv, DAX40_M5.csv
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from src.common.config import load_config
from src.common.db import Database
from src.common.logging import get_logger, setup_logging
from src.market_data.csv_adapter import CSVFormat, load_csv
from src.market_data.storage import CandleStorage

load_dotenv()

logger = get_logger(__name__)

# Map filename prefixes to instrument names
_FILENAME_TO_INSTRUMENT: dict[str, str] = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "SPX500": "SPX500",
    "DOW30": "DOW30",
    "DAX40": "DAX40",
    "NIKKEI225": "NIKKEI225",
    "XAUUSD": "XAUUSD",
}


async def seed_file(
    storage: CandleStorage,
    file_path: Path,
    instrument: str,
    timeframe: str,
    csv_format: CSVFormat,
) -> int:
    """Seed a single CSV file into the database.

    Args:
        storage: Candle storage manager.
        file_path: Path to CSV file.
        instrument: Instrument name.
        timeframe: Timeframe string.
        csv_format: CSV format type.

    Returns:
        Number of candles stored.
    """
    await logger.ainfo(
        "loading_csv",
        file=str(file_path),
        instrument=instrument,
    )
    df = load_csv(file_path, csv_format)

    if df.is_empty():
        await logger.awarning("empty_csv", file=str(file_path))
        return 0

    count = await storage.upsert_candles(instrument, timeframe, df)
    await logger.ainfo(
        "csv_seeded",
        instrument=instrument,
        candles=count,
        start=str(df["time"][0]),
        end=str(df["time"][-1]),
    )
    return count


async def seed(
    file_path: str | None,
    directory: str | None,
    instrument: str | None,
    timeframe: str,
    csv_format: CSVFormat,
) -> None:
    """Run CSV seeding."""
    config = load_config()
    setup_logging(config.logging.level, config.logging.json_format)

    db = Database(config.database)
    await db.connect()
    storage = CandleStorage(db)

    try:
        if file_path:
            if not instrument:
                await logger.aerror("instrument_required_for_single_file")
                return
            await seed_file(storage, Path(file_path), instrument, timeframe, csv_format)

        elif directory:
            dir_path = Path(directory)
            csv_files = sorted(dir_path.glob("*.csv"))
            if not csv_files:
                await logger.awarning("no_csv_files", dir=directory)
                return

            total = 0
            for csv_file in csv_files:
                # Derive instrument name from filename
                stem = csv_file.stem.upper().replace("_M5", "").replace("_H1", "")
                instr = _FILENAME_TO_INSTRUMENT.get(stem)
                if instr is None:
                    await logger.awarning(
                        "unknown_instrument_file",
                        file=csv_file.name,
                        hint="Add mapping to _FILENAME_TO_INSTRUMENT",
                    )
                    continue
                count = await seed_file(storage, csv_file, instr, timeframe, csv_format)
                total += count

            await logger.ainfo("csv_seed_complete", total_candles=total)
    finally:
        await db.disconnect()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Seed market data from CSV files")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to a single CSV file")
    group.add_argument("--dir", type=str, help="Directory containing CSV files")
    parser.add_argument("--instrument", type=str, help="Instrument name (required with --file)")
    parser.add_argument("--timeframe", type=str, default="M5", help="Timeframe (default: M5)")
    parser.add_argument(
        "--format",
        type=str,
        default="generic",
        choices=["generic", "fxcm", "metatrader"],
        help="CSV format (default: generic)",
    )

    args = parser.parse_args()
    asyncio.run(seed(
        file_path=args.file,
        directory=args.dir,
        instrument=args.instrument,
        timeframe=args.timeframe,
        csv_format=CSVFormat(args.format),
    ))


if __name__ == "__main__":
    main()
