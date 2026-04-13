"""Download tick data from Dukascopy and store in TimescaleDB.

Downloads day by day using dukascopy-node (npx), parses CSV,
and bulk-inserts into the ticks hypertable.

Usage:
    uv run python -m scripts.seed_ticks_dukascopy \
        --instrument XAUUSD --start 2024-04-01 --end 2026-04-08

Requirements:
    - Node.js + npx (for dukascopy-node)
    - TimescaleDB with ticks table created
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import subprocess
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

from src.common.config import load_config

# Dukascopy instrument names
INSTRUMENT_MAP: dict[str, str] = {
    "XAUUSD": "xauusd",
    "EUR/USD": "eurusd",
    "GBP/USD": "gbpusd",
}

BATCH_SIZE = 10000


def _download_day(
    duka_instrument: str,
    date: datetime,
    output_dir: Path,
    timeout: int = 120,
) -> Path | None:
    """Download one day of tick data via dukascopy-node.

    Returns path to CSV file, or None if download failed.
    """
    date_str = date.strftime("%Y-%m-%dT00:00:00")
    next_day = (date + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00")
    out_file = output_dir / f"{duka_instrument}-{date.strftime('%Y%m%d')}.csv"

    try:
        result = subprocess.run(
            [
                "npx", "--yes", "dukascopy-node",
                "-i", duka_instrument,
                "-from", date_str,
                "-to", next_day,
                "-t", "tick",
                "-f", "csv",
                "-dir", str(output_dir),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Find the generated CSV file
        csv_files = list(output_dir.glob(f"{duka_instrument}-tick-*.csv"))
        if csv_files:
            return csv_files[0]

        if "No data" in result.stdout or "0 B" in result.stdout:
            return None

        return None

    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT downloading {date.date()}")
        return None
    except Exception as e:
        print(f"    ERROR downloading {date.date()}: {e}")
        return None


def _parse_ticks(
    csv_path: Path,
    instrument: str,
) -> list[tuple[datetime, str, float, float]]:
    """Parse Dukascopy tick CSV into DB-ready tuples."""
    rows: list[tuple[datetime, str, float, float]] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("timestamp") or not row.get("bidPrice"):
                continue
            try:
                ts_ms = int(row["timestamp"])
                dt = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
                bid = float(row["bidPrice"])
                ask = float(row["askPrice"])
                rows.append((dt, instrument, bid, ask))
            except (ValueError, KeyError):
                continue
    return rows


async def _insert_ticks(
    conn: asyncpg.Connection,
    ticks: list[tuple[datetime, str, float, float]],
) -> int:
    """Bulk insert ticks into the database."""
    if not ticks:
        return 0

    total = 0
    for i in range(0, len(ticks), BATCH_SIZE):
        batch = ticks[i : i + BATCH_SIZE]
        await conn.executemany(
            """INSERT INTO ticks (time, instrument, bid, ask)
               VALUES ($1, $2, $3, $4)
               ON CONFLICT DO NOTHING""",
            batch,
        )
        total += len(batch)
    return total


async def seed(
    instrument: str,
    start: datetime,
    end: datetime,
) -> None:
    """Download and ingest tick data day by day."""
    load_dotenv()
    config = load_config()

    duka_name = INSTRUMENT_MAP.get(instrument)
    if not duka_name:
        print(f"Unknown instrument: {instrument}")
        print(f"Available: {', '.join(INSTRUMENT_MAP.keys())}")
        return

    conn = await asyncpg.connect(config.database.url)

    # Check existing data range
    existing = await conn.fetchrow(
        "SELECT MIN(time), MAX(time), COUNT(*) FROM ticks WHERE instrument = $1",
        instrument,
    )
    if existing and existing[2] > 0:
        print(f"Existing ticks for {instrument}: {existing[2]:,} "
              f"({existing[0].date()} → {existing[1].date()})")

    total_days = (end - start).days
    total_inserted = 0
    skipped = 0

    print(f"\nDownloading {instrument} ticks: {start.date()} → {end.date()} ({total_days} days)")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        current = start
        day_num = 0
        while current < end:
            day_num += 1

            # Skip weekends
            if current.weekday() >= 5:
                current += timedelta(days=1)
                skipped += 1
                continue

            # Check if we already have data for this day
            count = await conn.fetchval(
                """SELECT COUNT(*) FROM ticks
                   WHERE instrument = $1
                   AND time >= $2 AND time < $3""",
                instrument,
                current,
                current + timedelta(days=1),
            )
            if count and count > 100:
                current += timedelta(days=1)
                skipped += 1
                continue

            print(f"  [{day_num}/{total_days}] {current.date()} ...", end="", flush=True)

            csv_path = _download_day(duka_name, current, tmp_path)
            if csv_path is None:
                print(" no data (holiday?)")
                current += timedelta(days=1)
                continue

            ticks = _parse_ticks(csv_path, instrument)
            inserted = await _insert_ticks(conn, ticks)
            total_inserted += inserted

            # Cleanup CSV
            csv_path.unlink(missing_ok=True)

            print(f" {len(ticks):,} ticks → {inserted:,} inserted")

            current += timedelta(days=1)

    # Final stats
    final_count = await conn.fetchval(
        "SELECT COUNT(*) FROM ticks WHERE instrument = $1",
        instrument,
    )
    print(f"\nDone. Inserted: {total_inserted:,} ticks, "
          f"skipped: {skipped} days, total in DB: {final_count:,}")

    await conn.close()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download Dukascopy tick data into TimescaleDB",
    )
    parser.add_argument("--instrument", type=str, default="XAUUSD")
    parser.add_argument("--start", type=str, required=True,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True,
                        help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)

    asyncio.run(seed(args.instrument, start, end))


if __name__ == "__main__":
    main()
