"""Retry failed days for tick data seeding.

Retries days that timed out during initial seed.
Uses longer timeout (300s) and purges partial data before re-downloading.

Usage:
    uv run python -m scripts.seed_ticks_retry \
        --instrument XAUUSD --start 2024-04-01 --end 2026-04-08 \
        --log retry_ticks.log
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

from scripts.seed_ticks_dukascopy import (
    INSTRUMENT_MAP,
    _download_day,
    _insert_ticks,
    _parse_ticks,
)
from src.common.config import load_config

# Days that timed out during initial seed
TIMEOUT_DAYS = [
    "2024-05-30",
    "2024-07-16",
    "2025-01-09",
    "2025-03-28",
    "2025-03-31",
    "2025-12-09",
    "2025-12-16",
    "2026-02-23",
    "2026-03-02",
]

RETRY_TIMEOUT = 300


def _get_retry_dates(start: datetime, end: datetime) -> list[datetime]:
    """Build list of timeout dates to retry."""
    dates: list[datetime] = []

    for d in TIMEOUT_DAYS:
        dt = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=UTC)
        if start <= dt < end:
            dates.append(dt)

    return sorted(dates)


async def retry_seed(
    instrument: str,
    start: datetime,
    end: datetime,
    log_file: str,
) -> None:
    """Retry downloading tick data for failed/skipped days."""
    load_dotenv()
    config = load_config()

    duka_name = INSTRUMENT_MAP.get(instrument)
    if not duka_name:
        print(f"Unknown instrument: {instrument}")
        return

    conn = await asyncpg.connect(config.database.url)

    dates = _get_retry_dates(start, end)
    total = len(dates)
    total_inserted = 0
    skipped = 0

    print(f"Retrying {total} timeout days ({instrument})")

    with open(log_file, "w") as log:
        log.write(f"Retry seed {instrument}: {start.date()} -> {end.date()}\n")
        log.write(f"Total days to retry: {total}\n\n")
        log.flush()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            for i, date in enumerate(dates, 1):
                # Check if we already have data for this day
                count = await conn.fetchval(
                    """SELECT COUNT(*) FROM ticks
                       WHERE instrument = $1
                       AND time >= $2 AND time < $3""",
                    instrument,
                    date,
                    date + timedelta(days=1),
                )
                # A normal weekday has ~200k ticks; skip only if substantially complete
                if count and count > 50000:
                    msg = f"[{i}/{total}] {date.date()} (TIMEOUT) — already {count:,} ticks, skip"
                    print(msg)
                    log.write(msg + "\n")
                    log.flush()
                    skipped += 1
                    continue

                msg_prefix = f"[{i}/{total}] {date.date()} (TIMEOUT)"
                print(f"  {msg_prefix} ...", end="", flush=True)

                # Delete partial data before re-downloading
                if count and count > 0:
                    await conn.execute(
                        """DELETE FROM ticks
                           WHERE instrument = $1
                           AND time >= $2 AND time < $3""",
                        instrument,
                        date,
                        date + timedelta(days=1),
                    )
                    print(f" (purged {count:,} partial)", end="", flush=True)

                csv_path = _download_day(duka_name, date, tmp_path, timeout=RETRY_TIMEOUT)

                if csv_path is None:
                    msg = f"  {msg_prefix} — no data"
                    print(" no data")
                    log.write(msg + "\n")
                    log.flush()
                    continue

                ticks = _parse_ticks(csv_path, instrument)
                inserted = await _insert_ticks(conn, ticks)
                total_inserted += inserted

                csv_path.unlink(missing_ok=True)

                msg = f"  {msg_prefix} — {len(ticks):,} ticks -> {inserted:,} inserted"
                print(f" {len(ticks):,} ticks -> {inserted:,} inserted")
                log.write(msg + "\n")
                log.flush()

        # Final stats
        final_count = await conn.fetchval(
            "SELECT COUNT(*) FROM ticks WHERE instrument = $1",
            instrument,
        )
        summary = (
            f"\nDone. Inserted: {total_inserted:,}, "
            f"skipped: {skipped}, total in DB: {final_count:,}"
        )
        print(summary)
        log.write(summary + "\n")

    await conn.close()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Retry failed tick data downloads",
    )
    parser.add_argument("--instrument", type=str, default="XAUUSD")
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--log", type=str, default="retry_ticks.log")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)

    asyncio.run(retry_seed(args.instrument, start, end, args.log))


if __name__ == "__main__":
    main()
