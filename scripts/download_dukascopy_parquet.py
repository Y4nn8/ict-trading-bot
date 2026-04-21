"""Download Dukascopy tick data and aggregate directly to Parquet.

Bypasses the DB entirely — downloads ticks via dukascopy-node,
aggregates into candles with Polars, and writes monthly Parquet files.

Usage:
    uv run python -m scripts.download_dukascopy_parquet \
        --instrument XAUUSD --start 2016-04-01 --end 2024-04-01 \
        --output-dir data/morpheus/xauusd

    # Custom bucket size:
    uv run python -m scripts.download_dukascopy_parquet \
        --instrument XAUUSD --start 2020-01-01 --end 2024-04-01 \
        --bucket-seconds 30 --output-dir data/morpheus/xauusd_30s

Requirements:
    - Node.js + npx (for dukascopy-node)
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

INSTRUMENT_MAP: dict[str, str] = {
    "XAUUSD": "xauusd",
    "EURUSD": "eurusd",
    "GBPUSD": "gbpusd",
    "USDJPY": "usdjpy",
    "LIGHTCMDUSD": "lightcmdusd",
}


def download_day(
    duka_instrument: str,
    date: datetime,
    output_dir: Path,
    timeout: int = 120,
    retries: int = 3,
) -> Path | None:
    """Download one day of tick data via dukascopy-node."""
    date_str = date.strftime("%Y-%m-%dT00:00:00")
    next_day = (date + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00")

    for attempt in range(retries):
        try:
            subprocess.run(
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
            csv_files = list(
                output_dir.glob(f"{duka_instrument}-tick-*.csv"),
            )
            if csv_files:
                return csv_files[0]
        except subprocess.TimeoutExpired:
            if attempt < retries - 1:
                print(f" timeout, retry {attempt + 2}/{retries}...", end="")
                time.sleep(5)
                continue
        except OSError:
            break
    return None


def parse_tick_csv(csv_path: Path) -> pl.DataFrame:
    """Parse Dukascopy tick CSV into a Polars DataFrame."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("timestamp") or not row.get("bidPrice"):
                continue
            try:
                ts_ms = int(row["timestamp"])
                bid = float(row["bidPrice"])
                ask = float(row["askPrice"])
                rows.append((ts_ms, bid, ask))
            except (ValueError, KeyError):
                continue

    if not rows:
        return pl.DataFrame(
            schema={"time": pl.Datetime("us", "UTC"), "bid": pl.Float64, "ask": pl.Float64},
        )

    return pl.DataFrame(
        {
            "time": [datetime.fromtimestamp(r[0] / 1000, tz=UTC) for r in rows],
            "bid": [r[1] for r in rows],
            "ask": [r[2] for r in rows],
        },
    )


def aggregate_to_candles(ticks: pl.DataFrame, bucket_seconds: int) -> pl.DataFrame:
    """Aggregate ticks into OHLC candles."""
    if ticks.is_empty():
        return pl.DataFrame(
            schema={
                "time": pl.Datetime("us", "UTC"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "tick_count": pl.Int32,
                "spread": pl.Float64,
            },
        )

    mid = (pl.col("bid") + pl.col("ask")) / 2.0

    return (
        ticks.sort("time")
        .group_by_dynamic("time", every=f"{bucket_seconds}s")
        .agg(
            mid.first().alias("open"),
            mid.max().alias("high"),
            mid.min().alias("low"),
            mid.last().alias("close"),
            pl.len().cast(pl.Int32).alias("tick_count"),
            (pl.col("ask").last() - pl.col("bid").last()).alias("spread"),
        )
        .sort("time")
    )


def month_key(dt: datetime) -> str:
    """Return 'YYYY-MM' string for a datetime."""
    return f"{dt.year:04d}-{dt.month:02d}"


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download Dukascopy ticks → candle Parquet (no DB)",
    )
    parser.add_argument("--instrument", default="XAUUSD")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--bucket-seconds", type=int, default=10)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    duka_name = INSTRUMENT_MAP.get(args.instrument)
    if not duka_name:
        print(f"Unknown instrument: {args.instrument}")
        print(f"Available: {', '.join(INSTRUMENT_MAP)}")
        sys.exit(1)

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Skip months that already have parquet files
    existing = {p.stem for p in args.output_dir.glob("*.parquet")}

    current = start
    month_ticks: dict[str, list[pl.DataFrame]] = {}
    total_days = (end - start).days
    day_num = 0

    print(f"Downloading {args.instrument}: {start.date()} → {end.date()}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        while current < end:
            day_num += 1
            mk = month_key(current)

            if mk in existing:
                current += timedelta(days=1)
                continue

            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue

            print(
                f"  [{day_num}/{total_days}] {current.date()} ...",
                end="", flush=True,
            )

            csv_path = download_day(duka_name, current, tmp_path)
            if csv_path is None:
                print(" no data")
                current += timedelta(days=1)
                continue

            ticks = parse_tick_csv(csv_path)
            csv_path.unlink(missing_ok=True)

            if ticks.is_empty():
                print(" empty")
                current += timedelta(days=1)
                continue

            print(f" {len(ticks):,} ticks")

            if mk not in month_ticks:
                month_ticks[mk] = []
            month_ticks[mk].append(ticks)

            # Write completed months as we go
            next_day = current + timedelta(days=1)
            next_mk = month_key(next_day)
            if next_mk != mk and mk in month_ticks:
                all_ticks = pl.concat(month_ticks.pop(mk))
                candles = aggregate_to_candles(all_ticks, args.bucket_seconds)
                out_path = args.output_dir / f"{mk}.parquet"
                candles.write_parquet(out_path)
                print(f"  → {mk}: {len(candles):,} candles written")

            current += timedelta(days=1)

    # Flush remaining month
    for mk, frames in month_ticks.items():
        all_ticks = pl.concat(frames)
        candles = aggregate_to_candles(all_ticks, args.bucket_seconds)
        out_path = args.output_dir / f"{mk}.parquet"
        candles.write_parquet(out_path)
        print(f"  → {mk}: {len(candles):,} candles written")

    total_files = len(list(args.output_dir.glob("*.parquet")))
    print(f"\nDone. {total_files} parquet files in {args.output_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
