"""Download free historical M5 data from FXCM.

FXCM provides free historical tick and candle data via their public servers.
No API key required.

Usage:
    uv run python -m scripts.download_fxcm --instruments EURUSD,GBPUSD --weeks 26
    uv run python -m scripts.download_fxcm --all --weeks 12
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiohttp

FXCM_BASE_URL = "https://tickdata.fxcorporate.com"

# FXCM instrument symbols
FXCM_INSTRUMENTS: dict[str, str] = {
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "USDJPY": "USDJPY",
    "XAUUSD": "XAUUSD",
    "SPX500": "SPX500",
    "US30": "US30",      # Dow Jones
    "GER40": "GER40",    # DAX
    "JPN225": "JPN225",  # Nikkei
}

# Map our instrument names to FXCM symbols
OUR_TO_FXCM: dict[str, str] = {
    "EUR/USD": "EURUSD",
    "GBP/USD": "GBPUSD",
    "XAUUSD": "XAUUSD",
    "SPX500": "SPX500",
    "DOW30": "US30",
    "DAX40": "GER40",
    "NIKKEI225": "JPN225",
}


async def download_instrument(
    session: aiohttp.ClientSession,
    fxcm_symbol: str,
    year: int,
    week: int,
    output_dir: Path,
) -> Path | None:
    """Download a single week of M5 data from FXCM.

    Args:
        session: HTTP session.
        fxcm_symbol: FXCM instrument symbol.
        year: Year.
        week: Week number (1-52).
        output_dir: Directory to save files.

    Returns:
        Path to downloaded file, or None if not available.
    """
    url = f"{FXCM_BASE_URL}/{fxcm_symbol}/{year}/{week}.csv.gz"

    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                return None

            compressed = await resp.read()
            csv_data = gzip.decompress(compressed).decode("utf-8")

            out_file = output_dir / f"{fxcm_symbol}_{year}_W{week:02d}.csv"
            out_file.write_text(csv_data)
            return out_file
    except Exception as e:
        print(f"  Error downloading {fxcm_symbol} {year}/W{week}: {e}")
        return None


async def download_all(
    instruments: list[str],
    weeks: int,
    output_dir: Path,
) -> None:
    """Download historical data for multiple instruments.

    Args:
        instruments: List of our instrument names.
        weeks: Number of weeks of history.
        output_dir: Directory to save files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(tz=UTC)
    total_files = 0

    async with aiohttp.ClientSession() as session:
        for instrument in instruments:
            fxcm_sym = OUR_TO_FXCM.get(instrument)
            if not fxcm_sym:
                print(f"  Skipping {instrument}: no FXCM mapping")
                continue

            print(f"Downloading {instrument} ({fxcm_sym})...")
            instrument_dir = output_dir / fxcm_sym
            instrument_dir.mkdir(exist_ok=True)

            for w in range(weeks):
                target_date = now - timedelta(weeks=w)
                year = target_date.isocalendar()[0]
                week = target_date.isocalendar()[1]

                result = await download_instrument(
                    session, fxcm_sym, year, week, instrument_dir
                )
                if result:
                    print(f"  ✓ {year}/W{week:02d}")
                    total_files += 1
                else:
                    print(f"  ✗ {year}/W{week:02d} (not available)")

    print(f"\nDone. {total_files} files downloaded to {output_dir}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Download free FXCM historical data")
    parser.add_argument(
        "--instruments",
        type=str,
        default=None,
        help="Comma-separated instrument names (default: all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available instruments",
    )
    parser.add_argument(
        "--weeks",
        type=int,
        default=26,
        help="Number of weeks of history (default: 26 = ~6 months)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/fxcm",
        help="Output directory (default: data/fxcm)",
    )

    args = parser.parse_args()

    if args.all:
        instruments = list(OUR_TO_FXCM.keys())
    elif args.instruments:
        instruments = [i.strip() for i in args.instruments.split(",")]
    else:
        instruments = list(OUR_TO_FXCM.keys())

    asyncio.run(download_all(instruments, args.weeks, Path(args.output)))


if __name__ == "__main__":
    main()
