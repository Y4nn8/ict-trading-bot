"""Aggregate 10-second Parquet candles to 1-minute candles.

Reads 10s candles from a directory, aggregates to M1, and writes
output Parquet files partitioned by month.

Usage:
    uv run python -m scripts.aggregate_to_m1 \
        --input-dir data/morpheus/xauusd \
        --output-dir data/morpheus/xauusd_m1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from src.common.logging import get_logger
from src.morpheus.dataset import load_parquet_dir

logger = get_logger(__name__)


def aggregate_to_m1(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate 10s candles to M1 candles.

    Args:
        df: DataFrame with columns [time, open, high, low, close,
            tick_count, spread], sorted by time.

    Returns:
        M1 DataFrame with same columns.
    """
    return (
        df.group_by_dynamic("time", every="1m")
        .agg(
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("tick_count").sum(),
            pl.col("spread").mean(),
        )
        .sort("time")
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Aggregate 10s candles to M1")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading_10s", input_dir=str(args.input_dir))
    df = load_parquet_dir(args.input_dir)
    logger.info("loaded", rows=df.height)

    m1 = aggregate_to_m1(df)
    logger.info("aggregated_to_m1", rows=m1.height)

    # Partition by month
    m1 = m1.with_columns(
        pl.col("time").dt.strftime("%Y-%m").alias("_month"),
    )
    for month, group in m1.group_by("_month"):
        month_str = month[0]
        out_path = args.output_dir / f"{month_str}.parquet"
        group.drop("_month").write_parquet(out_path)
        logger.info("wrote", path=str(out_path), rows=group.height)

    logger.info("done", total_m1_rows=m1.height)


if __name__ == "__main__":
    main(sys.argv[1:])
