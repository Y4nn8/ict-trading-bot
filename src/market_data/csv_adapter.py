"""CSV file adapter for importing historical market data.

Supports multiple CSV formats:
- Generic: time,open,high,low,close,volume
- FXCM: date,bidopen,bidhigh,bidlow,bidclose,askopen,askhigh,asklow,askclose,tickqty
- MetaTrader: date,time,open,high,low,close,tickvol,vol,spread
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

import polars as pl

from src.common.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)


class CSVFormat(StrEnum):
    """Supported CSV format types."""

    GENERIC = "generic"
    FXCM = "fxcm"
    METATRADER = "metatrader"


def load_csv(
    path: Path,
    csv_format: CSVFormat = CSVFormat.GENERIC,
    timezone: str = "UTC",
) -> pl.DataFrame:
    """Load OHLCV data from a CSV file.

    Args:
        path: Path to the CSV file.
        csv_format: Format of the CSV file.
        timezone: Timezone of the timestamps.

    Returns:
        DataFrame with columns: time, open, high, low, close, volume.
    """
    if csv_format == CSVFormat.FXCM:
        return _load_fxcm(path, timezone)
    if csv_format == CSVFormat.METATRADER:
        return _load_metatrader(path, timezone)
    return _load_generic(path, timezone)


def _load_generic(path: Path, timezone: str) -> pl.DataFrame:
    """Load generic CSV: time,open,high,low,close,volume."""
    df = pl.read_csv(path, try_parse_dates=True)

    # Normalize column names to lowercase
    df = df.rename({c: c.lower().strip() for c in df.columns})

    # Find time column
    time_col = _find_column(df, ["time", "date", "datetime", "timestamp"])
    if time_col is None:
        msg = f"No time column found in {path}. Columns: {df.columns}"
        raise ValueError(msg)

    # Cast time column
    if df[time_col].dtype == pl.Utf8:
        df = df.with_columns(pl.col(time_col).str.to_datetime().alias("time"))
    else:
        df = df.with_columns(pl.col(time_col).alias("time"))

    # Ensure UTC
    if df["time"].dtype == pl.Datetime:
        df = df.with_columns(pl.col("time").dt.replace_time_zone(timezone))

    # Select and rename standard columns
    col_map = {
        "open": _find_column(df, ["open", "o"]),
        "high": _find_column(df, ["high", "h"]),
        "low": _find_column(df, ["low", "l"]),
        "close": _find_column(df, ["close", "c"]),
        "volume": _find_column(df, ["volume", "vol", "v", "tickqty"]),
    }

    select_exprs = [pl.col("time")]
    for target, source in col_map.items():
        if source and source in df.columns:
            select_exprs.append(pl.col(source).cast(pl.Float64).alias(target))
        elif target == "volume":
            select_exprs.append(pl.lit(0.0).alias(target))
        else:
            msg = f"Missing required column '{target}' in {path}"
            raise ValueError(msg)

    result = df.select(select_exprs).sort("time")
    logger.info("csv_loaded", path=str(path), format="generic", rows=len(result))
    return result


def _load_fxcm(path: Path, timezone: str) -> pl.DataFrame:
    """Load FXCM format CSV.

    FXCM columns: date,bidopen,bidhigh,bidlow,bidclose,askopen,...,tickqty
    """
    df = pl.read_csv(path, try_parse_dates=True)
    df = df.rename({c: c.lower().strip() for c in df.columns})

    # Find date column
    time_col = _find_column(df, ["date", "datetime", "time"])
    if time_col is None:
        msg = f"No time column found in {path}"
        raise ValueError(msg)

    if df[time_col].dtype == pl.Utf8:
        df = df.with_columns(pl.col(time_col).str.to_datetime().alias("time"))
    else:
        df = df.with_columns(pl.col(time_col).alias("time"))

    if df["time"].dtype == pl.Datetime:
        df = df.with_columns(pl.col("time").dt.replace_time_zone(timezone))

    # Use bid prices (standard for forex analysis)
    result = df.select(
        pl.col("time"),
        pl.col("bidopen").cast(pl.Float64).alias("open"),
        pl.col("bidhigh").cast(pl.Float64).alias("high"),
        pl.col("bidlow").cast(pl.Float64).alias("low"),
        pl.col("bidclose").cast(pl.Float64).alias("close"),
        (
            pl.col("tickqty").cast(pl.Float64).alias("volume")
            if "tickqty" in df.columns
            else pl.lit(0.0).alias("volume")
        ),
    ).sort("time")

    logger.info("csv_loaded", path=str(path), format="fxcm", rows=len(result))
    return result


def _load_metatrader(path: Path, timezone: str) -> pl.DataFrame:
    """Load MetaTrader export CSV.

    MT columns: date,time,open,high,low,close,tickvol,vol,spread
    or single datetime column.
    """
    df = pl.read_csv(path, try_parse_dates=True)
    df = df.rename({c: c.lower().strip() for c in df.columns})

    # MT sometimes has separate date and time columns
    if "date" in df.columns and "time" in df.columns and df["date"].dtype == pl.Utf8:
        df = df.with_columns(
            (pl.col("date") + pl.lit(" ") + pl.col("time"))
            .str.to_datetime()
            .dt.replace_time_zone(timezone)
            .alias("time")
        )
    elif "date" in df.columns:
        if df["date"].dtype == pl.Utf8:
            df = df.with_columns(
                pl.col("date").str.to_datetime().dt.replace_time_zone(timezone).alias("time")
            )
        else:
            df = df.with_columns(pl.col("date").alias("time"))

    vol_col = _find_column(df, ["tickvol", "vol", "volume"])

    result = df.select(
        pl.col("time"),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        (
            pl.col(vol_col).cast(pl.Float64).alias("volume")
            if vol_col
            else pl.lit(0.0).alias("volume")
        ),
    ).sort("time")

    logger.info("csv_loaded", path=str(path), format="metatrader", rows=len(result))
    return result


def _find_column(df: pl.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column name from candidates."""
    columns_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in columns_lower:
            return columns_lower[candidate]
    return None
