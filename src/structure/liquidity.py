"""Liquidity pool detection.

Identifies areas where stop-losses and pending orders likely cluster:
- Equal highs/lows (swing highs/lows that form at the same price level)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from datetime import datetime


class LiquidityType(StrEnum):
    """Type of liquidity pool."""

    EQUAL_HIGHS = "equal_highs"
    EQUAL_LOWS = "equal_lows"


@dataclass(frozen=True, slots=True)
class LiquidityPool:
    """A detected liquidity pool."""

    time: datetime
    price: float
    liquidity_type: LiquidityType
    touch_count: int
    index: int


def detect_liquidity_vectorized(
    df: pl.DataFrame,
    tolerance_pct: float = 0.02,
    lookback: int = 50,
    min_touches: int = 2,
    swings: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Detect liquidity pools (equal swing highs/lows).

    Compares swing points against other swing points of the same type
    within a lookback window. Only swing highs are compared to swing
    highs, and swing lows to swing lows.

    Args:
        df: DataFrame with columns: time, high, low (used as fallback).
        tolerance_pct: Percentage tolerance for considering levels "equal".
        lookback: Number of preceding swing points to compare against.
        min_touches: Minimum number of touches to form a liquidity pool.
        swings: Pre-computed swings DataFrame with columns:
                time, price, swing_type, index.
                If None or empty, returns an empty DataFrame.

    Returns:
        DataFrame with columns: time, price, liquidity_type, touch_count, index.
    """
    if swings is None or swings.is_empty():
        return _empty_liq_df()

    pools: list[dict[str, object]] = []

    for swing_type, liq_type in [
        ("swing_high", LiquidityType.EQUAL_HIGHS),
        ("swing_low", LiquidityType.EQUAL_LOWS),
    ]:
        typed = swings.filter(pl.col("swing_type") == swing_type)
        if typed.is_empty():
            continue

        prices = typed["price"].to_numpy()
        times = typed["time"]
        indices = typed["index"].to_numpy()

        for i in range(1, len(prices)):
            start = max(0, i - lookback)
            window_prices = prices[start:i]

            current = prices[i]
            tolerance = current * tolerance_pct / 100
            matches = int(np.sum(np.abs(window_prices - current) <= tolerance))

            if matches + 1 >= min_touches:
                pools.append({
                    "time": times[i],
                    "price": float(current),
                    "liquidity_type": liq_type,
                    "touch_count": matches + 1,  # include current swing
                    "index": int(indices[i]),
                })

    if not pools:
        return _empty_liq_df()

    return pl.DataFrame(pools, schema=_LIQ_SCHEMA)


@dataclass
class LiquidityState:
    """Incremental state for liquidity detection."""

    recent_swing_highs: list[float] = field(default_factory=list)
    recent_swing_lows: list[float] = field(default_factory=list)
    recent_high_times: list[datetime] = field(default_factory=list)
    recent_low_times: list[datetime] = field(default_factory=list)
    recent_high_indices: list[int] = field(default_factory=list)
    recent_low_indices: list[int] = field(default_factory=list)
    lookback: int = 50
    tolerance_pct: float = 0.02
    min_touches: int = 2


def detect_liquidity_incremental(
    swing: dict[str, object] | None,
    state: LiquidityState,
) -> list[LiquidityPool]:
    """Detect liquidity pools incrementally from swing points.

    Args:
        swing: A detected swing point dict with keys: time, price,
               swing_type, index. Pass None if no swing on this candle.
        state: Mutable liquidity detection state.

    Returns:
        List of newly detected LiquidityPools.
    """
    if swing is None:
        return []

    price = float(swing["price"])  # type: ignore[arg-type]
    swing_type = str(swing["swing_type"])
    time = cast("datetime", swing["time"])
    index = int(swing["index"])  # type: ignore[call-overload]

    if swing_type == "swing_high":
        prices_list = state.recent_swing_highs
        times_list = state.recent_high_times
        indices_list = state.recent_high_indices
        liq_type = LiquidityType.EQUAL_HIGHS
    elif swing_type == "swing_low":
        prices_list = state.recent_swing_lows
        times_list = state.recent_low_times
        indices_list = state.recent_low_indices
        liq_type = LiquidityType.EQUAL_LOWS
    else:
        return []

    results: list[LiquidityPool] = []

    # Check against recent swings of the same type
    tolerance = price * state.tolerance_pct / 100
    matches = sum(1 for p in prices_list if abs(p - price) <= tolerance)
    if matches + 1 >= state.min_touches:
        results.append(LiquidityPool(
            time=time,
            price=price,
            liquidity_type=liq_type,
            touch_count=matches + 1,
            index=index,
        ))

    # Add to history and trim
    prices_list.append(price)
    times_list.append(time)
    indices_list.append(index)
    if len(prices_list) > state.lookback:
        del prices_list[0]
        del times_list[0]
        del indices_list[0]

    return results


_LIQ_SCHEMA: dict[str, Any] = {
    "time": pl.Datetime("us", "UTC"),
    "price": pl.Float64,
    "liquidity_type": pl.Utf8,
    "touch_count": pl.Int64,
    "index": pl.Int64,
}


def _empty_liq_df() -> pl.DataFrame:
    return pl.DataFrame(schema=_LIQ_SCHEMA)
