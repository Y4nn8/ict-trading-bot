"""Liquidity pool detection.

Identifies areas where stop-losses and pending orders likely cluster:
- Equal highs/lows (double/triple tops/bottoms within tolerance)
- Session highs/lows (previous day high/low, weekly high/low)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

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
) -> pl.DataFrame:
    """Detect liquidity pools (equal highs/lows) on a full DataFrame.

    Scans for clusters of swing highs or swing lows that are within
    tolerance_pct of each other.

    Args:
        df: DataFrame with columns: time, high, low.
        tolerance_pct: Percentage tolerance for considering levels "equal".
        lookback: Number of bars to look back for matching levels.
        min_touches: Minimum number of touches to form a liquidity pool.

    Returns:
        DataFrame with columns: time, price, liquidity_type, touch_count, index.
    """
    if len(df) < lookback:
        return _empty_liq_df()

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    times = df["time"]

    pools: list[dict[str, object]] = []
    used_high_clusters: set[int] = set()
    used_low_clusters: set[int] = set()

    for i in range(lookback, len(highs)):
        window_highs = highs[i - lookback : i + 1]
        window_lows = lows[i - lookback : i + 1]

        # Check equal highs
        current_high = highs[i]
        tolerance = current_high * tolerance_pct / 100
        matches = np.where(np.abs(window_highs - current_high) <= tolerance)[0]
        if len(matches) >= min_touches and i not in used_high_clusters:
            pools.append({
                "time": times[i],
                "price": float(current_high),
                "liquidity_type": LiquidityType.EQUAL_HIGHS,
                "touch_count": len(matches),
                "index": i,
            })
            used_high_clusters.add(i)

        # Check equal lows
        current_low = lows[i]
        tolerance = current_low * tolerance_pct / 100
        matches = np.where(np.abs(window_lows - current_low) <= tolerance)[0]
        if len(matches) >= min_touches and i not in used_low_clusters:
            pools.append({
                "time": times[i],
                "price": float(current_low),
                "liquidity_type": LiquidityType.EQUAL_LOWS,
                "touch_count": len(matches),
                "index": i,
            })
            used_low_clusters.add(i)

    if not pools:
        return _empty_liq_df()

    return pl.DataFrame(pools, schema=_LIQ_SCHEMA)


@dataclass
class LiquidityState:
    """Incremental state for liquidity detection."""

    recent_highs: list[float] = field(default_factory=list)
    recent_lows: list[float] = field(default_factory=list)
    recent_times: list[datetime] = field(default_factory=list)
    recent_indices: list[int] = field(default_factory=list)
    lookback: int = 50
    tolerance_pct: float = 0.02
    min_touches: int = 2
    candle_count: int = 0


def detect_liquidity_incremental(
    candle: dict[str, object],
    state: LiquidityState,
) -> list[LiquidityPool]:
    """Detect liquidity pools incrementally.

    Args:
        candle: Dict with keys: time, high, low.
        state: Mutable liquidity detection state.

    Returns:
        List of newly detected LiquidityPools.
    """
    h = float(candle["high"])  # type: ignore[arg-type]
    lo = float(candle["low"])  # type: ignore[arg-type]

    state.recent_highs.append(h)
    state.recent_lows.append(lo)
    state.recent_times.append(candle["time"])  # type: ignore[arg-type]
    state.recent_indices.append(state.candle_count)
    state.candle_count += 1

    # Trim to lookback
    if len(state.recent_highs) > state.lookback + 1:
        state.recent_highs = state.recent_highs[-(state.lookback + 1) :]
        state.recent_lows = state.recent_lows[-(state.lookback + 1) :]
        state.recent_times = state.recent_times[-(state.lookback + 1) :]
        state.recent_indices = state.recent_indices[-(state.lookback + 1) :]

    if len(state.recent_highs) < state.lookback:
        return []

    results: list[LiquidityPool] = []

    # Check equal highs
    tol_h = h * state.tolerance_pct / 100
    matches_h = sum(1 for rh in state.recent_highs if abs(rh - h) <= tol_h)
    if matches_h >= state.min_touches:
        results.append(LiquidityPool(
            time=state.recent_times[-1],
            price=h,
            liquidity_type=LiquidityType.EQUAL_HIGHS,
            touch_count=matches_h,
            index=state.recent_indices[-1],
        ))

    # Check equal lows
    tol_l = lo * state.tolerance_pct / 100
    matches_l = sum(1 for rl in state.recent_lows if abs(rl - lo) <= tol_l)
    if matches_l >= state.min_touches:
        results.append(LiquidityPool(
            time=state.recent_times[-1],
            price=lo,
            liquidity_type=LiquidityType.EQUAL_LOWS,
            touch_count=matches_l,
            index=state.recent_indices[-1],
        ))

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
