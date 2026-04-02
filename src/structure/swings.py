"""Swing high/low detection using Williams fractals.

A swing high is a candle whose high is higher than the N candles on each side.
A swing low is a candle whose low is lower than the N candles on each side.
Default N=2 (5-bar Williams fractal).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from datetime import datetime


class SwingType(StrEnum):
    """Type of swing point."""

    HIGH = "swing_high"
    LOW = "swing_low"


@dataclass(frozen=True, slots=True)
class SwingPoint:
    """A detected swing point."""

    time: datetime
    price: float
    swing_type: SwingType
    index: int


def detect_swings_vectorized(
    df: pl.DataFrame,
    left_bars: int = 2,
    right_bars: int = 2,
) -> pl.DataFrame:
    """Detect swing highs and lows on a full DataFrame (vectorized).

    Uses Williams fractal logic: a swing high at bar i means
    high[i] >= max(high[i-left..i+right]).
    Similarly for swing low with min(low[...]).

    Args:
        df: DataFrame with columns: time, high, low.
        left_bars: Number of bars to the left to compare.
        right_bars: Number of bars to the right to compare.

    Returns:
        DataFrame with columns: time, price, swing_type, index.
    """
    if len(df) < left_bars + right_bars + 1:
        return _empty_swing_df()

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    times = df["time"]

    swing_points: list[dict[str, object]] = []
    n = len(highs)

    for i in range(left_bars, n - right_bars):
        # Check swing high
        left_highs = highs[i - left_bars : i]
        right_highs = highs[i + 1 : i + right_bars + 1]
        if bool(np.all(highs[i] >= left_highs)) and bool(
            np.all(highs[i] >= right_highs)
        ):
            swing_points.append({
                "time": times[i],
                "price": float(highs[i]),
                "swing_type": SwingType.HIGH,
                "index": i,
            })

        # Check swing low
        left_lows = lows[i - left_bars : i]
        right_lows = lows[i + 1 : i + right_bars + 1]
        if bool(np.all(lows[i] <= left_lows)) and bool(
            np.all(lows[i] <= right_lows)
        ):
            swing_points.append({
                "time": times[i],
                "price": float(lows[i]),
                "swing_type": SwingType.LOW,
                "index": i,
            })

    if not swing_points:
        return _empty_swing_df()

    return pl.DataFrame(swing_points, schema=_SWING_SCHEMA)


@dataclass
class SwingState:
    """Incremental state for swing detection.

    Maintains a buffer of recent candles to detect swings
    as new candles arrive.
    """

    left_bars: int = 2
    right_bars: int = 2
    buffer_highs: list[float] | None = None
    buffer_lows: list[float] | None = None
    buffer_times: list[datetime] | None = None
    buffer_indices: list[int] | None = None
    candle_count: int = 0

    def __post_init__(self) -> None:
        if self.buffer_highs is None:
            self.buffer_highs = []
            self.buffer_lows = []
            self.buffer_times = []
            self.buffer_indices = []


def detect_swings_incremental(
    candle: dict[str, object],
    state: SwingState,
) -> list[SwingPoint]:
    """Detect swings incrementally as new candles arrive.

    The swing at position i can only be confirmed after right_bars
    candles have been seen after it. This function returns any newly
    confirmed swing points.

    Args:
        candle: Dict with keys: time, high, low.
        state: Mutable swing detection state.

    Returns:
        List of newly confirmed SwingPoints (usually 0 or 1).
    """
    assert state.buffer_highs is not None
    assert state.buffer_lows is not None
    assert state.buffer_times is not None
    assert state.buffer_indices is not None
    state.buffer_highs.append(float(candle["high"]))  # type: ignore[arg-type]
    state.buffer_lows.append(float(candle["low"]))  # type: ignore[arg-type]
    state.buffer_times.append(candle["time"])  # type: ignore[arg-type]
    state.buffer_indices.append(state.candle_count)
    state.candle_count += 1

    window_size = state.left_bars + state.right_bars + 1
    results: list[SwingPoint] = []

    if len(state.buffer_highs) < window_size:
        return results

    # Check the candidate at position left_bars from the end of the window
    candidate_idx = len(state.buffer_highs) - state.right_bars - 1
    h = state.buffer_highs[candidate_idx]
    lo = state.buffer_lows[candidate_idx]

    left_start = candidate_idx - state.left_bars
    left_highs = state.buffer_highs[left_start:candidate_idx]
    right_highs = state.buffer_highs[candidate_idx + 1 : candidate_idx + state.right_bars + 1]

    if all(h >= lh for lh in left_highs) and all(h >= rh for rh in right_highs):
        results.append(SwingPoint(
            time=state.buffer_times[candidate_idx],
            price=h,
            swing_type=SwingType.HIGH,
            index=state.buffer_indices[candidate_idx],
        ))

    left_lows = state.buffer_lows[left_start:candidate_idx]
    right_lows = state.buffer_lows[candidate_idx + 1 : candidate_idx + state.right_bars + 1]

    if all(lo <= ll for ll in left_lows) and all(lo <= rl for rl in right_lows):
        results.append(SwingPoint(
            time=state.buffer_times[candidate_idx],
            price=lo,
            swing_type=SwingType.LOW,
            index=state.buffer_indices[candidate_idx],
        ))

    # Trim buffer to keep only what's needed
    max_buffer = window_size + state.left_bars
    if len(state.buffer_highs) > max_buffer:
        trim = len(state.buffer_highs) - max_buffer
        state.buffer_highs = state.buffer_highs[trim:]
        state.buffer_lows = state.buffer_lows[trim:]
        state.buffer_times = state.buffer_times[trim:]
        state.buffer_indices = state.buffer_indices[trim:]

    return results


_SWING_SCHEMA: dict[str, Any] = {
    "time": pl.Datetime("us", "UTC"),
    "price": pl.Float64,
    "swing_type": pl.Utf8,
    "index": pl.Int64,
}


def _empty_swing_df() -> pl.DataFrame:
    return pl.DataFrame(schema=_SWING_SCHEMA)
