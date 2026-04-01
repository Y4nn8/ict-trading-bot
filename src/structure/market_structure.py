"""Market structure detection: Break of Structure (BOS) and Change of Character (CHoCH).

BOS = price breaks a swing in the direction of the current trend (trend continuation).
CHoCH = price breaks a swing against the current trend (trend reversal signal).

Requires swing points as input (from swings.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import polars as pl

from src.structure.swings import SwingPoint, SwingType, detect_swings_vectorized

if TYPE_CHECKING:
    from datetime import datetime


class Trend(StrEnum):
    """Market trend state."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    UNDEFINED = "undefined"


class MSBreakType(StrEnum):
    """Type of market structure break."""

    BOS = "BOS"
    CHOCH = "CHoCH"


@dataclass(frozen=True, slots=True)
class MSBreak:
    """A detected market structure break."""

    time: datetime
    break_type: MSBreakType
    direction: Trend
    broken_level: float
    swing_type: SwingType
    index: int


def detect_market_structure_vectorized(
    df: pl.DataFrame,
    left_bars: int = 2,
    right_bars: int = 2,
) -> tuple[pl.DataFrame, Trend]:
    """Detect BOS and CHoCH on a full DataFrame.

    Logic:
    1. Detect swings from the price data.
    2. Track the current trend (bullish/bearish/undefined).
    3. When price breaks above the last swing high:
       - If trend is bullish → BOS (continuation)
       - If trend is bearish → CHoCH (reversal to bullish)
    4. When price breaks below the last swing low:
       - If trend is bearish → BOS (continuation)
       - If trend is bullish → CHoCH (reversal to bearish)

    Args:
        df: DataFrame with columns: time, high, low, close.
        left_bars: Swing detection left bars.
        right_bars: Swing detection right bars.

    Returns:
        Tuple of (DataFrame of MSBreaks, final Trend).
    """
    swings_df = detect_swings_vectorized(df, left_bars, right_bars)
    if swings_df.is_empty():
        return _empty_ms_df(), Trend.UNDEFINED

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    times = df["time"]

    # Convert swings to list for iteration
    swing_records = swings_df.to_dicts()

    last_swing_high: float | None = None
    last_swing_low: float | None = None
    trend = Trend.UNDEFINED
    breaks: list[dict[str, object]] = []

    # Scan candles and check for breaks of the most recent swing levels
    swing_idx = 0

    for i in range(len(highs)):
        # Update swing levels as we reach confirmed swing points
        while swing_idx < len(swing_records):
            s = swing_records[swing_idx]
            s_idx = int(s["index"])
            # A swing at index s_idx is confirmed right_bars candles later
            confirmed_at = s_idx + right_bars
            if confirmed_at > i:
                break
            if s["swing_type"] == SwingType.HIGH:
                last_swing_high = float(s["price"])
            else:
                last_swing_low = float(s["price"])
            swing_idx += 1

        # Check for break above swing high
        if last_swing_high is not None and float(highs[i]) > last_swing_high:
            if trend == Trend.BULLISH or trend == Trend.UNDEFINED:
                break_type = MSBreakType.BOS
            else:
                break_type = MSBreakType.CHOCH

            breaks.append({
                "time": times[i],
                "break_type": break_type,
                "direction": Trend.BULLISH,
                "broken_level": last_swing_high,
                "swing_type": SwingType.HIGH,
                "index": i,
            })
            trend = Trend.BULLISH
            last_swing_high = None  # Consumed

        # Check for break below swing low
        if last_swing_low is not None and float(lows[i]) < last_swing_low:
            if trend == Trend.BEARISH or trend == Trend.UNDEFINED:
                break_type = MSBreakType.BOS
            else:
                break_type = MSBreakType.CHOCH

            breaks.append({
                "time": times[i],
                "break_type": break_type,
                "direction": Trend.BEARISH,
                "broken_level": last_swing_low,
                "swing_type": SwingType.LOW,
                "index": i,
            })
            trend = Trend.BEARISH
            last_swing_low = None  # Consumed

    if not breaks:
        return _empty_ms_df(), trend

    return pl.DataFrame(breaks, schema=_MS_SCHEMA), trend


@dataclass
class MarketStructureIncrState:
    """Incremental state for market structure detection."""

    trend: Trend = Trend.UNDEFINED
    last_swing_high: float | None = None
    last_swing_low: float | None = None
    pending_swings: list[SwingPoint] = field(default_factory=list)
    right_bars: int = 2
    candle_count: int = 0


def detect_market_structure_incremental(
    candle: dict[str, object],
    new_swings: list[SwingPoint],
    state: MarketStructureIncrState,
) -> list[MSBreak]:
    """Detect BOS/CHoCH incrementally.

    Args:
        candle: Dict with keys: time, high, low.
        new_swings: Newly confirmed swing points from the swing detector.
        state: Mutable market structure state.

    Returns:
        List of newly detected MSBreak events.
    """
    state.candle_count += 1
    i = state.candle_count - 1

    # Add new swings to pending (they become active after right_bars confirmation)
    for swing in new_swings:
        state.pending_swings.append(swing)

    # Activate pending swings that are now confirmed
    activated: list[SwingPoint] = []
    remaining: list[SwingPoint] = []
    for swing in state.pending_swings:
        confirmed_at = swing.index + state.right_bars
        if confirmed_at <= i:
            activated.append(swing)
        else:
            remaining.append(swing)
    state.pending_swings = remaining

    for swing in activated:
        if swing.swing_type == SwingType.HIGH:
            state.last_swing_high = swing.price
        else:
            state.last_swing_low = swing.price

    # Check for structure breaks
    results: list[MSBreak] = []
    high = float(candle["high"])  # type: ignore[arg-type]
    low = float(candle["low"])  # type: ignore[arg-type]
    time = candle["time"]

    if state.last_swing_high is not None and high > state.last_swing_high:
        if state.trend in (Trend.BULLISH, Trend.UNDEFINED):
            break_type = MSBreakType.BOS
        else:
            break_type = MSBreakType.CHOCH

        results.append(MSBreak(
            time=time,  # type: ignore[arg-type]
            break_type=break_type,
            direction=Trend.BULLISH,
            broken_level=state.last_swing_high,
            swing_type=SwingType.HIGH,
            index=i,
        ))
        state.trend = Trend.BULLISH
        state.last_swing_high = None

    if state.last_swing_low is not None and low < state.last_swing_low:
        if state.trend in (Trend.BEARISH, Trend.UNDEFINED):
            break_type = MSBreakType.BOS
        else:
            break_type = MSBreakType.CHOCH

        results.append(MSBreak(
            time=time,  # type: ignore[arg-type]
            break_type=break_type,
            direction=Trend.BEARISH,
            broken_level=state.last_swing_low,
            swing_type=SwingType.LOW,
            index=i,
        ))
        state.trend = Trend.BEARISH
        state.last_swing_low = None

    return results


_MS_SCHEMA: dict[str, Any] = {
    "time": pl.Datetime("us", "UTC"),
    "break_type": pl.Utf8,
    "direction": pl.Utf8,
    "broken_level": pl.Float64,
    "swing_type": pl.Utf8,
    "index": pl.Int64,
}


def _empty_ms_df() -> pl.DataFrame:
    return pl.DataFrame(schema=_MS_SCHEMA)
