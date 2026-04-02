"""Fair Value Gap (FVG) detection.

An FVG is an imbalance in price where the wicks of candles 1 and 3
don't overlap, leaving a gap filled only by candle 2's body/wick.

Bullish FVG: candle1.high < candle3.low (gap up)
Bearish FVG: candle1.low > candle3.high (gap down)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from datetime import datetime


class FVGType(StrEnum):
    """Type of Fair Value Gap."""

    BULLISH = "bullish_fvg"
    BEARISH = "bearish_fvg"


@dataclass(frozen=True, slots=True)
class FVG:
    """A detected Fair Value Gap."""

    time: datetime
    fvg_type: FVGType
    top: float
    bottom: float
    midpoint: float
    index: int


def detect_fvg_vectorized(
    df: pl.DataFrame,
    min_gap_atr_ratio: float = 0.0,
) -> pl.DataFrame:
    """Detect Fair Value Gaps on a full DataFrame (vectorized).

    Args:
        df: DataFrame with columns: time, high, low, close.
        min_gap_atr_ratio: Minimum gap size as ratio of ATR to filter noise.
            0.0 means no filtering.

    Returns:
        DataFrame with columns: time, fvg_type, top, bottom, midpoint, index.
    """
    if len(df) < 3:
        return _empty_fvg_df()

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    times = df["time"]

    fvgs: list[dict[str, object]] = []

    for i in range(2, len(highs)):
        c1_high = highs[i - 2]
        c3_low = lows[i]

        # Bullish FVG: gap between candle1 high and candle3 low
        if c3_low > c1_high:
            gap_size = float(c3_low - c1_high)
            fvgs.append({
                "time": times[i - 1],  # FVG belongs to the middle candle
                "fvg_type": FVGType.BULLISH,
                "top": float(c3_low),
                "bottom": float(c1_high),
                "midpoint": float(c1_high + gap_size / 2),
                "index": i - 1,
            })

        # Bearish FVG: gap between candle3 high and candle1 low
        c1_low = lows[i - 2]
        c3_high = highs[i]

        if c3_high < c1_low:
            gap_size = float(c1_low - c3_high)
            fvgs.append({
                "time": times[i - 1],
                "fvg_type": FVGType.BEARISH,
                "top": float(c1_low),
                "bottom": float(c3_high),
                "midpoint": float(c3_high + gap_size / 2),
                "index": i - 1,
            })

    if not fvgs:
        return _empty_fvg_df()

    return pl.DataFrame(fvgs, schema=_FVG_SCHEMA)


@dataclass
class FVGState:
    """Incremental state for FVG detection."""

    prev_highs: list[float] | None = None
    prev_lows: list[float] | None = None
    prev_times: list[datetime] | None = None
    candle_count: int = 0

    def __post_init__(self) -> None:
        if self.prev_highs is None:
            self.prev_highs = []
            self.prev_lows = []
            self.prev_times = []


def detect_fvg_incremental(
    candle: dict[str, object],
    state: FVGState,
) -> list[FVG]:
    """Detect FVGs incrementally as new candles arrive.

    Args:
        candle: Dict with keys: time, high, low.
        state: Mutable FVG detection state.

    Returns:
        List of newly detected FVGs (usually 0 or 1).
    """
    assert state.prev_highs is not None
    assert state.prev_lows is not None
    assert state.prev_times is not None

    h = float(candle["high"])  # type: ignore[arg-type]
    lo = float(candle["low"])  # type: ignore[arg-type]

    state.prev_highs.append(h)
    state.prev_lows.append(lo)
    state.prev_times.append(candle["time"])  # type: ignore[arg-type]
    state.candle_count += 1

    results: list[FVG] = []

    if len(state.prev_highs) < 3:
        return results

    c1_high = state.prev_highs[-3]
    c1_low = state.prev_lows[-3]
    c3_high = state.prev_highs[-1]
    c3_low = state.prev_lows[-1]
    mid_time = state.prev_times[-2]
    mid_idx = state.candle_count - 2

    # Bullish FVG
    if c3_low > c1_high:
        gap_size = c3_low - c1_high
        results.append(FVG(
            time=mid_time,
            fvg_type=FVGType.BULLISH,
            top=c3_low,
            bottom=c1_high,
            midpoint=c1_high + gap_size / 2,
            index=mid_idx,
        ))

    # Bearish FVG
    if c3_high < c1_low:
        gap_size = c1_low - c3_high
        results.append(FVG(
            time=mid_time,
            fvg_type=FVGType.BEARISH,
            top=c1_low,
            bottom=c3_high,
            midpoint=c3_high + gap_size / 2,
            index=mid_idx,
        ))

    # Keep only last 3 candles
    if len(state.prev_highs) > 3:
        state.prev_highs = state.prev_highs[-3:]
        state.prev_lows = state.prev_lows[-3:]
        state.prev_times = state.prev_times[-3:]

    return results


_FVG_SCHEMA: dict[str, Any] = {
    "time": pl.Datetime("us", "UTC"),
    "fvg_type": pl.Utf8,
    "top": pl.Float64,
    "bottom": pl.Float64,
    "midpoint": pl.Float64,
    "index": pl.Int64,
}


def _empty_fvg_df() -> pl.DataFrame:
    return pl.DataFrame(schema=_FVG_SCHEMA)
