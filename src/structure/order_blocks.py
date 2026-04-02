"""Order Block (OB) detection.

An Order Block is the last opposing candle before a displacement move.
- Bullish OB: last bearish candle before a strong bullish move (displacement up).
- Bearish OB: last bullish candle before a strong bearish move (displacement down).

The OB zone is defined by the candle's range [low, high].
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from datetime import datetime


class OBType(StrEnum):
    """Type of Order Block."""

    BULLISH = "bullish_ob"
    BEARISH = "bearish_ob"


@dataclass(frozen=True, slots=True)
class OrderBlock:
    """A detected Order Block."""

    time: datetime
    ob_type: OBType
    top: float
    bottom: float
    index: int


def detect_order_blocks_vectorized(
    df: pl.DataFrame,
    displacement_factor: float = 2.0,
    atr_period: int = 14,
) -> pl.DataFrame:
    """Detect Order Blocks on a full DataFrame.

    An OB is identified when a displacement move (candle body > displacement_factor * ATR)
    follows an opposing candle.

    Args:
        df: DataFrame with columns: time, open, high, low, close.
        displacement_factor: Minimum body size as multiple of ATR.
        atr_period: Period for ATR calculation.

    Returns:
        DataFrame with columns: time, ob_type, top, bottom, index.
    """
    if len(df) < atr_period + 2:
        return _empty_ob_df()

    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    times = df["time"]

    # Compute ATR
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ),
    )
    atr = np.full(len(df), np.nan)
    for i in range(atr_period, len(tr)):
        atr[i + 1] = float(np.mean(tr[i - atr_period + 1 : i + 1]))

    obs: list[dict[str, object]] = []

    for i in range(1, len(opens)):
        if np.isnan(atr[i]):
            continue

        body = abs(closes[i] - opens[i])
        is_displacement = body > displacement_factor * atr[i]

        if not is_displacement:
            continue

        is_bullish_displacement = closes[i] > opens[i]
        prev_is_bearish = closes[i - 1] < opens[i - 1]
        prev_is_bullish = closes[i - 1] > opens[i - 1]

        if is_bullish_displacement and prev_is_bearish:
            obs.append({
                "time": times[i - 1],
                "ob_type": OBType.BULLISH,
                "top": float(highs[i - 1]),
                "bottom": float(lows[i - 1]),
                "index": i - 1,
            })
        elif not is_bullish_displacement and prev_is_bullish:
            obs.append({
                "time": times[i - 1],
                "ob_type": OBType.BEARISH,
                "top": float(highs[i - 1]),
                "bottom": float(lows[i - 1]),
                "index": i - 1,
            })

    if not obs:
        return _empty_ob_df()

    return pl.DataFrame(obs, schema=_OB_SCHEMA)


@dataclass
class OBState:
    """Incremental state for Order Block detection."""

    prev_open: float | None = None
    prev_close: float | None = None
    prev_high: float | None = None
    prev_low: float | None = None
    prev_time: datetime | None = None
    prev_index: int = -1
    recent_tr: list[float] | None = None
    atr_period: int = 14
    displacement_factor: float = 2.0
    candle_count: int = 0

    def __post_init__(self) -> None:
        if self.recent_tr is None:
            self.recent_tr = []


def detect_order_blocks_incremental(
    candle: dict[str, object],
    state: OBState,
) -> list[OrderBlock]:
    """Detect Order Blocks incrementally.

    Args:
        candle: Dict with keys: time, open, high, low, close.
        state: Mutable OB detection state.

    Returns:
        List of newly detected OrderBlocks.
    """
    assert state.recent_tr is not None

    o = float(candle["open"])  # type: ignore[arg-type]
    h = float(candle["high"])  # type: ignore[arg-type]
    lo = float(candle["low"])  # type: ignore[arg-type]
    c = float(candle["close"])  # type: ignore[arg-type]
    state.candle_count += 1

    results: list[OrderBlock] = []

    # Compute TR
    if state.prev_close is not None:
        tr = max(h - lo, abs(h - state.prev_close), abs(lo - state.prev_close))
        state.recent_tr.append(tr)
        if len(state.recent_tr) > state.atr_period:
            state.recent_tr = state.recent_tr[-state.atr_period :]

        # Check for displacement
        if len(state.recent_tr) >= state.atr_period:
            atr = sum(state.recent_tr) / len(state.recent_tr)
            body = abs(c - o)

            if body > state.displacement_factor * atr and state.prev_open is not None:
                is_bullish = c > o
                prev_is_bearish = state.prev_close < state.prev_open
                prev_is_bullish = state.prev_close > state.prev_open

                if is_bullish and prev_is_bearish:
                    results.append(OrderBlock(
                        time=state.prev_time,  # type: ignore[arg-type]
                        ob_type=OBType.BULLISH,
                        top=state.prev_high,  # type: ignore[arg-type]
                        bottom=state.prev_low,  # type: ignore[arg-type]
                        index=state.prev_index,
                    ))
                elif not is_bullish and prev_is_bullish:
                    results.append(OrderBlock(
                        time=state.prev_time,  # type: ignore[arg-type]
                        ob_type=OBType.BEARISH,
                        top=state.prev_high,  # type: ignore[arg-type]
                        bottom=state.prev_low,  # type: ignore[arg-type]
                        index=state.prev_index,
                    ))

    state.prev_open = o
    state.prev_close = c
    state.prev_high = h
    state.prev_low = lo
    state.prev_time = candle["time"]  # type: ignore[assignment]
    state.prev_index = state.candle_count - 1

    return results


_OB_SCHEMA: dict[str, Any] = {
    "time": pl.Datetime("us", "UTC"),
    "ob_type": pl.Utf8,
    "top": pl.Float64,
    "bottom": pl.Float64,
    "index": pl.Int64,
}


def _empty_ob_df() -> pl.DataFrame:
    return pl.DataFrame(schema=_OB_SCHEMA)
