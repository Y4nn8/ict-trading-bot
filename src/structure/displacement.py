"""Displacement / impulse move detection.

A displacement is a strong, directional price move characterized by:
- Large body candles (body > N * ATR)
- Sequential candles in the same direction
- Often creates FVGs and breaks structure
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from datetime import datetime


class DisplacementDirection(StrEnum):
    """Direction of a displacement move."""

    BULLISH = "bullish_displacement"
    BEARISH = "bearish_displacement"


@dataclass(frozen=True, slots=True)
class Displacement:
    """A detected displacement move."""

    time: datetime
    direction: DisplacementDirection
    body_atr_ratio: float
    index: int


def detect_displacement_vectorized(
    df: pl.DataFrame,
    atr_period: int = 14,
    threshold: float = 1.5,
) -> pl.DataFrame:
    """Detect displacement moves on a full DataFrame.

    A displacement is a candle whose body exceeds threshold * ATR.

    Args:
        df: DataFrame with columns: time, open, high, low, close.
        atr_period: Period for ATR calculation.
        threshold: Minimum body/ATR ratio to qualify as displacement.

    Returns:
        DataFrame with columns: time, direction, body_atr_ratio, index.
    """
    if len(df) < atr_period + 1:
        return _empty_disp_df()

    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    times = df["time"]

    # Compute TR and ATR
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

    displacements: list[dict[str, object]] = []

    for i in range(atr_period + 1, len(opens)):
        if np.isnan(atr[i]):
            continue

        body = abs(closes[i] - opens[i])
        ratio = body / atr[i]

        if ratio >= threshold:
            direction = (
                DisplacementDirection.BULLISH
                if closes[i] > opens[i]
                else DisplacementDirection.BEARISH
            )
            displacements.append({
                "time": times[i],
                "direction": direction,
                "body_atr_ratio": float(ratio),
                "index": i,
            })

    if not displacements:
        return _empty_disp_df()

    return pl.DataFrame(displacements, schema=_DISP_SCHEMA)


@dataclass
class DisplacementState:
    """Incremental state for displacement detection."""

    recent_tr: list[float] = field(default_factory=list)
    prev_close: float | None = None
    atr_period: int = 14
    threshold: float = 1.5
    candle_count: int = 0


def detect_displacement_incremental(
    candle: dict[str, object],
    state: DisplacementState,
) -> list[Displacement]:
    """Detect displacements incrementally.

    Args:
        candle: Dict with keys: time, open, high, low, close.
        state: Mutable displacement detection state.

    Returns:
        List of newly detected Displacements.
    """
    o = float(candle["open"])  # type: ignore[arg-type]
    h = float(candle["high"])  # type: ignore[arg-type]
    lo = float(candle["low"])  # type: ignore[arg-type]
    c = float(candle["close"])  # type: ignore[arg-type]
    state.candle_count += 1

    results: list[Displacement] = []

    if state.prev_close is not None:
        tr = max(h - lo, abs(h - state.prev_close), abs(lo - state.prev_close))
        state.recent_tr.append(tr)
        if len(state.recent_tr) > state.atr_period:
            state.recent_tr = state.recent_tr[-state.atr_period :]

        if len(state.recent_tr) >= state.atr_period:
            atr = sum(state.recent_tr) / len(state.recent_tr)
            body = abs(c - o)
            ratio = body / atr if atr > 0 else 0.0

            if ratio >= state.threshold:
                direction = (
                    DisplacementDirection.BULLISH
                    if c > o
                    else DisplacementDirection.BEARISH
                )
                results.append(Displacement(
                    time=candle["time"],  # type: ignore[arg-type]
                    direction=direction,
                    body_atr_ratio=ratio,
                    index=state.candle_count - 1,
                ))

    state.prev_close = c
    return results


_DISP_SCHEMA: dict[str, Any] = {
    "time": pl.Datetime("us", "UTC"),
    "direction": pl.Utf8,
    "body_atr_ratio": pl.Float64,
    "index": pl.Int64,
}


def _empty_disp_df() -> pl.DataFrame:
    return pl.DataFrame(schema=_DISP_SCHEMA)
