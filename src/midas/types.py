"""Shared types for the Midas tick-level scalping engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime


@dataclass(frozen=True, slots=True)
class Tick:
    """A single market tick with bid/ask prices."""

    time: datetime
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        """Mid price."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid


@dataclass(slots=True)
class PartialCandle:
    """A 10-second candle currently being built from ticks.

    Uses mid price for OHLC, stores latest bid/ask for spread info.
    """

    bucket_start: datetime
    open: float
    high: float
    low: float
    close: float
    tick_count: int
    bid: float
    ask: float
    elapsed_seconds: float

    @property
    def range(self) -> float:
        """High - low range of the candle."""
        return self.high - self.low

    @property
    def position_in_range(self) -> float:
        """Where close sits in the candle range (0=low, 1=high)."""
        r = self.range
        return (self.close - self.low) / r if r > 0 else 0.5


class MidasSignal(StrEnum):
    """LightGBM prediction outputs."""

    BUY = "BUY"
    SELL = "SELL"
    PASS = "PASS"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


@dataclass(frozen=True, slots=True)
class LabelConfig:
    """Configuration for tick labeling with SL/TP lookahead.

    Supports two modes:
      - **Fixed**: uses ``sl_points``/``tp_points`` directly.
      - **ATR-based**: when ``k_sl``/``k_tp`` are set, computes
        per-row SL = k_sl * ATR. Falls back to ``sl_points``/``tp_points``
        when ATR is zero.

    Args:
        sl_points: Stop loss distance in price points (fixed / fallback).
        tp_points: Take profit distance in price points (fixed / fallback).
        timeout_seconds: Max lookahead for SL/TP resolution.
        k_sl: SL multiplier for ATR-based mode (None = fixed mode).
        k_tp: TP multiplier for ATR-based mode (None = fixed mode).
        atr_column: Column name containing ATR values.
    """

    sl_points: float = 3.0
    tp_points: float = 3.0
    timeout_seconds: float = 300.0
    k_sl: float | None = None
    k_tp: float | None = None
    atr_column: str = "scalp__m1_atr"
