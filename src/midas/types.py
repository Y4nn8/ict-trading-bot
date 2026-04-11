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

    Args:
        sl_points: Stop loss distance in price points.
        tp_points: Take profit distance in price points.
        timeout_seconds: Max lookahead for SL/TP resolution.
    """

    sl_points: float = 3.0
    tp_points: float = 3.0
    timeout_seconds: float = 300.0
