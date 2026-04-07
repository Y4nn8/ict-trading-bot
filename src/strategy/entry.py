"""Entry condition evaluation for ICT/SMC setups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.common.models import Direction


@dataclass(frozen=True, slots=True)
class EntrySignal:
    """A validated entry signal with price levels."""

    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    setup_type: str = "ict_confluence"


class EntryEvaluator:
    """Evaluates whether current conditions warrant an entry.

    Args:
        min_confluence: Minimum confluence score to trigger entry.
        default_sl_atr_multiple: Default SL distance as ATR multiple.
        default_rr_ratio: Default risk:reward ratio for TP.
    """

    def __init__(
        self,
        min_confluence: float = 0.4,
        default_sl_atr_multiple: float = 1.5,
        default_rr_ratio: float = 2.0,
    ) -> None:
        self._min_confluence = min_confluence
        self._sl_atr_mult = default_sl_atr_multiple
        self._rr_ratio = default_rr_ratio

    def evaluate(
        self,
        candle: dict[str, Any],
        context: dict[str, Any],
        confluence_score: float,
    ) -> EntrySignal | None:
        """Evaluate entry conditions.

        Args:
            candle: Current candle data.
            context: Structure context at this candle.
            confluence_score: Pre-computed confluence score.

        Returns:
            EntrySignal if conditions are met, None otherwise.
        """
        if confluence_score < self._min_confluence:
            return None

        # Determine direction from market structure breaks
        ms_breaks = context.get("ms_breaks", [])
        if not ms_breaks:
            return None

        latest_break = ms_breaks[-1]
        direction_str = latest_break.get("direction", "")

        if direction_str == "bullish":
            direction = Direction.LONG
        elif direction_str == "bearish":
            direction = Direction.SHORT
        else:
            return None

        # HTF directional filter: skip if M5 direction opposes H1 trend
        htf_trend = context.get("htf_trend")
        if htf_trend is not None and htf_trend != "undefined" and direction_str != htf_trend:
            return None

        close = float(candle["close"])
        high = float(candle["high"])
        low = float(candle["low"])

        # Use pre-computed rolling ATR if available, fall back to candle range
        atr_raw = candle.get("atr")
        atr_estimate = float(atr_raw) if atr_raw is not None and atr_raw == atr_raw else high - low

        if atr_estimate <= 0:
            return None

        sl_distance = atr_estimate * self._sl_atr_mult
        tp_distance = sl_distance * self._rr_ratio

        if direction == Direction.LONG:
            entry_price = close
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:
            entry_price = close
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        return EntrySignal(
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
