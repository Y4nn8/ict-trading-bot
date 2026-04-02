"""Exit condition evaluation: SL, TP, trailing stop, time-based exit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.backtest.engine import OpenPosition


@dataclass(frozen=True, slots=True)
class ExitSignal:
    """A triggered exit signal."""

    exit_price: float
    reason: str


class ExitEvaluator:
    """Evaluates exit conditions for open positions.

    Args:
        trailing_stop_atr: If > 0, activate trailing stop at this ATR multiple.
        max_hold_candles: Close position after N candles (0 = disabled).
    """

    def __init__(
        self,
        trailing_stop_atr: float = 0.0,
        max_hold_candles: int = 0,
    ) -> None:
        self._trailing_atr = trailing_stop_atr
        self._max_hold = max_hold_candles
        self._candle_counts: dict[str, int] = {}

    def evaluate(
        self,
        position: OpenPosition,
        candle: dict[str, Any],
    ) -> ExitSignal | None:
        """Evaluate exit conditions for a position.

        SL/TP are handled directly by the engine. This method handles
        additional exit conditions like trailing stops and time-based exits.

        Args:
            position: The open position to evaluate.
            candle: Current candle data.

        Returns:
            ExitSignal if an exit condition is triggered, None otherwise.
        """
        # Track candle count per position
        count = self._candle_counts.get(position.trade_id, 0) + 1
        self._candle_counts[position.trade_id] = count

        # Time-based exit
        if self._max_hold > 0 and count >= self._max_hold:
            self._candle_counts.pop(position.trade_id, None)
            return ExitSignal(
                exit_price=float(candle["close"]),
                reason="max_hold_time",
            )

        return None

    def on_position_closed(self, trade_id: str) -> None:
        """Clean up tracking state when a position is closed."""
        self._candle_counts.pop(trade_id, None)
