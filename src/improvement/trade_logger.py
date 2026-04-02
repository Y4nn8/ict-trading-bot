"""Full context capture per trade for improvement analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.common.logging import get_logger

if TYPE_CHECKING:
    from datetime import datetime

logger = get_logger(__name__)


@dataclass
class TradeContext:
    """Full context captured for a trade, used by improvement loops."""

    trade_id: str
    instrument: str
    timeframe: str
    direction: str
    entry_price: float
    exit_price: float | None
    stop_loss: float
    take_profit: float
    confluence_score: float
    pnl: float | None
    r_multiple: float | None
    entry_time: datetime
    exit_time: datetime | None
    setup_type: str
    active_fvgs: list[dict[str, Any]]
    active_obs: list[dict[str, Any]]
    ms_trend: str
    session: str
    killzone: str
    news_context: dict[str, Any] | None = None


class TradeLogger:
    """Logs full trade context for post-trade analysis.

    Stores trade contexts in memory and optionally persists to DB.
    """

    def __init__(self) -> None:
        self._trade_contexts: list[TradeContext] = []

    def log_trade(self, context: TradeContext) -> None:
        """Log a trade with full context.

        Args:
            context: Complete trade context.
        """
        self._trade_contexts.append(context)
        logger.info(
            "trade_logged",
            trade_id=context.trade_id,
            instrument=context.instrument,
            pnl=context.pnl,
        )

    def get_recent_trades(self, count: int = 50) -> list[TradeContext]:
        """Get the most recent logged trades.

        Args:
            count: Number of trades to return.

        Returns:
            List of recent TradeContext objects.
        """
        return self._trade_contexts[-count:]

    def get_all_trades(self) -> list[TradeContext]:
        """Get all logged trades."""
        return list(self._trade_contexts)

    def clear(self) -> None:
        """Clear all logged trades."""
        self._trade_contexts.clear()
