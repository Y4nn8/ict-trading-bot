"""Real-time position and portfolio tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.common.logging import get_logger
from src.common.models import Direction

if TYPE_CHECKING:
    from datetime import datetime

logger = get_logger(__name__)


@dataclass
class LivePosition:
    """A currently open live position."""

    deal_id: str
    epic: str
    instrument: str
    direction: Direction
    size: float
    entry_price: float
    stop_loss: float | None = None
    take_profit: float | None = None
    opened_at: datetime | None = None
    current_price: float | None = None
    unrealized_pnl: float = 0.0


class Portfolio:
    """Tracks all open positions and account state.

    Args:
        initial_capital: Starting capital.
    """

    def __init__(self, initial_capital: float = 10000.0) -> None:
        self._initial_capital = initial_capital
        self._realized_pnl: float = 0.0
        self._positions: dict[str, LivePosition] = {}

    @property
    def positions(self) -> dict[str, LivePosition]:
        """Get all open positions keyed by deal_id."""
        return self._positions

    @property
    def position_count(self) -> int:
        """Get the number of open positions."""
        return len(self._positions)

    @property
    def capital(self) -> float:
        """Get current capital (initial + realized PnL)."""
        return self._initial_capital + self._realized_pnl

    @property
    def equity(self) -> float:
        """Get current equity (capital + unrealized PnL)."""
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        return self.capital + unrealized

    def add_position(self, position: LivePosition) -> None:
        """Register a newly opened position.

        Args:
            position: The position to track.
        """
        self._positions[position.deal_id] = position
        logger.info(
            "position_tracked",
            deal_id=position.deal_id,
            instrument=position.instrument,
            direction=position.direction,
        )

    def close_position(self, deal_id: str, exit_price: float) -> float:
        """Close a tracked position and compute PnL.

        Args:
            deal_id: The deal ID to close.
            exit_price: The exit price.

        Returns:
            Realized PnL for this position.
        """
        pos = self._positions.pop(deal_id, None)
        if pos is None:
            logger.warning("position_not_found", deal_id=deal_id)
            return 0.0

        if pos.direction == Direction.LONG:
            pnl = (exit_price - pos.entry_price) * pos.size
        else:
            pnl = (pos.entry_price - exit_price) * pos.size

        self._realized_pnl += pnl
        logger.info(
            "position_closed",
            deal_id=deal_id,
            pnl=round(pnl, 2),
            capital=round(self.capital, 2),
        )
        return pnl

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update current prices and unrealized PnL for all positions.

        Args:
            prices: Dict of epic -> current price.
        """
        for pos in self._positions.values():
            price = prices.get(pos.epic)
            if price is None:
                continue
            pos.current_price = price
            if pos.direction == Direction.LONG:
                pos.unrealized_pnl = (price - pos.entry_price) * pos.size
            else:
                pos.unrealized_pnl = (pos.entry_price - price) * pos.size

    def get_summary(self) -> dict[str, Any]:
        """Get portfolio summary."""
        return {
            "capital": round(self.capital, 2),
            "equity": round(self.equity, 2),
            "realized_pnl": round(self._realized_pnl, 2),
            "open_positions": self.position_count,
            "positions": [
                {
                    "deal_id": p.deal_id,
                    "instrument": p.instrument,
                    "direction": p.direction,
                    "size": p.size,
                    "entry_price": p.entry_price,
                    "unrealized_pnl": round(p.unrealized_pnl, 2),
                }
                for p in self._positions.values()
            ],
        }
