"""Risk management: circuit breakers and drawdown limits."""

from __future__ import annotations


class RiskManager:
    """Enforces risk limits and circuit breakers.

    Args:
        max_daily_drawdown_pct: Max daily drawdown as % of capital.
        max_total_drawdown_pct: Max total drawdown as % of initial capital.
        max_positions: Max simultaneous open positions.
    """

    def __init__(
        self,
        max_daily_drawdown_pct: float = 3.0,
        max_total_drawdown_pct: float = 10.0,
        max_positions: int = 5,
    ) -> None:
        self._max_daily_dd_pct = max_daily_drawdown_pct
        self._max_total_dd_pct = max_total_drawdown_pct
        self._max_positions = max_positions

    def is_circuit_broken(
        self,
        daily_pnl: float,
        current_capital: float,
        initial_capital: float,
    ) -> bool:
        """Check if any circuit breaker is triggered.

        Args:
            daily_pnl: Today's cumulative PnL.
            current_capital: Current account balance.
            initial_capital: Starting account balance.

        Returns:
            True if trading should be halted.
        """
        # Daily drawdown check
        if current_capital > 0:
            daily_dd_pct = abs(min(daily_pnl, 0)) / current_capital * 100
            if daily_dd_pct >= self._max_daily_dd_pct:
                return True

        # Total drawdown check
        total_dd = initial_capital - current_capital
        if total_dd > 0:
            total_dd_pct = total_dd / initial_capital * 100
            if total_dd_pct >= self._max_total_dd_pct:
                return True

        return False

    def can_open_position(self, current_positions: int) -> bool:
        """Check if a new position can be opened.

        Args:
            current_positions: Number of currently open positions.

        Returns:
            True if under the position limit.
        """
        return current_positions < self._max_positions
