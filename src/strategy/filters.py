"""Trade filters: spread, session, correlation, max positions."""

from __future__ import annotations

from typing import Any


class TradeFilter:
    """Filters that must pass before a trade entry is allowed.

    Args:
        max_spread_pips: Maximum allowed spread in pips.
        require_killzone: Only trade during killzones.
        max_positions: Maximum simultaneous positions.
    """

    def __init__(
        self,
        max_spread_pips: float = 3.0,
        require_killzone: bool = False,
        max_positions: int = 5,
    ) -> None:
        self._max_spread = max_spread_pips
        self._require_kz = require_killzone
        self._max_positions = max_positions

    def passes(
        self,
        candle: dict[str, Any],
        context: dict[str, Any],
        current_positions: int,
    ) -> bool:
        """Check if all filters pass for a potential trade.

        Args:
            candle: Current candle data.
            context: Structure context.
            current_positions: Number of currently open positions.

        Returns:
            True if all filters pass.
        """
        # Max positions check
        if current_positions >= self._max_positions:
            return False

        # Spread filter
        spread = candle.get("spread")
        if spread is not None and float(spread) > self._max_spread * 0.0001:
            return False

        # Killzone filter
        return not (self._require_kz and not context.get("in_killzone", False))
