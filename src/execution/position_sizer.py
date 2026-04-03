"""Dynamic position sizing based on confluence score and risk parameters."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RiskTiers:
    """Maps confluence score ranges to risk percentages."""

    low_threshold: float = 0.4
    high_threshold: float = 0.7
    low_risk_pct: float = 0.5
    medium_risk_pct: float = 1.0
    high_risk_pct: float = 2.0


class PositionSizer:
    """Computes position size based on risk and confluence.

    Args:
        tiers: Risk tier configuration.
        max_risk_pct: Absolute maximum risk per trade.
    """

    def __init__(
        self,
        tiers: RiskTiers | None = None,
        max_risk_pct: float = 2.0,
    ) -> None:
        self._tiers = tiers or RiskTiers()
        self._max_risk_pct = max_risk_pct

    def compute_size(
        self,
        capital: float,
        confluence_score: float,
        entry_price: float,
        stop_loss: float,
        value_per_point: float = 1.0,
        min_size: float = 0.5,
        size_step: float = 0.5,
    ) -> float:
        """Compute position size in contracts.

        The size is calculated so that the risk (SL distance * size * value_per_point)
        equals the target risk amount. Then rounded down to the nearest size_step
        and clamped to min_size.

        Args:
            capital: Current account capital.
            confluence_score: Trade confluence score (0-1).
            entry_price: Entry price.
            stop_loss: Stop loss price.
            value_per_point: Value per point per contract (e.g. €1 for DAX €1).
            min_size: Minimum position size (broker constraint).
            size_step: Size increment step (e.g. 0.5 contracts).

        Returns:
            Position size in contracts. 0 if the minimum size exceeds risk budget.
        """
        risk_pct = self._get_risk_pct(confluence_score)
        risk_amount = capital * risk_pct / 100

        sl_distance = abs(entry_price - stop_loss)
        if sl_distance <= 0 or value_per_point <= 0:
            return 0.0

        # Risk per contract = SL distance * value per point
        risk_per_contract = sl_distance * value_per_point

        # Raw size in contracts
        raw_size = risk_amount / risk_per_contract

        # Round down to size_step
        size = math.floor(raw_size / size_step) * size_step if size_step > 0 else raw_size

        # Check if minimum size fits within risk budget
        min_risk = min_size * risk_per_contract
        if min_risk > risk_amount:
            return 0.0

        return max(size, min_size)

    def _get_risk_pct(self, confluence_score: float) -> float:
        """Map confluence score to risk percentage."""
        if confluence_score >= self._tiers.high_threshold:
            return min(self._tiers.high_risk_pct, self._max_risk_pct)
        if confluence_score >= self._tiers.low_threshold:
            return min(self._tiers.medium_risk_pct, self._max_risk_pct)
        return min(self._tiers.low_risk_pct, self._max_risk_pct)
