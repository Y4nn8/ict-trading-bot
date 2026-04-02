"""Dynamic position sizing based on confluence score and risk parameters."""

from __future__ import annotations

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
    ) -> float:
        """Compute position size in units.

        Args:
            capital: Current account capital.
            confluence_score: Trade confluence score (0-1).
            entry_price: Entry price.
            stop_loss: Stop loss price.

        Returns:
            Position size in units. 0 if risk is too high.
        """
        risk_pct = self._get_risk_pct(confluence_score)
        risk_amount = capital * risk_pct / 100

        sl_distance = abs(entry_price - stop_loss)
        if sl_distance <= 0:
            return 0.0

        size = risk_amount / sl_distance
        return max(size, 0.0)

    def _get_risk_pct(self, confluence_score: float) -> float:
        """Map confluence score to risk percentage."""
        if confluence_score >= self._tiers.high_threshold:
            return min(self._tiers.high_risk_pct, self._max_risk_pct)
        if confluence_score >= self._tiers.low_threshold:
            return min(self._tiers.medium_risk_pct, self._max_risk_pct)
        return min(self._tiers.low_risk_pct, self._max_risk_pct)
