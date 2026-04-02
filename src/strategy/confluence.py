"""Rule-based confluence scoring for ICT/SMC setups.

Each factor contributes a weighted score. The total determines
trade quality and position sizing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ConfluenceWeights:
    """Configurable weights for confluence factors."""

    fvg: float = 0.15
    order_block: float = 0.20
    market_structure: float = 0.25
    displacement: float = 0.10
    killzone: float = 0.15
    premium_discount: float = 0.15


class ConfluenceScorer:
    """Scores trade setups based on ICT confluence factors.

    Args:
        weights: Factor weights (should sum to ~1.0).
    """

    def __init__(self, weights: ConfluenceWeights | None = None) -> None:
        self._weights = weights or ConfluenceWeights()

    def score(self, candle: dict[str, Any], context: dict[str, Any]) -> float:
        """Compute confluence score for a candle/context.

        Args:
            candle: Current candle data.
            context: Pre-computed structure context at this candle.

        Returns:
            Confluence score between 0.0 and 1.0.
        """
        total = 0.0

        # FVG present at this candle
        if context.get("fvgs"):
            total += self._weights.fvg

        # Order block present
        if context.get("order_blocks"):
            total += self._weights.order_block

        # Market structure break at this candle
        if context.get("ms_breaks"):
            total += self._weights.market_structure

        # Displacement at this candle
        if context.get("displacements"):
            total += self._weights.displacement

        # In a killzone
        if context.get("in_killzone"):
            total += self._weights.killzone

        # In premium/discount zone (bonus if in correct zone for direction)
        zone = context.get("zone")
        if zone in ("discount", "premium"):
            total += self._weights.premium_discount

        return min(total, 1.0)
