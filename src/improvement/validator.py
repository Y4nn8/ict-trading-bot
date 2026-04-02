"""Walk-forward validation for proposed improvements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a proposed improvement."""

    accepted: bool
    baseline_sharpe: float
    new_sharpe: float
    baseline_mdd: float
    new_mdd: float
    improvement_pct: float
    reason: str


class ImprovementValidator:
    """Validates improvements using walk-forward comparison.

    Args:
        min_improvement_pct: Minimum Sharpe improvement to accept.
        max_mdd_degradation_pct: Maximum MDD increase allowed.
        max_sharpe_jump_pct: Reject suspicious jumps (overfitting).
    """

    def __init__(
        self,
        min_improvement_pct: float = 2.0,
        max_mdd_degradation_pct: float = 5.0,
        max_sharpe_jump_pct: float = 50.0,
    ) -> None:
        self._min_improvement = min_improvement_pct
        self._max_mdd_degradation = max_mdd_degradation_pct
        self._max_sharpe_jump = max_sharpe_jump_pct

    def validate(
        self,
        baseline_metrics: dict[str, Any],
        new_metrics: dict[str, Any],
    ) -> ValidationResult:
        """Compare baseline vs new metrics and decide acceptance.

        Args:
            baseline_metrics: Metrics from current strategy.
            new_metrics: Metrics from modified strategy.

        Returns:
            ValidationResult with acceptance decision.
        """
        baseline_sharpe = float(baseline_metrics.get("sharpe_ratio", 0))
        new_sharpe = float(new_metrics.get("sharpe_ratio", 0))
        baseline_mdd = float(baseline_metrics.get("max_drawdown_pct", 0))
        new_mdd = float(new_metrics.get("max_drawdown_pct", 0))

        improvement_pct = (
            (new_sharpe - baseline_sharpe) / abs(baseline_sharpe) * 100
            if baseline_sharpe != 0
            else 100.0
        )

        mdd_degradation = new_mdd - baseline_mdd

        # Decision logic
        accepted = True
        reason = "accepted"

        if improvement_pct < self._min_improvement:
            accepted = False
            reason = f"Insufficient improvement: {improvement_pct:.1f}%"
        elif improvement_pct > self._max_sharpe_jump:
            accepted = False
            reason = f"Suspicious Sharpe jump: {improvement_pct:.1f}% (overfitting risk)"
        elif mdd_degradation > self._max_mdd_degradation:
            accepted = False
            reason = f"MDD degradation: {mdd_degradation:.1f}% exceeds limit"

        logger.info(
            "validation_complete",
            accepted=accepted,
            improvement_pct=round(improvement_pct, 2),
            reason=reason,
        )

        return ValidationResult(
            accepted=accepted,
            baseline_sharpe=baseline_sharpe,
            new_sharpe=new_sharpe,
            baseline_mdd=baseline_mdd,
            new_mdd=new_mdd,
            improvement_pct=improvement_pct,
            reason=reason,
        )
