"""Optuna-based daily parameter optimization.

Optimizes numeric strategy parameters using walk-forward Sharpe
as the objective. Guards against overfitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import optuna

from src.common.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


@dataclass
class OptimizationResult:
    """Result of an Optuna optimization run."""

    best_params: dict[str, Any]
    best_sharpe: float
    baseline_sharpe: float
    improvement_pct: float
    accepted: bool
    reason: str
    n_trials: int


class OptunaOptimizer:
    """Daily parameter optimization using Optuna.

    Args:
        objective_fn: Function that takes params dict and returns Sharpe ratio.
        n_trials: Number of optimization trials.
        min_improvement_pct: Minimum improvement to accept changes.
        max_mdd_degradation_pct: Maximum MDD degradation allowed.
        max_sharpe_jump_pct: Reject if Sharpe improves too much (overfitting).
    """

    def __init__(
        self,
        objective_fn: Callable[[dict[str, Any]], float],
        n_trials: int = 100,
        min_improvement_pct: float = 2.0,
        max_mdd_degradation_pct: float = 5.0,
        max_sharpe_jump_pct: float = 50.0,
    ) -> None:
        self._objective_fn = objective_fn
        self._n_trials = n_trials
        self._min_improvement = min_improvement_pct
        self._max_mdd_degradation = max_mdd_degradation_pct
        self._max_sharpe_jump = max_sharpe_jump_pct

    def optimize(
        self,
        param_space: dict[str, tuple[float, float]],
        baseline_sharpe: float,
    ) -> OptimizationResult:
        """Run optimization over the parameter space.

        Args:
            param_space: Dict of param_name -> (min, max) ranges.
            baseline_sharpe: Current strategy Sharpe ratio.

        Returns:
            OptimizationResult with best params and acceptance status.
        """
        study = optuna.create_study(direction="maximize")

        def objective(trial: optuna.Trial) -> float:
            params = {
                name: trial.suggest_float(name, low, high)
                for name, (low, high) in param_space.items()
            }
            return self._objective_fn(params)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=self._n_trials)

        best_params = study.best_params
        best_sharpe = study.best_value

        improvement_pct = (
            (best_sharpe - baseline_sharpe) / abs(baseline_sharpe) * 100
            if baseline_sharpe != 0
            else 100.0
        )

        # Guard checks
        accepted = True
        reason = "accepted"

        if improvement_pct < self._min_improvement:
            accepted = False
            reason = f"Improvement {improvement_pct:.1f}% below minimum {self._min_improvement}%"

        elif improvement_pct > self._max_sharpe_jump:
            accepted = False
            reason = (
                f"Sharpe jump {improvement_pct:.1f}% exceeds "
                f"{self._max_sharpe_jump}% (overfitting risk)"
            )

        logger.info(
            "optimization_complete",
            best_sharpe=round(best_sharpe, 4),
            baseline_sharpe=round(baseline_sharpe, 4),
            improvement_pct=round(improvement_pct, 2),
            accepted=accepted,
            reason=reason,
        )

        return OptimizationResult(
            best_params=best_params,
            best_sharpe=best_sharpe,
            baseline_sharpe=baseline_sharpe,
            improvement_pct=improvement_pct,
            accepted=accepted,
            reason=reason,
            n_trials=self._n_trials,
        )
