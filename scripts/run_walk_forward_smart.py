"""Walk-forward validation with SMART search space + warm-start.

Fixes 16 params that converged (spread < 30%) across the 200-trial
30-param analysis (2026-04-06). Optimizes 14 divergent params.
Bayesian warm-start: best params from window N seed window N+1.

Usage:
    uv run python -m scripts.run_walk_forward_smart \\
        --instrument EUR/USD --train-months 4 --test-weeks 1 --trials 200
"""

from src.backtest.walk_forward import run_walk_forward_cli
from src.strategy.params import StrategyParams

if __name__ == "__main__":
    run_walk_forward_cli(
        param_builder=StrategyParams.from_optuna_trial_smart,
        param_reconstructor=StrategyParams.from_smart_dict,
        description="Walk-forward validation (smart, 14 params)",
        default_trials=200,
    )
