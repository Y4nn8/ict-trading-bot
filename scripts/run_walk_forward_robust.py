"""Walk-forward validation with anti-overfitting measures.

Uses median of top N trials instead of single best, with 2-week
test windows by default for more statistically significant OOS results.

Usage:
    uv run python -m scripts.run_walk_forward_robust \\
        --instrument XAUUSD --train-months 5 --test-weeks 2 \\
        --trials 200 --top-n 10
"""

from src.backtest.walk_forward import run_walk_forward_cli
from src.strategy.params import StrategyParams

if __name__ == "__main__":
    run_walk_forward_cli(
        param_builder=StrategyParams.from_optuna_trial,
        param_reconstructor=StrategyParams.from_dict,
        description="Walk-forward validation (robust, median top-N)",
        default_trials=200,
    )
