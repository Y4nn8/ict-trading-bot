"""Walk-forward validation with full 30-param search space + warm-start.

Usage:
    uv run python -m scripts.run_walk_forward \\
        --instrument EUR/USD --train-months 4 --test-weeks 1 --trials 200
"""

from src.backtest.walk_forward import run_walk_forward_cli
from src.strategy.params import StrategyParams

if __name__ == "__main__":
    run_walk_forward_cli(
        param_builder=StrategyParams.from_optuna_trial,
        param_reconstructor=StrategyParams.from_dict,
        description="Walk-forward validation (full 30 params)",
        default_trials=30,
    )
