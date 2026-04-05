"""Walk-forward validation with REDUCED search space + warm-start.

Fixes converged parameters (confluence weights, swing bars, sl_atr_multiple,
risk tiers) and only optimizes ~15 params instead of 30.

Usage:
    uv run python -m scripts.run_walk_forward_reduced \\
        --instrument EUR/USD --train-months 4 --test-weeks 1 --trials 200
"""

from src.backtest.walk_forward import run_walk_forward_cli
from src.strategy.params import StrategyParams

if __name__ == "__main__":
    run_walk_forward_cli(
        param_builder=StrategyParams.from_optuna_trial_reduced,
        param_reconstructor=StrategyParams.from_dict,
        description="Walk-forward validation (reduced, ~15 params)",
        default_trials=200,
    )
