"""Walk-forward validation with SMART search space + warm-start.

Combines three optimizations vs the base walk-forward:
1. Fixed stable params (sl_atr=0.52, rr_ratio=3.0, swing_left=1)
2. Grouped correlated params (atr_period, risk_pct, 3 confluence groups)
3. Bayesian warm-start: best params from window N seed window N+1

~16 optimized params instead of 30.

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
        description="Walk-forward validation (smart, 16 params)",
        default_trials=200,
    )
