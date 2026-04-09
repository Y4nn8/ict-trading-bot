"""Walk-forward XAUUSD v2: reduced search space + no news.

Fixes 12 converged params, restricts 9 close params, optimizes 13 free.
Based on W1/W2 best-param convergence analysis (v3 run).
News module disabled by default (ablation showed it hurts XAUUSD PnL).

Usage:
    uv run python -m scripts.run_walk_forward_xauusd_v2 \
        --instrument XAUUSD --train-months 5 --test-weeks 1 --trials 200 --no-news
"""

import sys

from src.backtest.walk_forward import run_walk_forward_cli
from src.strategy.params import StrategyParams

if __name__ == "__main__":
    # Default to --no-news for XAUUSD v2 (ablation confirmed news hurts PnL)
    if "--no-news" not in sys.argv:
        sys.argv.append("--no-news")
    run_walk_forward_cli(
        param_builder=StrategyParams.from_optuna_trial_xauusd_v2,
        param_reconstructor=StrategyParams.from_xauusd_v2_dict,
        description="Walk-forward XAUUSD v2 (22 params, no news)",
        default_trials=200,
    )
