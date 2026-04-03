"""Optimize strategy parameters using Optuna.

Runs walk-forward backtests with different parameter combinations
to find the best Sharpe ratio.

Usage:
    uv run python -m scripts.optimize_strategy --instrument EUR/USD --trials 100
    uv run python -m scripts.optimize_strategy --instrument DAX40 --trials 50 --days 60
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime, timedelta

import optuna
import polars as pl  # noqa: TC002
from dotenv import load_dotenv

from src.backtest.engine import BacktestEngine
from src.backtest.metrics import compute_metrics
from src.backtest.report import format_report, generate_report
from src.backtest.vectorized import precompute
from src.common.config import load_config
from src.common.db import Database
from src.common.logging import get_logger, setup_logging
from src.market_data.storage import CandleStorage
from src.strategy.factory import build_strategy
from src.strategy.params import StrategyParams

load_dotenv()

logger = get_logger(__name__)

_cached_candles: dict[str, pl.DataFrame] = {}


async def load_candles(
    instrument: str, days: int
) -> pl.DataFrame:
    """Load candles from DB (cached)."""
    cache_key = f"{instrument}_{days}"
    if cache_key in _cached_candles:
        return _cached_candles[cache_key]

    config = load_config()
    db = Database(config.database)
    await db.connect()
    storage = CandleStorage(db)

    end = datetime.now(tz=UTC)
    start = end - timedelta(days=days)
    candles = await storage.fetch_candles(instrument, "M5", start=start, end=end)
    await db.disconnect()

    _cached_candles[cache_key] = candles
    return candles


def run_single_backtest(
    candles: pl.DataFrame,
    instrument: str,
    params: StrategyParams,
    initial_capital: float = 5000.0,
    max_mdd_pct: float = 5.0,
) -> float:
    """Run one backtest and return a composite score.

    Objective = PnL_normalized * Sharpe * PF_capped

    Hard constraint: if max drawdown > max_mdd_pct, trial is rejected.

    Args:
        candles: Polars DataFrame with candle data.
        instrument: Instrument name.
        params: Strategy parameters to test.
        initial_capital: Starting capital.
        max_mdd_pct: Maximum acceptable drawdown in % (hard reject above).

    Returns:
        Composite score (higher is better). Returns -10 on failure/rejection.
    """
    if candles.is_empty():
        return -10.0

    precomputed = precompute(candles, instrument, "M5", params=params)
    components = build_strategy(params)

    engine = BacktestEngine(
        precomputed=precomputed,
        confluence_scorer=components.confluence_scorer,
        entry_evaluator=components.entry_evaluator,
        exit_evaluator=components.exit_evaluator,
        trade_filter=components.trade_filter,
        position_sizer=components.position_sizer,
        risk_manager=components.risk_manager,
        sim_config=components.sim_config,
        initial_capital=initial_capital,
    )
    result = engine.run()

    if len(result.trades) < 5:
        return -10.0

    metrics = compute_metrics(result.trades, initial_capital)

    # Hard constraint: reject if MDD exceeds limit
    if metrics.max_drawdown_pct * 100 > max_mdd_pct:
        return -10.0

    # Composite score components:
    # 1. PnL normalized by capital (so +500 on 5000 = 0.1)
    pnl_norm = metrics.total_pnl / initial_capital

    # 2. Sharpe (already scale-independent)
    sharpe = max(metrics.sharpe_ratio, 0)

    # 3. Profit factor, capped at 5 to avoid outlier dominance
    pf = min(metrics.profit_factor, 5.0) if metrics.profit_factor > 0 else 0

    # Combined: PnL drives magnitude, Sharpe rewards consistency,
    # PF rewards efficiency
    score = pnl_norm * (1 + sharpe) * (1 + pf * 0.2)

    return score


async def optimize(
    instrument: str,
    days: int,
    n_trials: int,
    initial_capital: float,
) -> None:
    """Run Optuna optimization."""
    config = load_config()
    setup_logging(config.logging.level, json_format=False)

    await logger.ainfo(
        "loading_data", instrument=instrument, days=days
    )
    candles = await load_candles(instrument, days)

    def objective(trial: optuna.Trial) -> float:
        params = StrategyParams.from_optuna_trial(trial)
        sharpe = run_single_backtest(candles, instrument, params, initial_capital)
        return sharpe

    study = optuna.create_study(
        direction="maximize",
        study_name=f"ict_strategy_{instrument}",
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    await logger.ainfo("starting_optimization", trials=n_trials)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Results
    best = study.best_trial
    print("\n" + "=" * 50)
    print("OPTIMIZATION RESULTS")
    print("=" * 50)
    print(f"Best Sharpe: {best.value:.4f}")
    print(f"Best trial: #{best.number}")
    print(f"Total trials: {len(study.trials)}")
    print("\nBest parameters:")
    for key, value in sorted(best.params.items()):
        print(f"  {key}: {value}")

    # Run final backtest with best params
    print("\n--- Final backtest with best params ---")
    best_params = StrategyParams.from_dict(best.params)

    precomputed = precompute(candles, instrument, "M5", params=best_params)
    components = build_strategy(best_params)
    engine = BacktestEngine(
        precomputed=precomputed,
        confluence_scorer=components.confluence_scorer,
        entry_evaluator=components.entry_evaluator,
        exit_evaluator=components.exit_evaluator,
        trade_filter=components.trade_filter,
        position_sizer=components.position_sizer,
        risk_manager=components.risk_manager,
        sim_config=components.sim_config,
        initial_capital=initial_capital,
    )
    result = engine.run()
    report = generate_report(result.trades, initial_capital)
    print(format_report(report))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Optimize strategy with Optuna")
    parser.add_argument("--instrument", type=str, default="EUR/USD")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--capital", type=float, default=5000.0)

    args = parser.parse_args()
    asyncio.run(optimize(args.instrument, args.days, args.trials, args.capital))


if __name__ == "__main__":
    main()
