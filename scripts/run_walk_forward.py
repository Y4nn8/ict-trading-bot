"""Walk-forward validation: optimize on train, test on unseen data.

For each window:
1. Optimize params on train period (Optuna)
2. Backtest with best params on test period (never seen by optimizer)
3. Aggregate test results across all windows

Usage:
    uv run python -m scripts.run_walk_forward --instrument EUR/USD --train-months 4 --test-months 1
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import timedelta

import optuna
import polars as pl
from dotenv import load_dotenv

from src.backtest.engine import BacktestEngine
from src.backtest.metrics import PerformanceMetrics, compute_metrics
from src.backtest.vectorized import precompute
from src.common.config import load_config
from src.common.db import Database
from src.common.logging import get_logger, setup_logging
from src.market_data.storage import CandleStorage
from src.news.store import NewsStore
from src.strategy.factory import build_strategy
from src.strategy.params import StrategyParams

load_dotenv()

logger = get_logger(__name__)


def _run_backtest(
    candles: pl.DataFrame,
    instrument: str,
    params: StrategyParams,
    news_events: list[dict[str, object]],
    initial_capital: float,
) -> tuple[list[object], PerformanceMetrics]:
    """Run a single backtest, return trades and metrics."""
    if candles.is_empty():
        return [], compute_metrics([], initial_capital)

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
        news_events=news_events,
    )
    result = engine.run()
    metrics = compute_metrics(result.trades, initial_capital)
    return result.trades, metrics


def _score(metrics: PerformanceMetrics, max_mdd_pct: float) -> float:
    """Composite score (same as optimize_strategy.py)."""
    if metrics.total_trades < 5:
        return -10.0
    if metrics.max_drawdown_pct * 100 > max_mdd_pct:
        return -10.0

    pnl_norm = metrics.total_pnl / 5000  # Normalize by typical capital
    sharpe = max(metrics.sharpe_ratio, 0)
    pf = min(metrics.profit_factor, 5.0) if metrics.profit_factor > 0 else 0
    return pnl_norm * (1 + sharpe) * (1 + pf * 0.2)


async def run_walk_forward(
    instrument: str,
    train_months: int,
    test_months: int,
    trials_per_window: int,
    initial_capital: float,
    max_mdd_pct: float,
) -> None:
    """Run walk-forward validation."""
    config = load_config()
    setup_logging(config.logging.level, json_format=False)

    db = Database(config.database)
    await db.connect()
    storage = CandleStorage(db)
    news_store = NewsStore(db)

    try:
        # Load all available M5 data
        all_candles = await storage.fetch_candles(instrument, "M5")
        if all_candles.is_empty():
            await logger.aerror("no_candles", instrument=instrument)
            return

        data_start = all_candles["time"][0]
        data_end = all_candles["time"][-1]
        await logger.ainfo(
            "data_loaded",
            candles=len(all_candles),
            start=str(data_start),
            end=str(data_end),
        )

        # Load all news
        all_news = await news_store.get_events(data_start, data_end)
        await logger.ainfo("news_loaded", count=len(all_news))

        # Generate windows
        window_size = timedelta(days=(train_months + test_months) * 30)
        step = timedelta(days=test_months * 30)

        windows = []
        current = data_start
        while current + window_size <= data_end:
            train_start = current
            train_end = current + timedelta(days=train_months * 30)
            test_start = train_end
            test_end = test_start + timedelta(days=test_months * 30)
            windows.append((train_start, train_end, test_start, test_end))
            current += step

        if not windows:
            await logger.aerror("not_enough_data_for_walk_forward")
            return

        await logger.ainfo("windows_generated", count=len(windows))

        # Run each window
        test_results: list[PerformanceMetrics] = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            print(f"\n{'='*60}")
            print(f"WINDOW {i+1}/{len(windows)}")
            print(f"  Train: {train_start.date()} → {train_end.date()}")
            print(f"  Test:  {test_start.date()} → {test_end.date()}")
            print(f"{'='*60}")

            # Split candles
            train_candles = all_candles.filter(
                (pl.col("time") >= train_start) & (pl.col("time") < train_end)
            )
            test_candles = all_candles.filter(
                (pl.col("time") >= test_start) & (pl.col("time") < test_end)
            )

            # Split news
            train_news = [
                n for n in all_news
                if train_start <= n.get("time", data_start) < train_end
            ]
            test_news = [
                n for n in all_news
                if test_start <= n.get("time", data_start) < test_end
            ]

            print(f"  Train candles: {len(train_candles)}, news: {len(train_news)}")
            print(f"  Test candles:  {len(test_candles)}, news: {len(test_news)}")

            if train_candles.is_empty() or test_candles.is_empty():
                print("  SKIP: insufficient data")
                continue

            # Optimize on train — bind loop vars via default args
            def objective(
                trial: optuna.Trial,
                _candles: pl.DataFrame = train_candles,
                _news: list[dict[str, object]] = train_news,
            ) -> float:
                params = StrategyParams.from_optuna_trial(trial)
                _, metrics = _run_backtest(
                    _candles, instrument, params, _news, initial_capital
                )
                return _score(metrics, max_mdd_pct)

            study = optuna.create_study(direction="maximize")
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(objective, n_trials=trials_per_window, show_progress_bar=True)

            best_params = StrategyParams.from_dict(study.best_trial.params)
            _, train_metrics = _run_backtest(
                train_candles, instrument, best_params, train_news, initial_capital
            )

            print("\n  TRAIN results (in-sample):")
            print(f"    Trades: {train_metrics.total_trades}")
            print(f"    PnL: {train_metrics.total_pnl:.2f}")
            print(f"    Sharpe: {train_metrics.sharpe_ratio:.3f}")
            print(f"    MDD: {train_metrics.max_drawdown_pct*100:.1f}%")

            # Test on unseen data
            _, test_metrics = _run_backtest(
                test_candles, instrument, best_params, test_news, initial_capital
            )
            test_results.append(test_metrics)

            print("\n  TEST results (out-of-sample):")
            print(f"    Trades: {test_metrics.total_trades}")
            print(f"    PnL: {test_metrics.total_pnl:.2f}")
            print(f"    Win rate: {test_metrics.win_rate*100:.1f}%")
            print(f"    PF: {test_metrics.profit_factor:.2f}")
            print(f"    Sharpe: {test_metrics.sharpe_ratio:.3f}")
            print(f"    MDD: {test_metrics.max_drawdown_pct*100:.1f}%")

        # Aggregate test results
        if test_results:
            print(f"\n{'='*60}")
            print("WALK-FORWARD SUMMARY (out-of-sample only)")
            print(f"{'='*60}")

            total_trades = sum(m.total_trades for m in test_results)
            total_pnl = sum(m.total_pnl for m in test_results)
            avg_sharpe = sum(m.sharpe_ratio for m in test_results) / len(test_results)
            avg_pf = sum(
                m.profit_factor for m in test_results
                if m.profit_factor != float("inf")
            ) / max(1, sum(1 for m in test_results if m.profit_factor != float("inf")))
            avg_wr = sum(m.win_rate for m in test_results) / len(test_results)
            worst_mdd = max(m.max_drawdown_pct for m in test_results)

            print(f"  Windows: {len(test_results)}")
            print(f"  Total trades: {total_trades}")
            print(f"  Total PnL: {total_pnl:.2f}")
            print(f"  Avg Sharpe: {avg_sharpe:.3f}")
            print(f"  Avg PF: {avg_pf:.2f}")
            print(f"  Avg win rate: {avg_wr*100:.1f}%")
            print(f"  Worst MDD: {worst_mdd*100:.1f}%")

            profitable_windows = sum(1 for m in test_results if m.total_pnl > 0)
            print(f"  Profitable windows: {profitable_windows}/{len(test_results)}")

    finally:
        await db.disconnect()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Walk-forward validation")
    parser.add_argument("--instrument", type=str, default="EUR/USD")
    parser.add_argument("--train-months", type=int, default=4)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--trials", type=int, default=30,
                        help="Optuna trials per window (default: 30)")
    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument("--max-mdd", type=float, default=20.0)

    args = parser.parse_args()
    asyncio.run(run_walk_forward(
        args.instrument, args.train_months, args.test_months,
        args.trials, args.capital, args.max_mdd,
    ))


if __name__ == "__main__":
    main()
