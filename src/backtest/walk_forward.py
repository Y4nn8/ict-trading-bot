"""Walk-forward validation engine.

Shared logic for all walk-forward scripts. Each script provides:
- A param_builder: Callable[[optuna.Trial], StrategyParams]
- A param_reconstructor: Callable[[dict], StrategyParams]

Usage from scripts:
    from src.backtest.walk_forward import run_walk_forward_cli

    run_walk_forward_cli(
        param_builder=StrategyParams.from_optuna_trial_smart,
        param_reconstructor=StrategyParams.from_smart_dict,
        description="Walk-forward validation (smart)",
    )
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import optuna
import polars as pl
from dotenv import load_dotenv

from src.backtest.engine import BacktestEngine
from src.backtest.metrics import (
    PerformanceMetrics,
    SourceBreakdown,
    compute_metrics,
    compute_metrics_by_source,
)
from src.backtest.vectorized import precompute
from src.common.config import load_config
from src.common.db import Database
from src.common.logging import get_logger, setup_logging
from src.market_data.storage import CandleStorage
from src.news.store import NewsStore
from src.strategy.factory import build_strategy
from src.strategy.params import StrategyParams

if TYPE_CHECKING:
    from datetime import datetime

    from src.common.models import Trade

load_dotenv()

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    """A single walk-forward train/test window."""

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_metrics: PerformanceMetrics
    test_metrics: PerformanceMetrics


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward validation results."""

    windows: list[WalkForwardWindow] = field(default_factory=list)
    mean_sharpe: float = 0.0
    mean_sortino: float = 0.0
    mean_profit_factor: float = 0.0
    mean_win_rate: float = 0.0
    mean_avg_r: float = 0.0
    worst_window_mdd: float = 0.0
    total_test_trades: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_windows(
    start: datetime,
    end: datetime,
    train_months: int = 4,
    test_months: int = 1,
    step_months: int = 1,
) -> list[tuple[datetime, datetime, datetime, datetime]]:
    """Generate walk-forward window boundaries (month-based)."""
    windows: list[tuple[datetime, datetime, datetime, datetime]] = []
    current = start
    while True:
        train_start = current
        train_end = _add_months(train_start, train_months)
        test_start = train_end
        test_end = _add_months(test_start, test_months)
        if test_end > end:
            break
        windows.append((train_start, train_end, test_start, test_end))
        current = _add_months(current, step_months)
    return windows


def aggregate_walk_forward(
    windows: list[WalkForwardWindow],
) -> WalkForwardResult:
    """Aggregate metrics across walk-forward windows."""
    if not windows:
        return WalkForwardResult()
    test_metrics = [w.test_metrics for w in windows]
    sharpes = [m.sharpe_ratio for m in test_metrics]
    sortinos = [m.sortino_ratio for m in test_metrics]
    pfs = [m.profit_factor for m in test_metrics if m.profit_factor != float("inf")]
    win_rates = [m.win_rate for m in test_metrics]
    avg_rs = [m.avg_r_multiple for m in test_metrics]
    mdds = [m.max_drawdown_pct for m in test_metrics]
    total_trades = sum(m.total_trades for m in test_metrics)
    return WalkForwardResult(
        windows=windows,
        mean_sharpe=float(np.mean(sharpes)) if sharpes else 0.0,
        mean_sortino=float(np.mean(sortinos)) if sortinos else 0.0,
        mean_profit_factor=float(np.mean(pfs)) if pfs else 0.0,
        mean_win_rate=float(np.mean(win_rates)) if win_rates else 0.0,
        mean_avg_r=float(np.mean(avg_rs)) if avg_rs else 0.0,
        worst_window_mdd=max(mdds) if mdds else 0.0,
        total_test_trades=total_trades,
    )


def split_trades_by_time(
    trades: list[Trade],
    start: datetime,
    end: datetime,
) -> list[Trade]:
    """Filter trades that fall within a time window."""
    return [t for t in trades if t.opened_at >= start and t.opened_at < end]


def _add_months(dt: datetime, months: int) -> datetime:
    """Add months to a datetime, handling year rollover."""
    month = dt.month + months
    year = dt.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    day = min(dt.day, 28)
    return dt.replace(year=year, month=month, day=day)


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

_EMPTY_BREAKDOWN = None  # Sentinel, built lazily


def _empty_breakdown(initial_capital: float) -> SourceBreakdown:
    empty = compute_metrics([], initial_capital)
    return SourceBreakdown(ict=empty, news=empty, ict_trade_count=0, news_trade_count=0)


def run_backtest(
    candles: pl.DataFrame,
    instrument: str,
    params: StrategyParams,
    news_events: list[dict[str, object]],
    initial_capital: float,
    leverage: float = 30.0,
    value_per_point: float = 1.0,
    min_size: float = 0.5,
    avg_spread: float = 0.0,
    pip_size: float = 0.0001,
) -> tuple[list[object], PerformanceMetrics, SourceBreakdown]:
    """Run a single backtest, return trades, metrics, and source breakdown."""
    if candles.is_empty():
        empty = compute_metrics([], initial_capital)
        return [], empty, _empty_breakdown(initial_capital)

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
        leverage=leverage,
        value_per_point=value_per_point,
        min_size=min_size,
        avg_spread=avg_spread,
        pip_size=pip_size,
        news_events=news_events,
    )
    result = engine.run()
    metrics = compute_metrics(result.trades, initial_capital)
    breakdown = compute_metrics_by_source(result.trades, initial_capital)
    return result.trades, metrics, breakdown


def score_trial(metrics: PerformanceMetrics, max_mdd_pct: float) -> float:
    """Composite Optuna objective score."""
    if metrics.total_trades < 5:
        return -10.0
    if metrics.max_drawdown_pct * 100 > max_mdd_pct:
        return -10.0
    pnl_norm = metrics.total_pnl / 5000
    sharpe = max(metrics.sharpe_ratio, 0)
    pf = min(metrics.profit_factor, 5.0) if metrics.profit_factor > 0 else 0
    return pnl_norm * (1 + sharpe) * (1 + pf * 0.2)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_window_results(
    test_metrics: PerformanceMetrics,
    test_breakdown: SourceBreakdown,
    train_metrics: PerformanceMetrics,
) -> None:
    """Print results for a single walk-forward window."""
    print("\n  TRAIN results (in-sample):")
    print(f"    Trades: {train_metrics.total_trades}")
    print(f"    PnL: {train_metrics.total_pnl:.2f}")
    print(f"    Sharpe: {train_metrics.sharpe_ratio:.3f}")
    print(f"    MDD: {train_metrics.max_drawdown_pct*100:.1f}%")

    print("\n  TEST results (out-of-sample):")
    print(f"    Trades: {test_metrics.total_trades}")
    print(f"    PnL: {test_metrics.total_pnl:.2f}")
    print(f"    Win rate: {test_metrics.win_rate*100:.1f}%")
    print(f"    PF: {test_metrics.profit_factor:.2f}")
    print(f"    Sharpe: {test_metrics.sharpe_ratio:.3f}")
    print(f"    MDD: {test_metrics.max_drawdown_pct*100:.1f}%")
    print(f"    Avg R: {test_metrics.avg_r_multiple:.2f}")
    print(f"    Avg risk/trade: {test_metrics.avg_risk_pct:.2f}%")

    ict = test_breakdown.ict
    news = test_breakdown.news
    print(f"    --- ICT: {test_breakdown.ict_trade_count} trades, "
          f"PnL {ict.total_pnl:.2f}, "
          f"WR {ict.win_rate*100:.1f}%, "
          f"PF {ict.profit_factor:.2f}")
    print(f"    --- News: {test_breakdown.news_trade_count} trades, "
          f"PnL {news.total_pnl:.2f}, "
          f"WR {news.win_rate*100:.1f}%, "
          f"PF {news.profit_factor:.2f}")


def print_summary(
    test_results: list[PerformanceMetrics],
    all_test_breakdowns: list[SourceBreakdown],
) -> None:
    """Print aggregated walk-forward summary."""
    if not test_results:
        return

    print(f"\n{'='*60}")
    print("WALK-FORWARD SUMMARY (out-of-sample only)")
    print(f"{'='*60}")

    total_trades = sum(m.total_trades for m in test_results)
    total_pnl = sum(m.total_pnl for m in test_results)
    avg_sharpe = sum(m.sharpe_ratio for m in test_results) / len(test_results)
    finite_pfs = [m.profit_factor for m in test_results if m.profit_factor != float("inf")]
    avg_pf = sum(finite_pfs) / len(finite_pfs) if finite_pfs else 0.0
    avg_wr = sum(m.win_rate for m in test_results) / len(test_results)
    worst_mdd = max(m.max_drawdown_pct for m in test_results)
    avg_r = sum(m.avg_r_multiple for m in test_results) / len(test_results)
    avg_risk = sum(m.avg_risk_pct for m in test_results) / len(test_results)

    print(f"  Windows: {len(test_results)}")
    print(f"  Total trades: {total_trades}")
    print(f"  Total PnL: {total_pnl:.2f}")
    print(f"  Avg Sharpe: {avg_sharpe:.3f}")
    print(f"  Avg PF: {avg_pf:.2f}")
    print(f"  Avg win rate: {avg_wr*100:.1f}%")
    print(f"  Avg R: {avg_r:.2f}")
    print(f"  Avg risk/trade: {avg_risk:.2f}%")
    print(f"  Worst MDD: {worst_mdd*100:.1f}%")

    profitable = sum(1 for m in test_results if m.total_pnl > 0)
    print(f"  Profitable windows: {profitable}/{len(test_results)}")

    # Source breakdown
    if all_test_breakdowns:
        _print_source_breakdown(all_test_breakdowns)


def _print_source_breakdown(breakdowns: list[SourceBreakdown]) -> None:
    """Print ICT vs News trigger source comparison."""
    print(f"\n{'='*60}")
    print("TRIGGER SOURCE BREAKDOWN (out-of-sample)")
    print(f"{'='*60}")

    total_ict = sum(b.ict_trade_count for b in breakdowns)
    total_news = sum(b.news_trade_count for b in breakdowns)
    pnl_ict = sum(b.ict.total_pnl for b in breakdowns)
    pnl_news = sum(b.news.total_pnl for b in breakdowns)

    ict_wins = sum(b.ict.winning_trades for b in breakdowns)
    news_wins = sum(b.news.winning_trades for b in breakdowns)
    wr_ict = ict_wins / total_ict * 100 if total_ict > 0 else 0
    wr_news = news_wins / total_news * 100 if total_news > 0 else 0

    ict_gp = sum(b.ict.avg_winner * b.ict.winning_trades for b in breakdowns)
    ict_gl = abs(sum(b.ict.avg_loser * b.ict.losing_trades for b in breakdowns))
    news_gp = sum(b.news.avg_winner * b.news.winning_trades for b in breakdowns)
    news_gl = abs(sum(b.news.avg_loser * b.news.losing_trades for b in breakdowns))
    pf_ict = ict_gp / ict_gl if ict_gl > 0 else 0
    pf_news = news_gp / news_gl if news_gl > 0 else 0

    avg_r_ict = (
        sum(b.ict.avg_r_multiple * b.ict_trade_count for b in breakdowns) / total_ict
        if total_ict > 0 else 0
    )
    avg_r_news = (
        sum(b.news.avg_r_multiple * b.news_trade_count for b in breakdowns) / total_news
        if total_news > 0 else 0
    )

    total = total_ict + total_news
    pnl_total = pnl_ict + pnl_news

    print(f"  {'':20} {'ICT':>12} {'News':>12}")
    print(f"  {'Trades':20} {total_ict:>12} {total_news:>12}")
    print(f"  {'PnL':20} {pnl_ict:>12.2f} {pnl_news:>12.2f}")
    print(f"  {'Win rate':20} {wr_ict:>11.1f}% {wr_news:>11.1f}%")
    print(f"  {'Profit factor':20} {pf_ict:>12.2f} {pf_news:>12.2f}")
    print(f"  {'Avg R':20} {avg_r_ict:>12.2f} {avg_r_news:>12.2f}")
    print(f"  {'% of trades':20} "
          f"{total_ict/total*100 if total else 0:>11.1f}% "
          f"{total_news/total*100 if total else 0:>11.1f}%")
    print(f"  {'% of PnL':20} "
          f"{pnl_ict/pnl_total*100 if pnl_total else 0:>11.1f}% "
          f"{pnl_news/pnl_total*100 if pnl_total else 0:>11.1f}%")


def print_convergence(all_best_params: list[dict[str, object]]) -> None:
    """Print parameter convergence analysis."""
    if len(all_best_params) < 2:
        return

    print(f"\n{'='*60}")
    print("PARAMETER CONVERGENCE ANALYSIS")
    print(f"{'='*60}")

    all_keys = sorted(all_best_params[0].keys())
    converging: list[tuple[str, float]] = []
    diverging: list[tuple[str, float]] = []

    for key in all_keys:
        values = [p[key] for p in all_best_params if key in p]
        if any(isinstance(v, bool) for v in values):
            agrees = len(set(values)) == 1
            label = f"{values[0]}" if agrees else " vs ".join(str(v) for v in values)
            (converging if agrees else diverging).append((f"{key} = {label}", 0.0 if agrees else 1.0))
            continue
        if not all(isinstance(v, (int, float)) for v in values):
            continue

        fvalues = [float(v) for v in values]
        mean = sum(fvalues) / len(fvalues)
        spread = max(fvalues) - min(fvalues)
        cv = spread / abs(mean) if abs(mean) > 1e-9 else spread
        vals_str = ", ".join(f"{v:.4g}" for v in fvalues)

        bucket = converging if cv < 0.3 else diverging
        bucket.append((f"{key}: {vals_str}  (spread {cv:.0%})", cv))

    print(f"\n  CONVERGING (spread < 30% of mean) — {len(converging)} params:")
    for label, _cv in sorted(converging, key=lambda x: x[1]):
        print(f"    {label}")

    print(f"\n  DIVERGING (spread >= 30% of mean) — {len(diverging)} params:")
    for label, _cv in sorted(diverging, key=lambda x: -x[1]):
        print(f"    {label}")

    total = len(converging) + len(diverging)
    print(f"\n  Convergence ratio: {len(converging)}/{total}"
          f" ({len(converging)/total*100:.0f}%)")


# ---------------------------------------------------------------------------
# Main walk-forward loop
# ---------------------------------------------------------------------------

async def run_walk_forward(
    instrument: str,
    train_months: int,
    test_months: int,
    trials_per_window: int,
    initial_capital: float,
    max_mdd_pct: float,
    param_builder: Callable[[optuna.Trial], StrategyParams],
    param_reconstructor: Callable[[dict[str, Any]], StrategyParams],
    test_weeks: int | None = None,
) -> None:
    """Run walk-forward validation with Optuna optimization.

    Args:
        instrument: Instrument name (e.g. "EUR/USD").
        train_months: Training window in months.
        test_months: Test window in months (if test_weeks is None).
        trials_per_window: Optuna trials per window.
        initial_capital: Starting capital.
        max_mdd_pct: Max drawdown % threshold for scoring.
        param_builder: Creates StrategyParams from an Optuna trial.
        param_reconstructor: Rebuilds StrategyParams from best trial params dict.
        test_weeks: Test window in weeks (overrides test_months).
    """
    config = load_config()
    setup_logging(config.logging.level, json_format=False)

    db = Database(config.database)
    await db.connect()
    storage = CandleStorage(db)
    news_store = NewsStore(db)

    try:
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

        all_news = await news_store.get_events(data_start, data_end)
        await logger.ainfo("news_loaded", count=len(all_news))

        # Instrument specs
        inst_config = config.get_instrument(instrument)
        leverage = float(inst_config.leverage) if inst_config else 30.0
        value_per_point = float(inst_config.value_per_point) if inst_config else 1.0
        min_size = float(inst_config.min_size) if inst_config else 0.5
        pip_size = float(inst_config.pip_size) if inst_config else 0.0001
        avg_spread = float(inst_config.avg_spread) * pip_size if inst_config else 0.0
        await logger.ainfo(
            "instrument_specs",
            instrument=instrument,
            leverage=leverage,
            value_per_point=value_per_point,
            min_size=min_size,
            pip_size=pip_size,
            avg_spread_price=avg_spread,
        )

        # Generate windows
        if test_weeks is not None:
            test_delta = timedelta(weeks=test_weeks)
        else:
            test_delta = timedelta(days=test_months * 30)
        train_delta = timedelta(days=train_months * 30)
        window_size = train_delta + test_delta
        step = test_delta

        windows: list[tuple[Any, Any, Any, Any]] = []
        current = data_start
        while current + window_size <= data_end:
            train_start = current
            train_end = current + train_delta
            test_start = train_end
            test_end = test_start + test_delta
            windows.append((train_start, train_end, test_start, test_end))
            current += step

        if not windows:
            await logger.aerror("not_enough_data_for_walk_forward")
            return

        await logger.ainfo("windows_generated", count=len(windows))

        # Shared backtest kwargs
        bt_kwargs: dict[str, Any] = {
            "initial_capital": initial_capital,
            "leverage": leverage,
            "value_per_point": value_per_point,
            "min_size": min_size,
            "avg_spread": avg_spread,
            "pip_size": pip_size,
        }

        # Run each window
        test_results: list[PerformanceMetrics] = []
        all_test_breakdowns: list[SourceBreakdown] = []
        all_best_params: list[dict[str, object]] = []
        prev_best_trial_params: dict[str, object] | None = None

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            print(f"\n{'='*60}")
            print(f"WINDOW {i+1}/{len(windows)}")
            print(f"  Train: {train_start.date()} → {train_end.date()}")
            print(f"  Test:  {test_start.date()} → {test_end.date()}")
            print(f"{'='*60}")

            train_candles = all_candles.filter(
                (pl.col("time") >= train_start) & (pl.col("time") < train_end)
            )
            test_candles = all_candles.filter(
                (pl.col("time") >= test_start) & (pl.col("time") < test_end)
            )

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

            # Optimize on train
            def objective(
                trial: optuna.Trial,
                _candles: pl.DataFrame = train_candles,
                _news: list[dict[str, object]] = train_news,
                _leverage: float = leverage,
            ) -> float:
                params = param_builder(trial)
                _, metrics, _ = run_backtest(
                    _candles, instrument, params, _news, **bt_kwargs,
                )
                return score_trial(metrics, max_mdd_pct)

            study = optuna.create_study(direction="maximize")
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            # Warm-start: seed with previous window's best params
            if prev_best_trial_params is not None:
                study.enqueue_trial(prev_best_trial_params)
                print("  Warm-start: seeded with previous window's best params")

            study.optimize(objective, n_trials=trials_per_window, show_progress_bar=True)

            prev_best_trial_params = study.best_trial.params
            best_params = param_reconstructor(study.best_trial.params)
            all_best_params.append(study.best_trial.params)

            best_trial = study.best_trial
            print(f"\n  Best params (trial #{best_trial.number}, score={best_trial.value:.4f}):")
            for key, value in sorted(best_trial.params.items()):
                print(f"    {key}: {value}")

            # Evaluate on train and test
            _, train_metrics, _ = run_backtest(
                train_candles, instrument, best_params, train_news, **bt_kwargs,
            )
            _, test_metrics, test_breakdown = run_backtest(
                test_candles, instrument, best_params, test_news, **bt_kwargs,
            )
            test_results.append(test_metrics)
            all_test_breakdowns.append(test_breakdown)

            print_window_results(test_metrics, test_breakdown, train_metrics)

        # Aggregated reports
        print_summary(test_results, all_test_breakdowns)
        print_convergence(all_best_params)

    finally:
        await db.disconnect()


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def run_walk_forward_cli(
    param_builder: Callable[[optuna.Trial], StrategyParams],
    param_reconstructor: Callable[[dict[str, Any]], StrategyParams],
    description: str = "Walk-forward validation",
    default_trials: int = 30,
) -> None:
    """Parse CLI args and run walk-forward validation.

    Args:
        param_builder: Creates StrategyParams from an Optuna trial.
        param_reconstructor: Rebuilds StrategyParams from best trial params dict.
        description: CLI description.
        default_trials: Default number of Optuna trials per window.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--instrument", type=str, default="EUR/USD")
    parser.add_argument("--train-months", type=int, default=4)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--trials", type=int, default=default_trials,
                        help=f"Optuna trials per window (default: {default_trials})")
    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument("--max-mdd", type=float, default=20.0)
    parser.add_argument("--test-weeks", type=int, default=None,
                        help="Test period in weeks (overrides --test-months)")

    args = parser.parse_args()
    asyncio.run(run_walk_forward(
        args.instrument, args.train_months, args.test_months,
        args.trials, args.capital, args.max_mdd,
        param_builder=param_builder,
        param_reconstructor=param_reconstructor,
        test_weeks=args.test_weeks,
    ))
