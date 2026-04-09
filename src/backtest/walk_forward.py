"""Walk-forward validation engine.

Shared logic for all walk-forward scripts. Each script provides:
- A param_builder: Callable[[optuna.Trial], StrategyParams]
- A param_reconstructor: Callable[[dict], StrategyParams]

Usage from scripts:
    from src.backtest.walk_forward import run_walk_forward_cli

    run_walk_forward_cli(
        param_builder=StrategyParams.from_optuna_trial_xauusd,
        param_reconstructor=StrategyParams.from_xauusd_dict,
        description="Walk-forward validation (smart)",
    )
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
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
from src.backtest.vectorized import build_htf_trend_array, precompute
from src.common.config import load_config
from src.common.db import Database
from src.common.logging import get_logger, setup_logging
from src.market_data.storage import CandleStorage
from src.news.store import NewsStore
from src.strategy.factory import build_strategy

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from src.common.models import Trade
    from src.strategy.params import StrategyParams

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
    size_step: float = 0.5,
    avg_spread: float = 0.0,
    pip_size: float = 0.0001,
    min_stop_distance: float = 0.0,
    h1_candles: pl.DataFrame | None = None,
    compute_breakdown: bool = True,
) -> tuple[list[Trade], PerformanceMetrics, SourceBreakdown]:
    """Run a single backtest, return trades, metrics, and source breakdown."""
    if candles.is_empty():
        empty = compute_metrics([], initial_capital)
        return [], empty, _empty_breakdown(initial_capital)

    precomputed = precompute(candles, instrument, "M5", params=params)

    # Overlay H1 trend on M5 candles for multi-timeframe filtering
    if h1_candles is not None and not h1_candles.is_empty():
        precomputed.htf_trend = build_htf_trend_array(
            precomputed.candles, h1_candles,
            params.swing_left_bars, params.swing_right_bars,
        )

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
        size_step=size_step,
        avg_spread=avg_spread,
        pip_size=pip_size,
        news_events=news_events,
        ms_lookback_candles=params.ms_lookback_candles,
        be_trigger_pct=params.be_trigger_pct,
        be_offset_pct=params.be_offset_pct,
        min_stop_distance=min_stop_distance,
    )
    result = engine.run()
    metrics = compute_metrics(result.trades, initial_capital)
    breakdown = (
        compute_metrics_by_source(result.trades, initial_capital)
        if compute_breakdown
        else _empty_breakdown(initial_capital)
    )
    return result.trades, metrics, breakdown


def score_trial(
    metrics: PerformanceMetrics,
    max_mdd_pct: float,
    trading_days: int = 1,
    min_trades_per_day: float = 1.0,
    initial_capital: float = 5000.0,
) -> float:
    """Composite Optuna objective: PnL * Sharpe * n_trades.

    Rewards strategies that are profitable, consistent (high Sharpe),
    AND active (many trades). Hard rejects filter out degenerate trials.
    """
    min_trades = max(5, int(trading_days * min_trades_per_day))
    if metrics.total_trades < min_trades:
        return -10.0
    if metrics.max_drawdown_pct * 100 > max_mdd_pct:
        return -10.0
    if metrics.profit_factor < 1.0:
        return -10.0
    if metrics.sharpe_ratio < 0:
        return -10.0
    pnl_norm = metrics.total_pnl / initial_capital
    return pnl_norm * metrics.sharpe_ratio * metrics.total_trades


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
    windows: list[tuple[Any, Any, Any, Any]] | None = None,
) -> None:
    """Print aggregated walk-forward summary."""
    if not test_results:
        return

    # Per-window detail table
    print(f"\n{'='*60}")
    print("PER-WINDOW OOS RESULTS")
    print(f"{'='*60}")
    header = (f"  {'Window':<8} {'Dates':<25} {'Trades':>6} "
              f"{'PnL':>9} {'WR':>6} {'PF':>6} {'Sharpe':>7} {'MDD':>6}")
    print(header)
    print(f"  {'-'*len(header.strip())}")
    for i, m in enumerate(test_results):
        if windows and i < len(windows):
            dates = f"{windows[i][2].date()} → {windows[i][3].date()}"
        else:
            dates = ""
        pnl_str = f"{m.total_pnl:+.0f}"
        print(f"  W{i+1:<7} {dates:<25} {m.total_trades:>6} {pnl_str:>9} "
              f"{m.win_rate*100:>5.1f}% {m.profit_factor:>5.2f} "
              f"{m.sharpe_ratio:>7.2f} {m.max_drawdown_pct*100:>5.1f}%")

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


def _print_compare_table(
    compare_results: dict[str, list[PerformanceMetrics]],
) -> None:
    """Print comparison of different param selection strategies."""
    print(f"\n{'='*60}")
    print("PARAM SELECTION COMPARISON (out-of-sample)")
    print(f"{'='*60}")
    header = (f"  {'Method':<10} {'Trades':>6} {'PnL':>9} {'WR':>6} "
              f"{'PF':>6} {'Sharpe':>7} {'MDD':>6} {'Win W':>6}")
    print(header)
    print(f"  {'-'*len(header.strip())}")

    for label, results in compare_results.items():
        if not results:
            continue
        total_trades = sum(m.total_trades for m in results)
        total_pnl = sum(m.total_pnl for m in results)
        avg_wr = sum(m.win_rate for m in results) / len(results)
        finite_pfs = [
            m.profit_factor for m in results
            if m.profit_factor != float("inf")
        ]
        avg_pf = sum(finite_pfs) / len(finite_pfs) if finite_pfs else 0.0
        avg_sharpe = sum(m.sharpe_ratio for m in results) / len(results)
        worst_mdd = max(m.max_drawdown_pct for m in results)
        profitable = sum(1 for m in results if m.total_pnl > 0)
        print(
            f"  {label:<10} {total_trades:>6} {total_pnl:>+9.0f} "
            f"{avg_wr*100:>5.1f}% {avg_pf:>5.2f} "
            f"{avg_sharpe:>7.2f} {worst_mdd*100:>5.1f}% "
            f"{profitable}/{len(results)}"
        )


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
            score = 0.0 if agrees else 1.0
            (converging if agrees else diverging).append((f"{key} = {label}", score))
            continue
        if not all(isinstance(v, (int, float)) for v in values):
            continue

        fvalues = [float(v) for v in values]  # type: ignore[arg-type]
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

def _median_params(
    trials: list[optuna.trial.FrozenTrial],
    top_n: int,
) -> dict[str, Any]:
    """Compute median of params from top N trials.

    For numeric params, takes the median. For categorical params,
    takes the most common value.

    Args:
        trials: Sorted trials (best first).
        top_n: Number of top trials to use.

    Returns:
        Dict of median parameter values.
    """
    import statistics
    from collections import Counter

    top = trials[:top_n]
    keys = top[0].params.keys()
    result: dict[str, Any] = {}
    for key in keys:
        values = [t.params[key] for t in top]
        if all(isinstance(v, bool) for v in values):
            result[key] = Counter(values).most_common(1)[0][0]
        elif all(isinstance(v, (int, float)) for v in values):
            median_val = statistics.median(values)
            if all(isinstance(v, int) for v in values):
                result[key] = round(median_val)
            else:
                result[key] = median_val
        else:
            result[key] = Counter(values).most_common(1)[0][0]
    return result


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
    top_n_median: int = 0,
    compare_top_n: list[int] | None = None,
    seed_params: dict[str, Any] | None = None,
    disable_news: bool = False,
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
        top_n_median: If > 0, use median of top N trials instead of best trial.
        compare_top_n: If set, evaluate multiple strategies (best + median-N)
            on each OOS window and print a comparison table.
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

        all_news = (
            [] if disable_news
            else await news_store.get_events(data_start, data_end)
        )
        await logger.ainfo("news_loaded", count=len(all_news))

        # Fetch H1 candles for multi-timeframe trend filtering
        all_h1_candles = await storage.fetch_candles(instrument, "H1")
        await logger.ainfo("h1_loaded", candles=len(all_h1_candles))

        # Instrument specs
        inst_config = config.get_instrument(instrument)
        leverage = float(inst_config.leverage) if inst_config else 30.0
        value_per_point = float(inst_config.value_per_point) if inst_config else 1.0
        min_size = float(inst_config.min_size) if inst_config else 0.5
        size_step = float(inst_config.size_step) if inst_config else 0.5
        pip_size = float(inst_config.pip_size) if inst_config else 0.0001
        avg_spread = float(inst_config.avg_spread) * pip_size if inst_config else 0.0
        # min_stop_distance is in IG points (pips); convert to price units
        min_stop_distance = float(inst_config.min_stop_distance) * pip_size if inst_config else 0.0
        await logger.ainfo(
            "instrument_specs",
            instrument=instrument,
            leverage=leverage,
            value_per_point=value_per_point,
            min_size=min_size,
            pip_size=pip_size,
            avg_spread_price=avg_spread,
            min_stop_distance=min_stop_distance,
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
            "size_step": size_step,
            "avg_spread": avg_spread,
            "pip_size": pip_size,
            "min_stop_distance": min_stop_distance,
        }

        # Run each window
        test_results: list[PerformanceMetrics] = []
        all_test_breakdowns: list[SourceBreakdown] = []
        all_best_params: list[dict[str, object]] = []
        compare_results: dict[str, list[PerformanceMetrics]] = (
            defaultdict(list) if compare_top_n else {}
        )
        prev_best_trial_params: dict[str, object] | None = (
            dict(seed_params) if seed_params else None
        )

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

            # H1 candles for HTF trend — include extra lookback for structure detection
            h1_lookback = timedelta(days=30)
            train_h1 = all_h1_candles.filter(
                (pl.col("time") >= train_start - h1_lookback) & (pl.col("time") < train_end)
            ) if not all_h1_candles.is_empty() else all_h1_candles
            test_h1 = all_h1_candles.filter(
                (pl.col("time") >= test_start - h1_lookback) & (pl.col("time") < test_end)
            ) if not all_h1_candles.is_empty() else all_h1_candles

            train_news = [
                n for n in all_news
                if train_start <= n.get("time", data_start) < train_end
            ]
            test_news = [
                n for n in all_news
                if test_start <= n.get("time", data_start) < test_end
            ]

            n_tr, n_tn, n_th = len(train_candles), len(train_news), len(train_h1)
            n_te, n_en, n_eh = len(test_candles), len(test_news), len(test_h1)
            print(f"  Train: {n_tr} candles, {n_tn} news, {n_th} H1")
            print(f"  Test:  {n_te} candles, {n_en} news, {n_eh} H1")

            if train_candles.is_empty() or test_candles.is_empty():
                print("  SKIP: insufficient data")
                continue

            # Count trading days in train window
            train_trading_days = train_candles["time"].dt.date().n_unique()

            # Optimize on train
            def objective(
                trial: optuna.Trial,
                _candles: pl.DataFrame = train_candles,
                _news: list[dict[str, object]] = train_news,
                _h1: pl.DataFrame = train_h1,
                _leverage: float = leverage,
                _trading_days: int = train_trading_days,
            ) -> float:
                params = param_builder(trial)
                _, metrics, _ = run_backtest(
                    _candles, instrument, params, _news, **bt_kwargs,
                    h1_candles=_h1, compute_breakdown=False,
                )
                return score_trial(metrics, max_mdd_pct, _trading_days,
                                   initial_capital=initial_capital)

            study = optuna.create_study(direction="maximize")
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            # Warm-start: seed with previous window's best params
            if prev_best_trial_params is not None:
                study.enqueue_trial(prev_best_trial_params)
                print("  Warm-start: seeded with previous window's best params")

            study.optimize(objective, n_trials=trials_per_window, show_progress_bar=True)

            # Select params: median of top N or single best
            viable = [t for t in study.trials
                      if t.value is not None and t.value > -10.0]
            viable.sort(key=lambda t: t.value or 0.0, reverse=True)

            if top_n_median > 0 and len(viable) >= top_n_median:
                chosen_params = _median_params(viable, top_n_median)
                best_params = param_reconstructor(chosen_params)
                best_score = viable[0].value
                print(f"\n  Median of top {top_n_median} trials "
                      f"(best score={best_score:.4f}, "
                      f"{len(viable)} viable):")
                for key, value in sorted(chosen_params.items()):
                    print(f"    {key}: {value}")
            else:
                chosen_params = study.best_trial.params
                best_params = param_reconstructor(chosen_params)
                best_trial = study.best_trial
                print(f"\n  Best params (trial #{best_trial.number}, "
                      f"score={best_trial.value:.4f}):")
                for key, value in sorted(chosen_params.items()):
                    print(f"    {key}: {value}")

            prev_best_trial_params = chosen_params
            all_best_params.append(chosen_params)

            # Evaluate variants on train and test
            if compare_top_n and viable:
                variants: list[tuple[str, int]] = [("best", 0)] + [
                    (f"med-{n}", n) for n in compare_top_n
                ]
            else:
                # No compare: just evaluate the chosen params
                variants = [(
                    f"med-{top_n_median}" if top_n_median > 0 else "best", 0,
                )]

            for vi, (label, n) in enumerate(variants):
                if n == 0 and compare_top_n:
                    vdict = viable[0].params
                elif n > 0 and len(viable) >= n:
                    vdict = _median_params(viable, n)
                else:
                    vdict = chosen_params
                vparams = param_reconstructor(vdict)

                _, train_m, _ = run_backtest(
                    train_candles, instrument, vparams, train_news,
                    **bt_kwargs, h1_candles=train_h1, compute_breakdown=False,
                )
                _, test_m, test_bd = run_backtest(
                    test_candles, instrument, vparams, test_news,
                    **bt_kwargs, h1_candles=test_h1,
                )

                print(f"\n  --- {label.upper()} ---")
                for key, value in sorted(vdict.items()):
                    print(f"    {key}: {value}")
                print_window_results(test_m, test_bd, train_m)

                if compare_top_n:
                    compare_results[label].append(test_m)

                # Use first variant (main selection) for aggregated summary
                if vi == 0:
                    test_results.append(test_m)
                    all_test_breakdowns.append(test_bd)

        # Aggregated reports
        print_summary(test_results, all_test_breakdowns, windows)
        print_convergence(all_best_params)

        # Comparison table
        if compare_top_n and compare_results:
            _print_compare_table(compare_results)

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
    load_dotenv()
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
    parser.add_argument("--top-n", type=int, default=0,
                        help="Use median of top N trials instead of best (0=off)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare best vs median-5/10/20 on OOS")
    parser.add_argument("--seed-params", type=str, default=None,
                        help="YAML file with params to warm-start W1")
    parser.add_argument("--no-news", action="store_true",
                        help="Disable news module (pass empty events)")

    args = parser.parse_args()
    compare = [5, 10, 20] if args.compare else None

    seed: dict[str, Any] | None = None
    if args.seed_params:
        import yaml
        with open(args.seed_params) as f:
            seed = yaml.safe_load(f)
        print(f"Seed params loaded from {args.seed_params}")

    asyncio.run(run_walk_forward(
        args.instrument, args.train_months, args.test_months,
        args.trials, args.capital, args.max_mdd,
        param_builder=param_builder,
        param_reconstructor=param_reconstructor,
        test_weeks=args.test_weeks,
        top_n_median=args.top_n,
        compare_top_n=compare,
        seed_params=seed,
        disable_news=args.no_news,
    ))
