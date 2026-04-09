"""Ablation test: compare OOS results with and without news module.

Loads best params from config/best_params.yml and replays all
walk-forward OOS windows twice — once with news, once without.
Prints a side-by-side comparison table.

Usage:
    uv run python -m scripts.ablation_news \
        --instrument XAUUSD --train-months 5 --test-weeks 1
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import timedelta
from typing import Any

import polars as pl
import yaml
from dotenv import load_dotenv

from src.backtest.metrics import PerformanceMetrics, compute_metrics_by_source
from src.backtest.walk_forward import run_backtest
from src.common.config import load_config
from src.common.db import Database
from src.common.logging import setup_logging
from src.market_data.storage import CandleStorage
from src.news.store import NewsStore
from src.strategy.params import StrategyParams


def _load_best_params(path: str = "config/best_params.yml") -> StrategyParams:
    """Load best params from YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return StrategyParams.from_dict(raw)


def _print_comparison(
    windows: list[tuple[Any, Any, Any, Any]],
    with_news: list[PerformanceMetrics],
    without_news: list[PerformanceMetrics],
) -> None:
    """Print side-by-side comparison of with/without news."""
    print(f"\n{'='*80}")
    print("ABLATION: NEWS MODULE IMPACT (out-of-sample)")
    print(f"{'='*80}")

    header = (
        f"  {'Window':<8} {'Dates':<25} "
        f"{'Trades':>7} {'PnL':>9} {'WR':>6} {'PF':>6} {'Sharpe':>7} "
        f"{'Trades':>7} {'PnL':>9} {'WR':>6} {'PF':>6} {'Sharpe':>7}"
    )
    print(f"  {'':33} {'WITH NEWS':^40} {'WITHOUT NEWS':^40}")
    print(header)
    print(f"  {'-'*len(header.strip())}")

    for i, (wn, won) in enumerate(zip(with_news, without_news)):
        dates = ""
        if i < len(windows):
            dates = f"{windows[i][2].date()} → {windows[i][3].date()}"
        print(
            f"  W{i+1:<7} {dates:<25} "
            f"{wn.total_trades:>7} {wn.total_pnl:>+9.0f} "
            f"{wn.win_rate*100:>5.1f}% {wn.profit_factor:>5.2f} "
            f"{wn.sharpe_ratio:>7.2f} "
            f"{won.total_trades:>7} {won.total_pnl:>+9.0f} "
            f"{won.win_rate*100:>5.1f}% {won.profit_factor:>5.2f} "
            f"{won.sharpe_ratio:>7.2f}"
        )

    # Aggregates
    def _agg(results: list[PerformanceMetrics]) -> dict[str, float]:
        total_trades = sum(m.total_trades for m in results)
        total_pnl = sum(m.total_pnl for m in results)
        avg_sharpe = sum(m.sharpe_ratio for m in results) / len(results)
        finite_pfs = [m.profit_factor for m in results if m.profit_factor != float("inf")]
        avg_pf = sum(finite_pfs) / len(finite_pfs) if finite_pfs else 0.0
        avg_wr = sum(m.win_rate for m in results) / len(results)
        worst_mdd = max(m.max_drawdown_pct for m in results)
        profitable = sum(1 for m in results if m.total_pnl > 0)
        return {
            "trades": total_trades,
            "pnl": total_pnl,
            "sharpe": avg_sharpe,
            "pf": avg_pf,
            "wr": avg_wr,
            "mdd": worst_mdd,
            "profitable": profitable,
            "total": len(results),
        }

    wn_agg = _agg(with_news)
    won_agg = _agg(without_news)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  {'':25} {'With News':>15} {'Without News':>15} {'Delta':>15}")
    print(f"  {'-'*70}")
    print(f"  {'Total trades':25} {wn_agg['trades']:>15.0f} {won_agg['trades']:>15.0f} {wn_agg['trades']-won_agg['trades']:>+15.0f}")
    print(f"  {'Total PnL':25} {wn_agg['pnl']:>15.2f} {won_agg['pnl']:>15.2f} {wn_agg['pnl']-won_agg['pnl']:>+15.2f}")
    print(f"  {'Avg Sharpe':25} {wn_agg['sharpe']:>15.3f} {won_agg['sharpe']:>15.3f} {wn_agg['sharpe']-won_agg['sharpe']:>+15.3f}")
    print(f"  {'Avg PF':25} {wn_agg['pf']:>15.2f} {won_agg['pf']:>15.2f} {wn_agg['pf']-won_agg['pf']:>+15.2f}")
    print(f"  {'Avg Win Rate':25} {wn_agg['wr']*100:>14.1f}% {won_agg['wr']*100:>14.1f}% {(wn_agg['wr']-won_agg['wr'])*100:>+14.1f}%")
    print(f"  {'Worst MDD':25} {wn_agg['mdd']*100:>14.1f}% {won_agg['mdd']*100:>14.1f}%")
    print(f"  {'Profitable windows':25} {wn_agg['profitable']}/{wn_agg['total']:>11} {won_agg['profitable']}/{won_agg['total']:>11}")

    # Verdict
    pnl_delta_pct = ((wn_agg["pnl"] - won_agg["pnl"]) / abs(won_agg["pnl"]) * 100) if won_agg["pnl"] != 0 else 0
    print(f"\n  News module PnL impact: {pnl_delta_pct:+.1f}%")
    if wn_agg["pnl"] > won_agg["pnl"]:
        print("  → News module HELPS (+PnL)")
    elif wn_agg["pnl"] < won_agg["pnl"]:
        print("  → News module HURTS (-PnL) — consider disabling")
    else:
        print("  → News module has NO IMPACT")


async def run_ablation(
    instrument: str,
    train_months: int,
    test_weeks: int,
    initial_capital: float,
    params_path: str,
) -> None:
    """Run A/B comparison with and without news."""
    config = load_config()
    setup_logging(config.logging.level, json_format=False)

    params = _load_best_params(params_path)
    print(f"Loaded params from {params_path}")

    db = Database(config.database)
    await db.connect()
    storage = CandleStorage(db)
    news_store = NewsStore(db)

    try:
        all_candles = await storage.fetch_candles(instrument, "M5")
        if all_candles.is_empty():
            print(f"No candles for {instrument}")
            return

        data_start = all_candles["time"][0]
        data_end = all_candles["time"][-1]
        print(f"Data: {len(all_candles)} candles, {data_start.date()} → {data_end.date()}")

        all_news = await news_store.get_events(data_start, data_end)
        print(f"News: {len(all_news)} events")

        all_h1 = await storage.fetch_candles(instrument, "H1")
        print(f"H1: {len(all_h1)} candles")

        # Instrument specs
        inst_config = config.get_instrument(instrument)
        leverage = float(inst_config.leverage) if inst_config else 30.0
        value_per_point = float(inst_config.value_per_point) if inst_config else 1.0
        min_size = float(inst_config.min_size) if inst_config else 0.5
        size_step = float(inst_config.size_step) if inst_config else 0.5
        pip_size = float(inst_config.pip_size) if inst_config else 0.0001
        avg_spread = float(inst_config.avg_spread) * pip_size if inst_config else 0.0
        min_stop_distance = float(inst_config.min_stop_distance) * pip_size if inst_config else 0.0

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

        # Generate OOS windows (test-only, no optimization)
        train_delta = timedelta(days=train_months * 30)
        test_delta = timedelta(weeks=test_weeks)
        h1_lookback = timedelta(days=30)

        windows: list[tuple[Any, Any, Any, Any]] = []
        current = data_start
        while current + train_delta + test_delta <= data_end:
            train_end = current + train_delta
            test_start = train_end
            test_end = test_start + test_delta
            windows.append((current, train_end, test_start, test_end))
            current += test_delta

        print(f"Windows: {len(windows)}")

        results_with_news: list[PerformanceMetrics] = []
        results_without_news: list[PerformanceMetrics] = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            print(f"\n  Window {i+1}/{len(windows)}: {test_start.date()} → {test_end.date()}")

            test_candles = all_candles.filter(
                (pl.col("time") >= test_start) & (pl.col("time") < test_end)
            )
            test_h1 = all_h1.filter(
                (pl.col("time") >= test_start - h1_lookback) & (pl.col("time") < test_end)
            ) if not all_h1.is_empty() else all_h1

            test_news = [
                n for n in all_news
                if test_start <= n.get("time", data_start) < test_end
            ]

            if test_candles.is_empty():
                print("    SKIP: no candles")
                continue

            # WITH news
            _, metrics_wn, _ = run_backtest(
                test_candles, instrument, params, test_news,
                **bt_kwargs, h1_candles=test_h1,
            )
            results_with_news.append(metrics_wn)

            # WITHOUT news
            _, metrics_won, _ = run_backtest(
                test_candles, instrument, params, [],
                **bt_kwargs, h1_candles=test_h1,
            )
            results_without_news.append(metrics_won)

            print(f"    With news:    {metrics_wn.total_trades} trades, "
                  f"PnL {metrics_wn.total_pnl:+.0f}, PF {metrics_wn.profit_factor:.2f}")
            print(f"    Without news: {metrics_won.total_trades} trades, "
                  f"PnL {metrics_won.total_pnl:+.0f}, PF {metrics_won.profit_factor:.2f}")

        _print_comparison(windows, results_with_news, results_without_news)

    finally:
        await db.disconnect()


def main() -> None:
    """CLI entry point."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Ablation test: news module impact on OOS performance",
    )
    parser.add_argument("--instrument", type=str, default="EUR/USD")
    parser.add_argument("--train-months", type=int, default=5)
    parser.add_argument("--test-weeks", type=int, default=1)
    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument("--params", type=str, default="config/best_params.yml",
                        help="Path to best params YAML file")
    args = parser.parse_args()

    asyncio.run(run_ablation(
        args.instrument, args.train_months, args.test_weeks,
        args.capital, args.params,
    ))


if __name__ == "__main__":
    main()
