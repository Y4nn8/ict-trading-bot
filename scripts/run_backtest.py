"""Run a backtest on historical data from the database.

Usage:
    uv run python -m scripts.run_backtest --instrument EUR/USD --days 60
    uv run python -m scripts.run_backtest --instrument DAX40 --days 30
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime, timedelta

from dotenv import load_dotenv

from src.backtest.engine import BacktestEngine
from src.backtest.report import format_report, generate_report
from src.backtest.vectorized import precompute
from src.common.config import load_config
from src.common.db import Database
from src.common.logging import get_logger, setup_logging
from src.market_data.storage import CandleStorage
from src.news.store import NewsStore
from src.strategy.factory import build_strategy
from src.strategy.params import StrategyParams

logger = get_logger(__name__)


async def run_backtest(
    instrument: str,
    days: int,
    initial_capital: float,
    params: StrategyParams | None = None,
) -> None:
    """Run a full backtest pipeline.

    Args:
        instrument: Instrument name.
        days: Number of days of data to use.
        initial_capital: Starting capital.
        params: Strategy parameters. Uses defaults if None.
    """
    if params is None:
        params = StrategyParams()

    config = load_config()
    setup_logging(config.logging.level, json_format=False)

    db = Database(config.database)
    await db.connect()
    storage = CandleStorage(db)

    try:
        end = datetime.now(tz=UTC)
        start = end - timedelta(days=days)

        await logger.ainfo(
            "fetching_candles",
            instrument=instrument,
            start=start.isoformat(),
            end=end.isoformat(),
        )

        candles = await storage.fetch_candles(instrument, "M5", start=start, end=end)

        if candles.is_empty():
            await logger.aerror("no_candles_found", instrument=instrument)
            return

        await logger.ainfo("candles_fetched", count=len(candles))

        # Pre-compute with strategy params
        await logger.ainfo("precomputing_detectors")
        precomputed = precompute(candles, instrument, "M5", params=params)

        await logger.ainfo(
            "precompute_complete",
            swings=len(precomputed.swings),
            fvgs=len(precomputed.fvgs),
            order_blocks=len(precomputed.order_blocks),
            ms_breaks=len(precomputed.market_structure),
            displacements=len(precomputed.displacements),
        )

        # Load pre-interpreted news events
        news_store = NewsStore(db)
        news_events = await news_store.get_events(start, end)
        await logger.ainfo("news_events_loaded", count=len(news_events))

        inst_config = config.get_instrument(instrument)
        leverage = float(inst_config.leverage) if inst_config else 30.0
        value_per_point = float(inst_config.value_per_point) if inst_config else 1.0
        min_size = float(inst_config.min_size) if inst_config else 0.5
        pip_size = float(inst_config.pip_size) if inst_config else 0.0001
        avg_spread = float(inst_config.avg_spread) * pip_size if inst_config else 0.0

        # Build strategy components from params
        components = build_strategy(params)

        # Run backtest
        await logger.ainfo(
            "running_backtest",
            candles=len(candles),
            leverage=leverage,
            value_per_point=value_per_point,
            min_size=min_size,
        )
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
            news_events=news_events,
        )
        result = engine.run()

        await logger.ainfo(
            "backtest_complete",
            closed_trades=len(result.trades),
            open_positions=len(result.open_positions),
            margin_rejected=result.margin_rejected,
            margin_capped=result.margin_capped,
            peak_margin_usage_pct=f"{result.peak_margin_usage_pct:.1f}%",
        )

        report = generate_report(result.trades, initial_capital)
        print("\n" + format_report(report))
        print(f"  Margin rejected: {result.margin_rejected}")
        print(f"  Margin capped: {result.margin_capped}")
        print(f"  Peak margin usage: {result.peak_margin_usage_pct:.1f}%")

        if result.trades:
            print(f"\n--- First 20 of {len(result.trades)} trades ---")
            for i, trade in enumerate(result.trades[:20], 1):
                print(
                    f"  #{i}: {trade.direction} "
                    f"entry={trade.entry_price:.5f} "
                    f"exit={trade.exit_price:.5f} "
                    f"PnL={trade.pnl:.2f} R={trade.r_multiple:.2f}"
                )

    finally:
        await db.disconnect()


def main() -> None:
    """CLI entry point."""
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run backtest on historical data")
    parser.add_argument(
        "--instrument", type=str, default="EUR/USD",
    )
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--capital", type=float, default=5000.0)

    args = parser.parse_args()
    asyncio.run(run_backtest(args.instrument, args.days, args.capital))


if __name__ == "__main__":
    main()
