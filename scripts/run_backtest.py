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
from src.backtest.simulator import SimulationConfig
from src.backtest.vectorized import precompute
from src.common.config import load_config
from src.common.db import Database
from src.common.logging import get_logger, setup_logging
from src.execution.position_sizer import PositionSizer
from src.execution.risk_manager import RiskManager
from src.market_data.storage import CandleStorage
from src.strategy.confluence import ConfluenceScorer
from src.strategy.entry import EntryEvaluator
from src.strategy.exit import ExitEvaluator
from src.strategy.filters import TradeFilter

logger = get_logger(__name__)


async def run_backtest(
    instrument: str,
    days: int,
    initial_capital: float,
) -> None:
    """Run a full backtest pipeline.

    Args:
        instrument: Instrument name.
        days: Number of days of data to use.
        initial_capital: Starting capital.
    """
    config = load_config()
    setup_logging(config.logging.level, json_format=False)

    db = Database(config.database)
    await db.connect()
    storage = CandleStorage(db)

    try:
        # 1. Fetch candles from DB
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

        # 2. Pre-compute all detectors
        await logger.ainfo("precomputing_detectors")
        precomputed = precompute(candles, instrument, "M5")

        await logger.ainfo(
            "precompute_complete",
            swings=len(precomputed.swings),
            fvgs=len(precomputed.fvgs),
            order_blocks=len(precomputed.order_blocks),
            ms_breaks=len(precomputed.market_structure),
            displacements=len(precomputed.displacements),
        )

        # 3. Find instrument config for sizing params
        for ic in config.instruments:
            if ic.name == instrument:
                break

        # 4. Setup strategy components
        confluence_scorer = ConfluenceScorer()
        entry_evaluator = EntryEvaluator(
            min_confluence=config.strategy.min_confluence_score,
        )
        exit_evaluator = ExitEvaluator(max_hold_candles=72)  # 6 hours max
        trade_filter = TradeFilter(
            max_positions=config.risk.max_simultaneous_positions,
            require_killzone=True,
        )
        position_sizer = PositionSizer()
        risk_manager = RiskManager(
            max_daily_drawdown_pct=config.risk.max_daily_drawdown_pct,
            max_total_drawdown_pct=config.risk.max_total_drawdown_pct,
            max_positions=config.risk.max_simultaneous_positions,
        )
        sim_config = SimulationConfig(
            slippage_max_pips=config.backtest.simulation.slippage_max_pips,
            order_rejection_rate=config.backtest.simulation.order_rejection_rate,
        )

        # 5. Run backtest engine
        await logger.ainfo("running_backtest", candles=len(candles))
        engine = BacktestEngine(
            precomputed=precomputed,
            confluence_scorer=confluence_scorer,
            entry_evaluator=entry_evaluator,
            exit_evaluator=exit_evaluator,
            trade_filter=trade_filter,
            position_sizer=position_sizer,
            risk_manager=risk_manager,
            sim_config=sim_config,
            initial_capital=initial_capital,
        )
        result = engine.run()

        # 6. Generate report
        await logger.ainfo(
            "backtest_complete",
            closed_trades=len(result.trades),
            open_positions=len(result.open_positions),
        )

        report = generate_report(result.trades, initial_capital)
        print("\n" + format_report(report))

        # Print trade details if any
        if result.trades:
            print("\n--- Trade Details ---")
            for i, trade in enumerate(result.trades, 1):
                direction = "LONG" if trade.direction == "LONG" else "SHORT"
                print(
                    f"  #{i}: {direction} entry={trade.entry_price:.5f} "
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
        "--instrument",
        type=str,
        default="EUR/USD",
        help="Instrument to backtest (default: EUR/USD)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="Number of days of data (default: 60)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=5000.0,
        help="Initial capital (default: 5000)",
    )

    args = parser.parse_args()
    asyncio.run(run_backtest(args.instrument, args.days, args.capital))


if __name__ == "__main__":
    main()
