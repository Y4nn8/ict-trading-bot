"""Run Midas walk-forward: train LightGBM + simulate on tick data.

Usage:
    uv run python -m scripts.run_midas_wf \
        --instrument XAUUSD \
        --start 2025-01-01 --end 2025-03-01 \
        --train-days 30 --test-days 2
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime

from dotenv import load_dotenv

from src.common.config import load_config
from src.common.db import Database
from src.common.logging import setup_logging
from src.midas.trade_simulator import SimConfig
from src.midas.trainer import TrainerConfig
from src.midas.types import LabelConfig
from src.midas.walk_forward import WalkForwardConfig, run_midas_walk_forward


async def main(args: argparse.Namespace) -> None:
    """Run the walk-forward."""
    config = load_config()
    setup_logging(config.logging.level, json_format=False)

    db = Database(config.database)
    await db.connect()

    try:
        # Check tick availability
        row = await db.fetchrow(
            "SELECT MIN(time), MAX(time), COUNT(*) FROM ticks "
            "WHERE instrument = $1 AND time >= $2 AND time < $3",
            args.instrument, args.start, args.end,
        )
        if row is None or row[2] == 0:
            print(f"No ticks for {args.instrument} in range.")
            return

        print(f"Ticks: {row[2]:,} ({row[0].date()} → {row[1].date()})")

        wf_config = WalkForwardConfig(
            instrument=args.instrument,
            train_days=args.train_days,
            test_days=args.test_days,
            step_days=args.step_days,
            label_config=LabelConfig(
                sl_points=args.sl,
                tp_points=args.tp,
                timeout_seconds=args.timeout,
                k_sl=args.k_sl,
                k_tp=args.k_tp,
            ),
            trainer_config=TrainerConfig(
                n_estimators=args.trees,
                entry_threshold=args.threshold,
            ),
            sim_config=SimConfig(
                sl_points=args.sl,
                tp_points=args.tp,
                k_sl=args.k_sl,
                k_tp=args.k_tp,
                initial_capital=args.capital,
                max_spread=args.max_spread,
                slippage_min_pts=args.slippage_min,
                slippage_max_pts=args.slippage_max,
                slippage_seed=args.slippage_seed,
            ),
            sample_on_candle=not args.sample_on_tick,
            sample_rate=args.sample_rate,
        )

        await run_midas_walk_forward(
            wf_config,
            data_start=args.start,
            data_end=args.end,
            db=db,
        )

    finally:
        await db.disconnect()


def cli() -> None:
    """CLI entry point."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Midas walk-forward: LightGBM scalping on tick data",
    )
    parser.add_argument("--instrument", default="XAUUSD")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--train-days", type=int, default=30)
    parser.add_argument("--test-days", type=int, default=2)
    parser.add_argument("--step-days", type=int, default=2)
    parser.add_argument("--sl", type=float, default=3.0,
                        help="SL in points (fixed mode / ATR fallback)")
    parser.add_argument("--tp", type=float, default=3.0,
                        help="TP in points (fixed mode / ATR fallback)")
    parser.add_argument("--k-sl", type=float, default=None,
                        help="ATR multiplier for SL (enables ATR mode)")
    parser.add_argument("--k-tp", type=float, default=None,
                        help="ATR multiplier for TP (enables ATR mode)")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Label timeout in seconds")
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Entry probability threshold")
    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument("--max-spread", type=float, default=2.0)
    parser.add_argument("--slippage-min", type=float, default=0.1,
                        help="Min slippage in points (floor per market order)")
    parser.add_argument("--slippage-max", type=float, default=0.5,
                        help="Max slippage in points (set both to 0 to disable)")
    parser.add_argument("--slippage-seed", type=int, default=None,
                        help="RNG seed for reproducible slippage")
    parser.add_argument("--sample-on-tick", action="store_true",
                        help="Sample every tick instead of on candle close")
    parser.add_argument("--sample-rate", type=int, default=1,
                        help="When --sample-on-tick: every N ticks")
    args = parser.parse_args()
    args.start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
    args.end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)

    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
