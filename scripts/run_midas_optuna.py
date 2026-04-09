"""Run Midas nested Optuna optimization.

Outer loop: extractor params + SL/TP (expensive, re-replays ticks).
Inner loop: LightGBM hyperparams + threshold (cheap, retrains only).

Usage:
    uv run python -m scripts.run_midas_optuna \
        --instrument XAUUSD \
        --train-start 2025-01-01 --train-end 2025-02-01 \
        --test-start 2025-02-01 --test-end 2025-02-03 \
        --outer-trials 30 --inner-trials 30
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime

from dotenv import load_dotenv

from src.common.config import load_config
from src.common.db import Database
from src.common.logging import setup_logging
from src.midas.optimizer import OptimizerConfig, run_nested_optuna


async def main(args: argparse.Namespace) -> None:
    """Run the optimization."""
    config = load_config()
    setup_logging(config.logging.level, json_format=False)

    db = Database(config.database)
    await db.connect()

    try:
        # Check tick availability
        count = await db.fetchval(
            "SELECT COUNT(*) FROM ticks "
            "WHERE instrument = $1 AND time >= $2 AND time < $3",
            args.instrument, args.train_start, args.test_end,
        )
        print(f"Ticks available: {count:,}")

        if count == 0:
            print("No ticks found. Exiting.")
            return

        opt_config = OptimizerConfig(
            instrument=args.instrument,
            train_start=args.train_start,
            train_end=args.train_end,
            test_start=args.test_start,
            test_end=args.test_end,
            outer_trials=args.outer_trials,
            inner_trials=args.inner_trials,
            sample_on_candle=not args.sample_on_tick,
            score_metric=args.score,
        )

        result = await run_nested_optuna(opt_config, db)

        # Save best params to YAML
        if args.output:
            import yaml

            all_params = {
                **result.best_outer_params,
                **result.best_inner_params,
                "_score": result.best_score,
                "_n_trades": result.best_n_trades,
                "_win_rate": result.best_win_rate,
            }
            with open(args.output, "w") as f:
                yaml.safe_dump(all_params, f, sort_keys=True)
            print(f"\nBest params saved to {args.output}")

    finally:
        await db.disconnect()


def cli() -> None:
    """CLI entry point."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Midas nested Optuna: optimize extractor + LightGBM params",
    )
    parser.add_argument("--instrument", default="XAUUSD")
    parser.add_argument("--train-start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--train-end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--test-start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--test-end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--outer-trials", type=int, default=30)
    parser.add_argument("--inner-trials", type=int, default=30)
    parser.add_argument("--sample-on-tick", action="store_true",
                        help="Sample every tick instead of on candle close")
    parser.add_argument("--score", default="pnl",
                        choices=["pnl", "win_rate", "pnl_per_trade"])
    parser.add_argument("--output", type=str, default=None,
                        help="Save best params to YAML file")
    args = parser.parse_args()

    for attr in ("train_start", "train_end", "test_start", "test_end"):
        setattr(
            args, attr,
            datetime.strptime(getattr(args, attr), "%Y-%m-%d").replace(
                tzinfo=UTC,
            ),
        )

    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
