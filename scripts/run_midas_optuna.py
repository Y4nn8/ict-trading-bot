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
from src.midas.optimizer import (
    OptimizerConfig,
    default_output_prefix,
    load_fixed_outer_params,
    run_nested_optuna,
    write_trial_logs,
)


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

        # Load fixed outer params if provided
        fixed_outer = None
        if args.fix_outer_params:
            fixed_outer = load_fixed_outer_params(args.fix_outer_params)
            print(f"Fixed outer params from {args.fix_outer_params}: "
                  f"{len(fixed_outer)} params")

        opt_config = OptimizerConfig(
            instrument=args.instrument,
            train_start=args.train_start,
            train_end=args.train_end,
            test_start=args.test_start,
            test_end=args.test_end,
            outer_trials=args.outer_trials,
            inner_trials=args.inner_trials,
            sample_on_candle=not args.sample_on_tick,
            sl_range=tuple(args.sl_range),
            tp_range=tuple(args.tp_range),
            k_sl_range=tuple(args.k_sl_range),
            k_tp_range=tuple(args.k_tp_range),
            score_metric=args.score,
            fixed_outer_params=fixed_outer,
            slippage_min_pts=args.slippage_min,
            slippage_max_pts=args.slippage_max,
            slippage_seed=args.slippage_seed,
        )

        result = await run_nested_optuna(opt_config, db)

        # Save best params to YAML + model to .bin
        prefix = args.output or default_output_prefix()
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

            if result.best_trainer is not None:
                from pathlib import Path

                model_path = Path(args.output).with_suffix(".bin")
                result.best_trainer.save(model_path)
                print(f"Best model saved to {model_path}")

        # Always write trial + trade logs with timestamped prefix
        if result.trial_records:
            write_trial_logs(result.trial_records, prefix)

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
    parser.add_argument("--score", default="composite",
                        choices=["composite", "pnl", "win_rate", "pnl_per_trade"])
    parser.add_argument("--sl-range", type=float, nargs=2, default=[1.5, 8.0],
                        metavar=("MIN", "MAX"), help="SL fallback search range (pts)")
    parser.add_argument("--tp-range", type=float, nargs=2, default=[1.5, 8.0],
                        metavar=("MIN", "MAX"), help="TP fallback search range (pts)")
    parser.add_argument("--k-sl-range", type=float, nargs=2, default=[0.5, 3.0],
                        metavar=("MIN", "MAX"), help="k_sl ATR multiplier range")
    parser.add_argument("--k-tp-range", type=float, nargs=2, default=[0.5, 3.0],
                        metavar=("MIN", "MAX"), help="k_tp ATR multiplier range")
    parser.add_argument("--slippage-min", type=float, default=0.1,
                        help="Min slippage in points (floor per market order)")
    parser.add_argument("--slippage-max", type=float, default=0.5,
                        help="Max slippage in points (set both to 0 to disable)")
    parser.add_argument("--slippage-seed", type=int, default=None,
                        help="RNG seed for reproducible slippage")
    parser.add_argument("--fix-outer-params", type=str, default=None,
                        help="YAML file with fixed outer params (skip outer search)")
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
