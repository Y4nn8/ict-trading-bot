"""Run Midas walk-forward with per-window Optuna optimization.

For each window: nested Optuna finds optimal params on train,
evaluates on OOS. Writes trial/trade CSVs and param stability.

Usage:
    uv run python -m scripts.run_midas_wf_optuna \
        --instrument XAUUSD \
        --start 2025-01-01 --end 2025-04-01 \
        --train-days 30 --test-days 7 --step-days 7 \
        --outer-trials 10 --inner-trials 20
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
    default_output_prefix,
    load_fixed_inner_params,
    load_fixed_outer_params,
    load_outer_param_ranges,
)
from src.midas.walk_forward import (
    WalkForwardOptunaConfig,
    run_midas_wf_optuna,
)


async def main(args: argparse.Namespace) -> None:
    """Run the walk-forward Optuna pipeline."""
    config = load_config()
    setup_logging(config.logging.level, json_format=False)

    db = Database(config.database)
    await db.connect()

    try:
        row = await db.fetchrow(
            "SELECT MIN(time), MAX(time), COUNT(*) FROM ticks "
            "WHERE instrument = $1 AND time >= $2 AND time < $3",
            args.instrument, args.start, args.end,
        )
        if row is None or row[2] == 0:
            print(f"No ticks for {args.instrument} in range.")
            return

        print(f"Ticks: {row[2]:,} ({row[0].date()} → {row[1].date()})")

        # Load fixed outer params if provided
        fixed_outer = None
        if args.fix_outer_params:
            fixed_outer = load_fixed_outer_params(args.fix_outer_params)
            print(f"Fixed outer params from {args.fix_outer_params}: "
                  f"{len(fixed_outer)} params")

        # Load outer param range overrides if provided
        outer_ranges = None
        if args.outer_ranges_from:
            outer_ranges = load_outer_param_ranges(args.outer_ranges_from)
            print(f"Outer ranges from {args.outer_ranges_from}: "
                  f"{outer_ranges}")

        # Load fixed inner params if provided
        fixed_inner = None
        if args.fix_inner_params:
            fixed_inner = load_fixed_inner_params(args.fix_inner_params)
            print(f"Fixed inner params from {args.fix_inner_params}: "
                  f"{fixed_inner}")

        wf_config = WalkForwardOptunaConfig(
            instrument=args.instrument,
            train_days=args.train_days,
            test_days=args.test_days,
            step_days=args.step_days,
            outer_trials=args.outer_trials,
            inner_trials=args.inner_trials,
            sample_on_candle=not args.sample_on_tick,
            score_metric=args.score,
            min_daily_trades=args.min_daily_trades,
            trade_deficit_penalty=args.trade_deficit_penalty,
            validation_days=args.validation_days,
            fixed_inner_params=fixed_inner,
            align_monday=args.align_monday,
            sl_range=tuple(args.sl_range),
            tp_range=tuple(args.tp_range),
            k_sl_range=tuple(args.k_sl_range),
            k_tp_range=tuple(args.k_tp_range),
            fixed_outer_params=fixed_outer,
            outer_param_ranges=outer_ranges,
            slippage_min_pts=args.slippage_min,
            slippage_max_pts=args.slippage_max,
            slippage_seed=args.slippage_seed,
        )

        prefix = args.output or default_output_prefix()
        await run_midas_wf_optuna(
            wf_config,
            data_start=args.start,
            data_end=args.end,
            db=db,
            output_prefix=prefix,
        )

    finally:
        await db.disconnect()


def cli() -> None:
    """CLI entry point."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Midas walk-forward + Optuna: optimize per window",
    )
    parser.add_argument("--instrument", default="XAUUSD")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--train-days", type=int, default=30)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--outer-trials", type=int, default=10)
    parser.add_argument("--inner-trials", type=int, default=20)
    parser.add_argument("--sample-on-tick", action="store_true",
                        help="Sample every tick instead of on candle close")
    parser.add_argument("--score", default="composite",
                        choices=["composite", "pnl", "win_rate", "pnl_per_trade"])
    parser.add_argument("--min-daily-trades", type=int, default=10,
                        help="Min trades/day for trade deficit penalty (default: 10)")
    parser.add_argument("--trade-deficit-penalty", type=float, default=10.0,
                        help="Penalty per missing trade below minimum (default: 10.0)")
    parser.add_argument("--validation-days", type=int, default=0,
                        help="Validation window in days after selection (0=disabled)")
    parser.add_argument("--align-monday", action="store_true",
                        help="Snap window boundaries to Monday")
    parser.add_argument("--fix-inner-params", type=str, default=None,
                        help="YAML file with fixed inner params to reduce search space")
    parser.add_argument("--sl-range", type=float, nargs=2, default=[1.5, 8.0],
                        metavar=("MIN", "MAX"))
    parser.add_argument("--tp-range", type=float, nargs=2, default=[1.5, 8.0],
                        metavar=("MIN", "MAX"))
    parser.add_argument("--k-sl-range", type=float, nargs=2, default=[0.5, 3.0],
                        metavar=("MIN", "MAX"))
    parser.add_argument("--k-tp-range", type=float, nargs=2, default=[0.5, 3.0],
                        metavar=("MIN", "MAX"))
    parser.add_argument("--slippage-min", type=float, default=0.1)
    parser.add_argument("--slippage-max", type=float, default=0.5)
    parser.add_argument("--slippage-seed", type=int, default=None)
    parser.add_argument("--fix-outer-params", type=str, default=None,
                        help="YAML file with fixed outer params")
    parser.add_argument("--outer-ranges-from", type=str, default=None,
                        help="YAML file with restricted outer ranges {name: [lo, hi]}")
    parser.add_argument("--output", type=str, default=None,
                        help="Output prefix (default: timestamped)")
    args = parser.parse_args()

    args.start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
    args.end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)

    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
