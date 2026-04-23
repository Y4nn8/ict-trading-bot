"""Run Midas global (panel) Optuna optimisation.

One Optuna study across N disjoint windows. Each trial suggests ONE set
of hyperparameters applied to all windows (train on each, backtest on
each test slice, aggregate scores). The best trial is then validated on
held-out val slices.

Usage:
    uv run python -m scripts.run_midas_global_optuna \
        --instrument XAUUSD \
        --start 2025-04-01 --end 2026-04-01 \
        --train-days 14 --test-days 1 --val-days 1 \
        --n-windows 20 --n-trials 100 \
        --objective sharpe
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime

from dotenv import load_dotenv

from src.common.config import load_config
from src.common.db import Database
from src.common.logging import setup_logging
from src.midas.global_optimizer import (
    AGGREGATE_KEYS,
    GlobalOptunaConfig,
    generate_disjoint_windows,
    run_global_optuna,
)
from src.midas.optimizer import (
    default_output_prefix,
    load_fixed_inner_params,
    load_fixed_outer_params,
    load_outer_param_ranges,
)


async def main(args: argparse.Namespace) -> None:
    """Run the global Optuna pipeline."""
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

        windows = generate_disjoint_windows(
            data_start=args.start, data_end=args.end,
            train_days=args.train_days, test_days=args.test_days,
            val_days=args.val_days, step_days=args.step_days,
            n_windows=args.n_windows,
            business_days=args.business_days,
        )
        if not windows:
            print("No windows fit in the given range.")
            return
        print(f"Built {len(windows)} disjoint windows "
              f"(train={args.train_days}d, test={args.test_days}d, val={args.val_days}d, "
              f"step={args.step_days or args.train_days + args.test_days + args.val_days}d)")

        fixed_outer = None
        if args.fix_outer_params:
            fixed_outer = load_fixed_outer_params(args.fix_outer_params)
            print(f"Fixed outer params: {len(fixed_outer)} keys")
        outer_ranges = None
        if args.outer_ranges_from:
            outer_ranges = load_outer_param_ranges(args.outer_ranges_from)
            print(f"Outer ranges: {outer_ranges}")
        fixed_inner = None
        if args.fix_inner_params:
            fixed_inner = load_fixed_inner_params(args.fix_inner_params)
            print(f"Fixed inner params: {fixed_inner}")

        wf_config = GlobalOptunaConfig(
            instrument=args.instrument,
            windows=windows,
            n_trials=args.n_trials,
            objective=args.objective,
            sample_on_candle=not args.sample_on_tick,
            sample_rate=args.sample_rate,
            sl_range=tuple(args.sl_range),
            tp_range=tuple(args.tp_range),
            k_sl_range=tuple(args.k_sl_range),
            k_tp_range=tuple(args.k_tp_range),
            fixed_outer_params=fixed_outer,
            outer_param_ranges=outer_ranges,
            fixed_inner_params=fixed_inner,
            slippage_min_pts=args.slippage_min,
            slippage_max_pts=args.slippage_max,
            slippage_seed=args.slippage_seed,
            importance_threshold=args.importance_threshold,
            use_meta_labeling=args.meta_labeling,
            compute_train_robust=args.compute_train_robust,
        )

        prefix = args.output or default_output_prefix()
        await run_global_optuna(wf_config, db, prefix)

    finally:
        await db.disconnect()


def cli() -> None:
    """CLI entry point."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Midas global (panel) Optuna: 1 study, N disjoint windows",
    )
    parser.add_argument("--instrument", default="XAUUSD")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--train-days", type=int, default=14)
    parser.add_argument("--test-days", type=int, default=1)
    parser.add_argument("--val-days", type=int, default=1)
    parser.add_argument("--step-days", type=int, default=None,
                        help="Default: train+test+val (fully disjoint)")
    parser.add_argument("--n-windows", type=int, default=None,
                        help="Max number of windows (default: as many as fit)")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument(
        "--objective", default="sharpe", choices=list(AGGREGATE_KEYS),
        help="Aggregate metric to maximise. Others are logged.",
    )
    parser.add_argument("--sample-on-tick", action="store_true")
    parser.add_argument(
        "--sample-rate", type=int, default=1,
        help="Output 1 feature row every N candles (e.g. 6 → M1 on 10s base). "
             "Does NOT affect replay speed (still processes every tick), but "
             "shrinks the training dataset so LightGBM fits faster.",
    )
    parser.add_argument(
        "--compute-train-robust", action="store_true",
        help="During the validation pass (best trial only), also backtest "
             "each window on its train slice to compute robust scores. "
             "Adds ~10-15 min per window.",
    )
    parser.add_argument(
        "--business-days",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Count train/test/val/step in business days (skip weekends). "
             "Default True ensures test/val days never land on Sat/Sun. "
             "Use --no-business-days for raw calendar days.",
    )
    parser.add_argument("--sl-range", type=float, nargs=2, default=[1.5, 8.0])
    parser.add_argument("--tp-range", type=float, nargs=2, default=[1.5, 8.0])
    parser.add_argument("--k-sl-range", type=float, nargs=2, default=[0.5, 3.0])
    parser.add_argument("--k-tp-range", type=float, nargs=2, default=[0.5, 3.0])
    parser.add_argument("--slippage-min", type=float, default=0.1)
    parser.add_argument("--slippage-max", type=float, default=0.5)
    parser.add_argument("--slippage-seed", type=int, default=None)
    parser.add_argument("--importance-threshold", type=float, default=0.0)
    parser.add_argument(
        "--meta-labeling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Lopez de Prado meta-labelling gate",
    )
    parser.add_argument("--fix-outer-params", type=str, default=None)
    parser.add_argument("--outer-ranges-from", type=str, default=None)
    parser.add_argument("--fix-inner-params", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    args.start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
    args.end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)

    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
