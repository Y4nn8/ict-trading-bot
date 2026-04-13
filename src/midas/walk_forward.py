"""Midas walk-forward orchestrator.

Slices time windows, runs labeling replays, trains LightGBM,
runs simulation replays, and reports metrics per window.

Also provides ``run_midas_wf_optuna`` which runs nested Optuna per window
and produces a param stability analysis across windows.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from src.common.models import Direction, Trade
from src.midas.labeler import build_exit_dataset, relabel_dataframe
from src.midas.optimizer import (
    OptimizationResult,
    OptimizerConfig,
    TrialRecord,
    run_nested_optuna,
    write_trial_logs,
)
from src.midas.replay_engine import (
    ReplayConfig,
    ReplayEngine,
    build_default_registry,
)
from src.midas.trade_simulator import MidasTrade, SimConfig, TradeSimulator
from src.midas.trainer import MidasTrainer, TrainerConfig
from src.midas.types import LabelConfig, Tick


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    """Walk-forward configuration.

    Args:
        instrument: Instrument to trade.
        train_days: Training window in days.
        test_days: Test window in days.
        step_days: Step between windows in days.
        label_config: Labeling SL/TP configuration.
        trainer_config: LightGBM configuration.
        sim_config: Trade simulation configuration.
        sample_on_candle: Extract features on candle close (default, recommended).
        sample_rate: Legacy tick-based sampling (only when sample_on_candle=False).
    """

    instrument: str = "XAUUSD"
    train_days: int = 30
    test_days: int = 2
    step_days: int = 2
    label_config: LabelConfig = field(default_factory=LabelConfig)
    trainer_config: TrainerConfig = field(default_factory=TrainerConfig)
    sim_config: SimConfig = field(default_factory=SimConfig)
    sample_on_candle: bool = True
    sample_rate: int = 1


@dataclass
class WindowResult:
    """Result of a single walk-forward window."""

    window_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    n_train_rows: int = 0
    n_test_ticks: int = 0
    n_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_pnl_per_trade: float = 0.0
    max_drawdown: float = 0.0
    val_log_loss: float = 0.0
    class_dist: dict[int, int] = field(default_factory=dict)
    top_features: list[tuple[str, float]] = field(default_factory=list)


def _midas_to_common_trade(
    mt: MidasTrade,
    instrument: str,
) -> Trade:
    """Convert MidasTrade to common Trade for metrics computation."""
    sl_dist = abs(mt.entry_price - mt.sl_price)
    r_multiple = (
        mt.pnl_points / sl_dist if sl_dist > 0 else 0.0
    )
    return Trade(
        opened_at=mt.entry_time,
        closed_at=mt.exit_time,
        instrument=instrument,
        direction=Direction.LONG if mt.direction == "BUY" else Direction.SHORT,
        entry_price=mt.entry_price,
        exit_price=mt.exit_price,
        stop_loss=mt.sl_price,
        take_profit=mt.tp_price,
        size=mt.size,
        pnl=mt.pnl,
        r_multiple=r_multiple,
        is_backtest=True,
    )


def _snap_to_monday(dt: datetime) -> datetime:
    """Snap a datetime forward to the next Monday (or same day if Monday)."""
    days_ahead = (7 - dt.weekday()) % 7  # 0 if already Monday
    return dt + timedelta(days=days_ahead)


def generate_windows(
    data_start: datetime,
    data_end: datetime,
    train_days: int,
    test_days: int,
    step_days: int,
    *,
    align_monday: bool = False,
    validation_days: int = 0,
) -> list[tuple[datetime, datetime, datetime, datetime, datetime | None, datetime | None]]:
    """Generate walk-forward windows.

    Returns:
        List of (train_start, train_end, test_start, test_end,
        val_start, val_end) tuples. val_start/val_end are None
        when ``validation_days`` is 0.

    Args:
        data_start: Start of available data.
        data_end: End of available data.
        train_days: Training window in days.
        test_days: Test (selection) window in days.
        step_days: Step between windows in days.
        align_monday: Snap the first window to start on Monday.
        validation_days: Validation window in days (0 = no validation).
    """
    windows: list[
        tuple[datetime, datetime, datetime, datetime, datetime | None, datetime | None]
    ] = []
    train_delta = timedelta(days=train_days)
    test_delta = timedelta(days=test_days)
    val_delta = timedelta(days=validation_days)
    step_delta = timedelta(days=step_days)

    current = _snap_to_monday(data_start) if align_monday else data_start
    total_delta = train_delta + test_delta + val_delta
    while current + total_delta <= data_end:
        train_end = current + train_delta
        test_end = train_end + test_delta
        if validation_days > 0:
            val_start = test_end
            val_end = test_end + val_delta
        else:
            val_start = None
            val_end = None
        windows.append((current, train_end, train_end, test_end, val_start, val_end))
        current += step_delta

    return windows


async def run_midas_walk_forward(
    config: WalkForwardConfig,
    data_start: datetime,
    data_end: datetime,
    db: Any,
) -> list[WindowResult]:
    """Run the full Midas walk-forward pipeline.

    Args:
        config: Walk-forward configuration.
        data_start: Start of available tick data.
        data_end: End of available tick data.
        db: Database connection.

    Returns:
        List of per-window results.
    """
    import polars as pl

    windows = generate_windows(
        data_start, data_end,
        config.train_days, config.test_days, config.step_days,
    )

    if not windows:
        print("No windows generated. Check date range and window sizes.")
        return []

    lc = config.label_config
    print(f"\nMidas Walk-Forward: {len(windows)} windows")
    print(f"  Train: {config.train_days}d, Test: {config.test_days}d, "
          f"Step: {config.step_days}d")
    if lc.k_sl is not None and lc.k_tp is not None:
        print(f"  SL: k_sl={lc.k_sl} * ATR, TP: k_tp={lc.k_tp} * ATR "
              f"(fallback {lc.sl_points}/{lc.tp_points}pts)")
    else:
        print(f"  SL: {lc.sl_points}pts, TP: {lc.tp_points}pts")
    print(f"  Timeout: {lc.timeout_seconds}s, "
          f"Entry threshold: {config.trainer_config.entry_threshold}")

    results: list[WindowResult] = []

    for i, (train_start, train_end, test_start, test_end, _, _) in enumerate(windows):
        print(f"\n{'='*60}")
        print(f"Window {i+1}/{len(windows)}: "
              f"train {train_start.date()}→{train_end.date()}, "
              f"test {test_start.date()}→{test_end.date()}")
        print(f"{'='*60}")

        wr = WindowResult(
            window_index=i,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )

        # --- Phase 1: Training replay (features) + relabel + train ---
        print("  [1/2] Training replay + labeling + LightGBM...")

        registry = build_default_registry(instrument=config.instrument)
        registry.configure_all({})

        with tempfile.TemporaryDirectory() as tmpdir:
            train_parquet = Path(tmpdir) / "train.parquet"

            train_engine = ReplayEngine(
                db, registry,
                ReplayConfig(
                    instrument=config.instrument,
                    start=train_start,
                    end=train_end,
                    sample_on_candle=config.sample_on_candle,
                    sample_rate=config.sample_rate,
                    output_path=train_parquet,
                ),
            )
            train_result = await train_engine.run()

            if not train_parquet.exists() or train_result.feature_rows < 100:
                print(f"  SKIP: insufficient features "
                      f"({train_result.feature_rows})")
                results.append(wr)
                continue

            df = pl.read_parquet(train_parquet)

        # Label within the DataFrame only (no look-ahead beyond train_end)
        label_result = relabel_dataframe(
            df,
            sl_points=config.label_config.sl_points,
            tp_points=config.label_config.tp_points,
            timeout_seconds=config.label_config.timeout_seconds,
            k_sl=config.label_config.k_sl,
            k_tp=config.label_config.k_tp,
            atr_column=config.label_config.atr_column,
        )

        print(f"    Ticks: {train_result.total_ticks:,}, "
              f"Features: {train_result.feature_rows:,}, "
              f"Labels: {label_result.total_labeled:,}")
        print(f"    BUY wins: {label_result.buy_wins}, "
              f"losses: {label_result.buy_losses}, "
              f"SELL wins: {label_result.sell_wins}, "
              f"losses: {label_result.sell_losses}, "
              f"timeouts: {label_result.timeouts}")

        target = MidasTrainer.build_target(
            label_result.buy_labels,
            label_result.sell_labels,
        )
        buy_arr = np.asarray(label_result.buy_labels)
        sell_arr = np.asarray(label_result.sell_labels)
        mask = ~((buy_arr == -1) & (sell_arr == -1))

        if mask.sum() < 100:
            print(f"  SKIP: only {mask.sum()} valid rows after filtering")
            results.append(wr)
            continue

        df_filtered = df.filter(pl.Series(mask))
        target_filtered = target[mask]

        wr.n_train_rows = len(df_filtered)

        # Build PnL-based sample weights
        buy_pnls = [label_result.buy_pnls[i] for i, m in enumerate(mask) if m]
        sell_pnls = [label_result.sell_pnls[i] for i, m in enumerate(mask) if m]
        weights = MidasTrainer.build_sample_weights(
            buy_pnls, sell_pnls, target_filtered,
        )

        trainer = MidasTrainer(config.trainer_config)
        train_metrics = trainer.train(
            df_filtered, target_filtered, sample_weights=weights,
        )

        wr.val_log_loss = train_metrics.val_log_loss
        wr.class_dist = train_metrics.class_distribution

        # Top 5 features
        sorted_imp = sorted(
            train_metrics.feature_importance.items(),
            key=lambda x: x[1], reverse=True,
        )[:5]
        wr.top_features = sorted_imp

        print(f"    Train rows: {wr.n_train_rows:,}, "
              f"Val loss: {wr.val_log_loss:.4f}")
        print(f"    Class dist: {wr.class_dist}")
        print(f"    Top features: {[f[0] for f in sorted_imp]}")

        # Train exit model with optimal-close labeling
        exit_ds = build_exit_dataset(
            df_filtered,
            target_filtered,
            sl_points=config.label_config.sl_points,
            tp_points=config.label_config.tp_points,
            timeout_seconds=config.label_config.timeout_seconds,
            k_sl=config.label_config.k_sl,
            k_tp=config.label_config.k_tp,
            atr_column=config.label_config.atr_column,
        )

        if exit_ds.n_rows >= 50:
            exit_df = df_filtered[exit_ds.row_indices].with_columns(
                pl.Series("pos_unrealized_pnl", exit_ds.unrealized_pnls),
                pl.Series("pos_duration_sec", exit_ds.durations),
                pl.Series("pos_direction", exit_ds.directions),
            )

            exit_result = trainer.train_exit(exit_df, exit_ds.exit_labels)
            n_close = int((exit_ds.exit_labels == 1).sum())
            n_hold = int((exit_ds.exit_labels == 0).sum())
            print(f"    Exit model: {exit_result.n_train} rows "
                  f"(HOLD={n_hold}, CLOSE={n_close}), "
                  f"val_loss={exit_result.val_log_loss:.4f}")
        else:
            print(f"    Exit model: SKIP ({exit_ds.n_rows} rows < 50)")

        # --- Phase 2: Test replay with live prediction ---
        print("  [2/2] Test replay + simulation...")

        simulator = TradeSimulator(config.sim_config)
        # Cache latest features for exit model (updated on sampled ticks)
        latest_features: dict[str, float] = {}
        _atr_col = config.label_config.atr_column

        def test_callback(
            tick: Tick,
            features: dict[str, float],
            *,
            _tr: MidasTrainer = trainer,
            _sim: TradeSimulator = simulator,
            _feat: dict[str, float] = latest_features,
            atr_col: str = _atr_col,
        ) -> None:
            _feat.update(features)
            # Exit model at sample time (candle close when sample_on_candle=True)
            if _tr.has_exit_model and _sim.open_count > 0:
                ctx = _sim.get_position_context(tick)
                if ctx is not None:
                    should_close, _ = _tr.predict_exit(
                        _feat,
                        pos_unrealized_pnl=ctx["pos_unrealized_pnl"],
                        pos_duration_sec=ctx["pos_duration_sec"],
                        pos_direction=ctx["pos_direction"],
                    )
                    if should_close:
                        _sim.early_close(tick)
            signal, confidence = _tr.predict(features)
            _sim.on_signal(
                tick, signal,
                atr=features.get(atr_col, 0.0),
                proba=confidence,
            )

        last_tick_holder: list[Tick | None] = [None]

        def exit_hook(
            tick: Tick,
            *,
            _sim: TradeSimulator = simulator,
            _holder: list[Tick | None] = last_tick_holder,
        ) -> None:
            # SL/TP mechanical exits (must stay per-tick)
            _sim.on_tick(tick)
            _holder[0] = tick

        test_registry = build_default_registry(instrument=config.instrument)
        test_registry.configure_all({})

        test_engine = ReplayEngine(
            db, test_registry,
            ReplayConfig(
                instrument=config.instrument,
                start=test_start,
                end=test_end,
                sample_on_candle=config.sample_on_candle,
                sample_rate=config.sample_rate,
            ),
            tick_callback=test_callback,
            every_tick_hook=exit_hook,
        )
        test_result = await test_engine.run()
        wr.n_test_ticks = test_result.total_ticks

        # Force-close remaining positions at last observed market price
        if simulator.open_count > 0 and last_tick_holder[0] is not None:
            simulator.close_all(last_tick_holder[0])

        trades = simulator.closed_trades
        wr.n_trades = len(trades)

        if trades:
            wr.total_pnl = sum(t.pnl for t in trades)
            wins = sum(1 for t in trades if t.is_win)
            wr.win_rate = wins / len(trades)
            wr.avg_pnl_per_trade = wr.total_pnl / len(trades)

            # Drawdown
            equity = [config.sim_config.initial_capital]
            for t in trades:
                equity.append(equity[-1] + t.pnl)
            peak = equity[0]
            max_dd = 0.0
            for e in equity:
                peak = max(peak, e)
                dd = (peak - e) / peak if peak > 0 else 0.0
                max_dd = max(max_dd, dd)
            wr.max_drawdown = max_dd

        results.append(wr)

        print(f"    Trades: {wr.n_trades}, PnL: {wr.total_pnl:+.2f}, "
              f"WR: {wr.win_rate*100:.1f}%, "
              f"Avg: {wr.avg_pnl_per_trade:+.2f}, "
              f"MDD: {wr.max_drawdown*100:.1f}%")

    # --- Summary ---
    _print_summary(results, config)
    return results


def _print_summary(
    results: list[WindowResult],
    config: WalkForwardConfig,
) -> None:
    """Print aggregated walk-forward summary."""
    active = [r for r in results if r.n_trades > 0]
    if not active:
        print("\nNo trades produced in any window.")
        return

    print(f"\n{'='*60}")
    print("WALK-FORWARD SUMMARY")
    print(f"{'='*60}")

    total_trades = sum(r.n_trades for r in active)
    total_pnl = sum(r.total_pnl for r in active)
    avg_wr = sum(r.win_rate for r in active) / len(active)
    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0.0
    worst_mdd = max(r.max_drawdown for r in active)
    profitable = sum(1 for r in active if r.total_pnl > 0)

    print(f"  Windows: {len(active)}/{len(results)} active")
    print(f"  Total trades: {total_trades}")
    print(f"  Total PnL: {total_pnl:+.2f}")
    print(f"  Avg PnL/trade: {avg_pnl:+.2f}")
    print(f"  Avg win rate: {avg_wr*100:.1f}%")
    print(f"  Worst MDD: {worst_mdd*100:.1f}%")
    print(f"  Profitable windows: {profitable}/{len(active)}")


# ---------------------------------------------------------------------------
# Walk-Forward + Optuna (multi-fenêtre)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WalkForwardOptunaConfig:
    """Walk-forward configuration with per-window Optuna optimization.

    Args:
        instrument: Instrument to trade.
        train_days: Training window in days.
        test_days: Test window in days.
        step_days: Step between windows in days.
        outer_trials: Number of outer Optuna trials per window.
        inner_trials: Number of inner Optuna trials per window.
        sample_on_candle: Extract features on candle close.
        score_metric: Metric to optimize (composite uses PnL + trade deficit).
        min_daily_trades: Minimum expected trades per trading day.
        trade_deficit_penalty: Penalty per missing trade below minimum.
        validation_days: Validation window in days (0 = no validation).
        fixed_inner_params: Fixed inner params (skip suggestion for these).
        align_monday: Snap window boundaries to Monday.
        sl_range: SL search range in points.
        tp_range: TP search range in points.
        k_sl_range: k_sl multiplier search range.
        k_tp_range: k_tp multiplier search range.
        gamma_range: Gamma ramp search range.
        max_margin_proba_range: Max margin proba search range.
        fixed_outer_params: Fixed extractor params (skip outer search).
        slippage_min_pts: Min adverse slippage per market order.
        slippage_max_pts: Max adverse slippage per market order.
        slippage_seed: RNG seed for reproducible slippage.
    """

    instrument: str = "XAUUSD"
    train_days: int = 30
    test_days: int = 7
    step_days: int = 7
    outer_trials: int = 10
    inner_trials: int = 20
    sample_on_candle: bool = True
    score_metric: str = "composite"
    min_daily_trades: int = 10
    trade_deficit_penalty: float = 10.0
    validation_days: int = 0
    fixed_inner_params: dict[str, Any] | None = None
    align_monday: bool = False
    sl_range: tuple[float, float] = (1.5, 8.0)
    tp_range: tuple[float, float] = (1.5, 8.0)
    k_sl_range: tuple[float, float] = (0.5, 3.0)
    k_tp_range: tuple[float, float] = (0.5, 3.0)
    gamma_range: tuple[float, float] = (0.5, 3.0)
    max_margin_proba_range: tuple[float, float] = (0.70, 0.95)
    fixed_outer_params: dict[str, Any] | None = None
    outer_param_ranges: dict[str, tuple[float, float]] | None = None
    slippage_min_pts: float = 0.0
    slippage_max_pts: float = 0.0
    slippage_seed: int | None = None


async def run_midas_wf_optuna(
    config: WalkForwardOptunaConfig,
    data_start: datetime,
    data_end: datetime,
    db: Any,
    output_prefix: str,
) -> list[OptimizationResult]:
    """Run walk-forward with nested Optuna per window.

    For each window: runs ``run_nested_optuna`` to find optimal params
    on train data, evaluates on OOS. Writes trial + trade logs and
    prints param stability analysis across windows.

    Args:
        config: Walk-forward Optuna configuration.
        data_start: Start of available tick data.
        data_end: End of available tick data.
        db: Database connection.
        output_prefix: Timestamped prefix for output files.

    Returns:
        List of OptimizationResult, one per window.
    """
    windows = generate_windows(
        data_start, data_end,
        config.train_days, config.test_days, config.step_days,
        align_monday=config.align_monday,
        validation_days=config.validation_days,
    )

    if not windows:
        print("No windows generated. Check date range and window sizes.")
        return []

    print(f"\nWalk-Forward Optuna: {len(windows)} windows")
    desc = f"  Train: {config.train_days}d, Selection: {config.test_days}d"
    if config.validation_days > 0:
        desc += f", Validation: {config.validation_days}d"
    desc += f", Step: {config.step_days}d"
    print(desc)
    print(f"  Per window: {config.outer_trials} outer x "
          f"{config.inner_trials} inner trials")

    results: list[OptimizationResult] = []
    all_records: list[TrialRecord] = []

    for i, (train_start, train_end, test_start, test_end, val_start, val_end) in enumerate(windows):
        print(f"\n{'#'*60}")
        window_desc = (f"WINDOW {i+1}/{len(windows)}: "
                       f"train {train_start.date()}→{train_end.date()}, "
                       f"sel {test_start.date()}→{test_end.date()}")
        if val_start is not None:
            window_desc += f", val {val_start.date()}→{val_end.date()}"  # type: ignore[union-attr]
        print(window_desc)
        print(f"{'#'*60}")

        opt_config = OptimizerConfig(
            instrument=config.instrument,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            outer_trials=config.outer_trials,
            inner_trials=config.inner_trials,
            sample_on_candle=config.sample_on_candle,
            score_metric=config.score_metric,
            min_daily_trades=config.min_daily_trades,
            trade_deficit_penalty=config.trade_deficit_penalty,
            validation_start=val_start,
            validation_end=val_end,
            fixed_inner_params=config.fixed_inner_params,
            sl_range=config.sl_range,
            tp_range=config.tp_range,
            k_sl_range=config.k_sl_range,
            k_tp_range=config.k_tp_range,
            gamma_range=config.gamma_range,
            max_margin_proba_range=config.max_margin_proba_range,
            fixed_outer_params=config.fixed_outer_params,
            outer_param_ranges=config.outer_param_ranges,
            slippage_min_pts=config.slippage_min_pts,
            slippage_max_pts=config.slippage_max_pts,
            slippage_seed=config.slippage_seed,
        )

        result = await run_nested_optuna(opt_config, db, window_idx=i)
        results.append(result)
        all_records.extend(result.trial_records)

        # Write trial + trade logs incrementally after each window
        if all_records:
            write_trial_logs(all_records, output_prefix)

    # Print per-window OOS summary + param stability
    _print_wf_optuna_summary(results, windows)

    if len(results) >= 2:
        _print_param_stability(results)

    return results


def _print_wf_optuna_summary(
    results: list[OptimizationResult],
    windows: list[tuple[datetime, datetime, datetime, datetime, datetime | None, datetime | None]],
) -> None:
    """Print per-window OOS summary table."""
    has_val = any(r.val_score is not None for r in results)

    print(f"\n{'='*70}")
    print("WALK-FORWARD OPTUNA SUMMARY")
    print(f"{'='*70}")
    header = (f"{'Window':<8} {'Test dates':<25} {'Score':>8} "
              f"{'Trades':>7} {'WR':>6} {'PnL':>10}")
    if has_val:
        header += f" {'ValPnL':>10}"
    print(header)
    print("-" * (80 if has_val else 70))

    total_trades = 0
    total_pnl = 0.0
    total_val_pnl = 0.0
    profitable = 0

    for i, (result, win) in enumerate(zip(results, windows, strict=True)):
        test_str = f"{win[2].date()} → {win[3].date()}"
        line = (f"W{i+1:<6} {test_str:<25} {result.best_score:>+8.1f} "
                f"{result.best_n_trades:>7} "
                f"{result.best_win_rate*100:>5.1f}% "
                f"{result.best_pnl:>+10.2f}")
        if has_val:
            val_pnl = result.val_pnl if result.val_score is not None else 0.0
            line += f" {val_pnl:>+10.2f}"
            total_val_pnl += val_pnl
        print(line)
        total_trades += result.best_n_trades
        total_pnl += result.best_pnl
        if result.best_pnl > 0:
            profitable += 1

    print("-" * (80 if has_val else 70))
    totals = (f"{'Total':<34} {'':<8} {total_trades:>7} "
              f"{'':<6} {total_pnl:>+10.2f}")
    if has_val:
        totals += f" {total_val_pnl:>+10.2f}"
    print(totals)
    print(f"  Profitable windows: {profitable}/{len(results)}")


def _print_param_stability(results: list[OptimizationResult]) -> None:
    """Print param stability analysis across windows.

    For each inner param, shows mean, std, and coefficient of variation (CV).
    Low CV = stable param, high CV = unstable / window-dependent.
    """
    # Collect best inner params per window
    all_inner: list[dict[str, Any]] = [
        r.best_inner_params for r in results if r.best_inner_params
    ]
    if len(all_inner) < 2:
        return

    # Also collect outer params
    all_outer: list[dict[str, Any]] = [
        r.best_outer_params for r in results if r.best_outer_params
    ]

    print(f"\n{'='*70}")
    print("PARAM STABILITY ACROSS WINDOWS")
    print(f"{'='*70}")
    print(f"{'Param':<30} {'Mean':>10} {'Std':>10} {'CV%':>8} {'Stable?':<8}")
    print("-" * 70)

    converged = 0
    total = 0

    for label, param_dicts in [("inner", all_inner), ("outer", all_outer)]:
        if not param_dicts:
            continue

        all_keys = sorted(
            {k for d in param_dicts for k in d if isinstance(d[k], (int, float))},
        )

        for key in all_keys:
            values = [
                float(d[key]) for d in param_dicts
                if key in d and isinstance(d[key], (int, float))
            ]
            if len(values) < 2:
                continue

            total += 1
            mean = np.mean(values)
            std = np.std(values)
            if abs(mean) > 1e-9:
                cv = float(std / abs(mean) * 100)
                stable = cv < 15.0
                cv_display = f"{cv:>7.1f}%"
            else:
                stable = False
                cv_display = f"{'N/A':>8}"
            if stable:
                converged += 1
            marker = "YES" if stable else ""
            name = f"{label}__{key}"
            print(f"{name:<30} {mean:>10.4f} {std:>10.4f} "
                  f"{cv_display} {marker:<8}")

    if total > 0:
        print(f"\nConverged: {converged}/{total} params "
              f"({converged/total*100:.0f}%) with CV < 15%")
