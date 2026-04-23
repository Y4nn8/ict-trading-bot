"""Global (panel) Optuna optimizer for the Midas engine.

Trains ONE LightGBM model per window, but ALL windows in a trial share
the same hyperparameters. The Optuna objective is a single scalar
aggregate (e.g. Sharpe) computed across the N per-window test PnLs,
which forces hyperparameters to generalise across regimes.

Performance optimisation: each trial replays ticks ONCE over the full
time span covering all windows, then slices the resulting feature
DataFrame per window for training. Test backtests still need a small
per-window replay because the simulator is tick-driven (SL/TP intra-bar).

Window layout (overlapping trains, disjoint tests):
    |- train 650j -|- test 1j -|- val 1j -|         ← W0
       |- train 650j -|- test 1j -|- val 1j -|      ← W1 (step=2)
          ...
"""

from __future__ import annotations

import contextlib
import csv
import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl

from src.midas.labeler import build_exit_dataset, relabel_dataframe
from src.midas.optimizer import (
    OptimizerConfig,
    _evaluate_oos_async,
    _suggest_inner_params,
    _suggest_outer_params,
    compute_robust_score,
)
from src.midas.replay_engine import (
    ReplayConfig,
    ReplayEngine,
    build_default_registry,
)
from src.midas.trade_simulator import MidasTrade, SimConfig
from src.midas.trainer import MidasTrainer, TrainerConfig
from src.midas.types import ATR_COLUMN_DEFAULT

if TYPE_CHECKING:
    from src.common.db import Database


AGGREGATE_KEYS: tuple[str, ...] = (
    "sharpe", "mean_pnl", "median_pnl", "sum_pnl",
    "pct_positive", "min_pnl", "max_pnl", "n_windows_traded",
)
"""All aggregate metrics computed per trial. The objective is one of
these names; the rest are logged for post-hoc analysis."""


@dataclass(frozen=True, slots=True)
class Window:
    """A single (train, test, val) window triple.

    Trains may overlap (the next window's train_start can be before
    this window's val_end). Tests and vals are kept disjoint to ensure
    independent OOS evaluations.
    """

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    val_start: datetime
    val_end: datetime


@dataclass
class GlobalOptunaConfig:
    """Configuration for global (panel) Optuna optimisation."""

    instrument: str = "XAUUSD"
    windows: list[Window] = field(default_factory=list)
    n_trials: int = 100
    objective: str = "sharpe"
    sample_on_candle: bool = True
    sample_rate: int = 1
    min_daily_trades: int = 0
    trade_deficit_penalty: float = 0.0
    atr_column: str = ATR_COLUMN_DEFAULT
    sl_range: tuple[float, float] = (1.5, 8.0)
    tp_range: tuple[float, float] = (1.5, 8.0)
    k_sl_range: tuple[float, float] = (0.5, 3.0)
    k_tp_range: tuple[float, float] = (0.5, 3.0)
    gamma_range: tuple[float, float] = (0.5, 3.0)
    max_margin_proba_range: tuple[float, float] = (0.70, 0.95)
    min_risk_pct_range: tuple[float, float] = (0.001, 0.02)
    fixed_outer_params: dict[str, Any] | None = None
    outer_param_ranges: dict[str, tuple[float, float]] | None = None
    fixed_inner_params: dict[str, Any] | None = None
    slippage_min_pts: float = 0.0
    slippage_max_pts: float = 0.0
    slippage_seed: int | None = None
    importance_threshold: float = 0.0
    use_meta_labeling: bool = False
    compute_train_robust: bool = False
    """If True, also backtest each window on its train slice during the
    validation pass (best trial only) to compute robust scores. This is
    expensive (one full train backtest per window) but only runs once."""


@dataclass
class TrialAggregate:
    """Per-trial aggregate across N windows."""

    trial_idx: int
    outer_params: dict[str, Any]
    inner_params: dict[str, Any]
    window_pnls: list[float]
    window_n_trades: list[int]
    aggregates: dict[str, float]


@dataclass
class GlobalOptunaResult:
    """Result of a global optimisation study."""

    best_trial_idx: int = -1
    best_outer_params: dict[str, Any] = field(default_factory=dict)
    best_inner_params: dict[str, Any] = field(default_factory=dict)
    best_aggregates: dict[str, float] = field(default_factory=dict)
    best_window_pnls: list[float] = field(default_factory=list)
    best_window_n_trades: list[int] = field(default_factory=list)
    val_window_pnls: list[float] = field(default_factory=list)
    val_window_n_trades: list[int] = field(default_factory=list)
    val_aggregates: dict[str, float] = field(default_factory=dict)
    train_window_pnls: list[float] = field(default_factory=list)
    """Per-window train backtest PnLs (best trial only, robust score input)."""
    robust_window_pnls: list[float] = field(default_factory=list)
    """Per-window robust scores (best trial only)."""
    robust_aggregates: dict[str, float] = field(default_factory=dict)
    trial_records: list[TrialAggregate] = field(default_factory=list)


def _add_days(dt: datetime, n: int, business_days: bool) -> datetime:
    """Add ``n`` days to ``dt``. In business-days mode, skip Sat/Sun."""
    if not business_days:
        return dt + timedelta(days=n)
    current = dt
    added = 0
    while added < n:
        current = current + timedelta(days=1)
        if current.weekday() < 5:  # Mon=0 .. Fri=4
            added += 1
    return current


def _next_business_day(dt: datetime) -> datetime:
    """Return the first Mon-Fri at or after ``dt``."""
    current = dt
    while current.weekday() >= 5:
        current = current + timedelta(days=1)
    return current


def generate_disjoint_windows(
    data_start: datetime,
    data_end: datetime,
    train_days: int = 14,
    test_days: int = 1,
    val_days: int = 1,
    step_days: int | None = None,
    n_windows: int | None = None,
    business_days: bool = False,
) -> list[Window]:
    """Build windows covering ``data_start..data_end``.

    With the default ``step_days = train+test+val``, windows are fully
    disjoint. With ``step_days = test+val``, trains overlap heavily but
    tests/vals stay disjoint (rolling walk-forward, no train look-ahead).

    Args:
        data_start: Earliest date available.
        data_end: Latest date available (exclusive end).
        train_days: Length of each train slice (in business days if
            ``business_days=True``, else calendar days).
        test_days: Length of each test slice.
        val_days: Length of each validation slice.
        step_days: Stride between consecutive window starts. Defaults
            to ``train_days + test_days + val_days`` (fully disjoint).
            Must be ``>= test_days + val_days`` so test/val slices stay
            disjoint across windows.
        n_windows: Max number of windows to produce. If ``None``, fit
            as many as possible in the available range.
        business_days: If True, advance all slices by Mon-Fri only,
            skipping Saturdays and Sundays. Ensures test/val days
            never land on a market-closed weekend (useful for FX/CFDs).
    """
    window_span = train_days + test_days + val_days
    if step_days is None:
        step_days = window_span
    min_step = test_days + val_days
    if step_days < min_step:
        msg = (
            f"step_days ({step_days}) must be >= test+val ({min_step}) "
            "to keep test/val slices disjoint across windows"
        )
        raise ValueError(msg)

    windows: list[Window] = []
    cursor = _next_business_day(data_start) if business_days else data_start
    while True:
        train_start = cursor
        train_end = _add_days(train_start, train_days, business_days)
        test_start = train_end
        test_end = _add_days(test_start, test_days, business_days)
        val_start = test_end
        val_end = _add_days(val_start, val_days, business_days)
        if val_end > data_end:
            break
        windows.append(Window(
            train_start=train_start, train_end=train_end,
            test_start=test_start, test_end=test_end,
            val_start=val_start, val_end=val_end,
        ))
        if n_windows is not None and len(windows) >= n_windows:
            break
        cursor = _add_days(cursor, step_days, business_days)
    return windows


def compute_aggregates(
    window_pnls: list[float],
    window_n_trades: list[int] | None = None,
) -> dict[str, float]:
    """Compute all per-trial aggregates from N per-window PnLs."""
    if not window_pnls:
        return dict.fromkeys(AGGREGATE_KEYS, 0.0)

    n = len(window_pnls)
    mean_pnl = sum(window_pnls) / n
    sum_pnl = sum(window_pnls)
    median_pnl = statistics.median(window_pnls)
    min_pnl = min(window_pnls)
    max_pnl = max(window_pnls)
    n_pos = sum(1 for p in window_pnls if p > 0)
    pct_positive = n_pos / n

    if n >= 2:
        std = statistics.pstdev(window_pnls)
        sharpe = mean_pnl / std if std > 0 else 0.0
    else:
        sharpe = 0.0

    if window_n_trades is None:
        n_windows_traded = float(sum(1 for p in window_pnls if p != 0))
    else:
        n_windows_traded = float(sum(1 for n_tr in window_n_trades if n_tr > 0))

    return {
        "sharpe": sharpe,
        "mean_pnl": mean_pnl,
        "median_pnl": median_pnl,
        "sum_pnl": sum_pnl,
        "pct_positive": pct_positive,
        "min_pnl": min_pnl,
        "max_pnl": max_pnl,
        "n_windows_traded": n_windows_traded,
    }


async def _replay_full_span(
    db: Any,
    config: GlobalOptunaConfig,
    outer_params: dict[str, Any],
    span_start: datetime,
    span_end: datetime,
    output_path: Path,
) -> int:
    """Replay ticks once over [span_start, span_end] with given extractor params.

    Returns the number of feature rows produced. The features are
    written to ``output_path`` for later slicing.
    """
    registry = build_default_registry(instrument=config.instrument)
    registry.configure_all(outer_params)
    replay = ReplayEngine(
        db, registry,
        ReplayConfig(
            instrument=config.instrument,
            start=span_start, end=span_end,
            sample_on_candle=config.sample_on_candle,
            sample_rate=config.sample_rate,
            output_path=output_path,
        ),
    )
    result = await replay.run()
    return result.feature_rows


def _train_from_slice(
    df_slice: pl.DataFrame,
    inner_params: dict[str, Any],
    config: GlobalOptunaConfig,
) -> MidasTrainer | None:
    """Relabel a feature DF slice and train entry + meta + exit models.

    Returns the trained MidasTrainer, or None on degenerate inputs.
    """
    if len(df_slice) < 100:
        return None

    label_result = relabel_dataframe(
        df_slice,
        sl_points=inner_params["sl_fallback"],
        tp_points=inner_params["tp_fallback"],
        timeout_seconds=inner_params["label_timeout"],
        k_sl=inner_params["k_sl"],
        k_tp=inner_params["k_tp"],
        atr_column=config.atr_column,
    )
    target = MidasTrainer.build_target(
        label_result.buy_labels, label_result.sell_labels,
    )
    buy_arr = np.asarray(label_result.buy_labels)
    sell_arr = np.asarray(label_result.sell_labels)
    mask = ~((buy_arr == -1) & (sell_arr == -1))
    if mask.sum() < 100:
        return None

    df_filtered = df_slice.filter(pl.Series(mask))
    target_filtered = target[mask]
    buy_pnls = [label_result.buy_pnls[i] for i, m in enumerate(mask) if m]
    sell_pnls = [label_result.sell_pnls[i] for i, m in enumerate(mask) if m]
    weights = MidasTrainer.build_sample_weights(
        buy_pnls, sell_pnls, target_filtered,
    )

    trainer_config = TrainerConfig(
        n_estimators=inner_params["n_estimators"],
        learning_rate=inner_params["learning_rate"],
        max_depth=inner_params["max_depth"],
        num_leaves=inner_params["num_leaves"],
        min_child_samples=inner_params["min_child_samples"],
        subsample=inner_params["subsample"],
        colsample_bytree=inner_params["colsample_bytree"],
        entry_threshold=inner_params["entry_threshold"],
        exit_threshold=inner_params["exit_threshold"],
        early_stopping_rounds=30,
        importance_threshold=config.importance_threshold,
    )
    trainer = MidasTrainer(trainer_config)
    try:
        trainer.train(df_filtered, target_filtered, sample_weights=weights)
    except (ValueError, lgb.basic.LightGBMError):
        return None

    if config.use_meta_labeling and trainer._entry_model is not None:
        primary_features = trainer._entry_features
        x_train = df_filtered.select(primary_features).to_numpy()
        proba = np.asarray(trainer._entry_model.predict(x_train))
        p_buy, p_sell = proba[:, 1], proba[:, 2]
        thr = trainer_config.entry_threshold
        primary = np.where(
            (p_buy >= thr) & (p_buy > p_sell), 1,
            np.where((p_sell >= thr) & (p_sell > p_buy), 2, 0),
        ).astype(np.int32)
        signal_mask = primary != 0
        if signal_mask.sum() >= 50:
            meta_df = df_filtered.filter(pl.Series(signal_mask))
            meta_proba = np.where(
                primary[signal_mask] == 1,
                p_buy[signal_mask], p_sell[signal_mask],
            ).astype(np.float32)
            meta_direction = primary[signal_mask]
            meta_label = (
                primary[signal_mask] == target_filtered[signal_mask]
            ).astype(np.int32)
            if 0 < meta_label.sum() < len(meta_label):
                with contextlib.suppress(ValueError, lgb.basic.LightGBMError):
                    trainer.train_meta(
                        meta_df, meta_proba, meta_direction, meta_label,
                    )

    exit_ds = build_exit_dataset(
        df_filtered, target_filtered,
        sl_points=inner_params["sl_fallback"],
        tp_points=inner_params["tp_fallback"],
        timeout_seconds=inner_params["label_timeout"],
        k_sl=inner_params["k_sl"], k_tp=inner_params["k_tp"],
        atr_column=config.atr_column,
    )
    if exit_ds.n_rows >= 50:
        exit_df = df_filtered[exit_ds.row_indices].with_columns(
            pl.Series("pos_unrealized_pnl", exit_ds.unrealized_pnls),
            pl.Series("pos_duration_sec", exit_ds.durations),
            pl.Series("pos_direction", exit_ds.directions),
        )
        with contextlib.suppress(ValueError, lgb.basic.LightGBMError):
            trainer.train_exit(exit_df, exit_ds.exit_labels)

    return trainer


async def _backtest_slice(
    trainer: MidasTrainer,
    eval_start: datetime,
    eval_end: datetime,
    outer_params: dict[str, Any],
    inner_params: dict[str, Any],
    config: GlobalOptunaConfig,
    db: Any,
) -> tuple[float, int, float, list[MidasTrade]]:
    """Backtest a trained model on [eval_start, eval_end] via tick replay."""
    def registry_factory() -> Any:
        return build_default_registry(instrument=config.instrument)

    sim_config = SimConfig(
        sl_points=inner_params["sl_fallback"],
        tp_points=inner_params["tp_fallback"],
        k_sl=inner_params["k_sl"], k_tp=inner_params["k_tp"],
        max_spread=2.0,
        gamma=inner_params["gamma"],
        max_margin_proba=inner_params["max_margin_proba"],
        sizing_threshold=inner_params["entry_threshold"],
        min_risk_pct=inner_params["min_risk_pct"],
        slippage_min_pts=config.slippage_min_pts,
        slippage_max_pts=config.slippage_max_pts,
        slippage_seed=config.slippage_seed,
    )
    eval_cfg = OptimizerConfig(
        instrument=config.instrument,
        train_start=eval_start, train_end=eval_end,
        test_start=eval_start, test_end=eval_end,
        outer_trials=0, inner_trials=0,
        sample_on_candle=config.sample_on_candle,
        sample_rate=config.sample_rate,
        min_daily_trades=config.min_daily_trades,
        trade_deficit_penalty=config.trade_deficit_penalty,
        atr_column=config.atr_column,
        slippage_min_pts=config.slippage_min_pts,
        slippage_max_pts=config.slippage_max_pts,
        slippage_seed=config.slippage_seed,
        score_metric="pnl",
    )
    _, n_trades, win_rate, pnl, trades = await _evaluate_oos_async(
        trainer, sim_config, db, eval_cfg,
        registry_factory, outer_params,
        eval_start=eval_start, eval_end=eval_end,
    )
    return pnl, n_trades, win_rate, trades


def _slice_features(
    df: pl.DataFrame,
    start: datetime,
    end: datetime,
) -> pl.DataFrame:
    """Filter a feature DataFrame to rows whose ``_time`` is in [start, end)."""
    start_ts = start.timestamp()
    end_ts = end.timestamp()
    return df.filter(
        (pl.col("_time") >= start_ts) & (pl.col("_time") < end_ts),
    )


async def run_global_optuna(
    config: GlobalOptunaConfig,
    db: Database,
    output_prefix: str,
) -> GlobalOptunaResult:
    """Run one global Optuna study across all windows.

    Per-trial flow (optimised):
      1. Replay full span ONCE → big feature DataFrame
      2. For each window: slice DF for train, train model, backtest test
      3. Aggregate test PnLs → Optuna objective

    Best-trial validation flow:
      4. For each window: re-train, backtest val
      5. If ``compute_train_robust``: also backtest train → robust score
    """
    if config.objective not in AGGREGATE_KEYS:
        msg = f"objective must be one of {AGGREGATE_KEYS}, got {config.objective!r}"
        raise ValueError(msg)
    if not config.windows:
        msg = "config.windows is empty; provide at least one Window"
        raise ValueError(msg)

    result = GlobalOptunaResult()
    windows = config.windows
    n_windows = len(windows)
    span_start = min(w.train_start for w in windows)
    span_end = max(w.val_end for w in windows)

    sample_registry = build_default_registry(instrument=config.instrument)
    registry_params = sample_registry.all_tunable_params()

    inner_cfg = OptimizerConfig(
        instrument=config.instrument,
        train_start=windows[0].train_start, train_end=windows[0].train_end,
        test_start=windows[0].test_start, test_end=windows[0].test_end,
        outer_trials=0, inner_trials=0,
        sl_range=config.sl_range, tp_range=config.tp_range,
        k_sl_range=config.k_sl_range, k_tp_range=config.k_tp_range,
        gamma_range=config.gamma_range,
        max_margin_proba_range=config.max_margin_proba_range,
        min_risk_pct_range=config.min_risk_pct_range,
        fixed_inner_params=config.fixed_inner_params,
    )

    print(f"\nGlobal Optuna: {config.n_trials} trials x {n_windows} windows"
          f" (objective: {config.objective})")
    print(f"  Span: {span_start.date()} -> {span_end.date()}"
          f"  ({(span_end - span_start).days}j total)")
    for i, w in enumerate(windows):
        print(f"  W{i}: train {w.train_start.date()}->{w.train_end.date()}"
              f" | test {w.test_start.date()}->{w.test_end.date()}"
              f" | val {w.val_start.date()}->{w.val_end.date()}")

    study = optuna.create_study(
        direction="maximize",
        study_name="midas_global",
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
    )

    trials_path = Path(f"{output_prefix}_trials.csv")
    trials_path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = ["trial_idx", *AGGREGATE_KEYS]
    outer_fields = [f"outer__{p.name}" for p in registry_params]
    inner_fields = [
        "inner__n_estimators", "inner__learning_rate", "inner__max_depth",
        "inner__num_leaves", "inner__min_child_samples", "inner__subsample",
        "inner__colsample_bytree", "inner__entry_threshold",
        "inner__exit_threshold",
        "inner__k_sl", "inner__k_tp", "inner__sl_fallback",
        "inner__tp_fallback", "inner__label_timeout",
        "inner__gamma", "inner__max_margin_proba", "inner__min_risk_pct",
    ]
    window_fields = [f"w{i}_pnl" for i in range(n_windows)]
    window_n_fields = [f"w{i}_n_trades" for i in range(n_windows)]
    trials_fieldnames = (
        base_fields + outer_fields + inner_fields
        + window_fields + window_n_fields
    )

    parquet_path = Path(f"{output_prefix}_features.parquet")

    with open(trials_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=trials_fieldnames)
        writer.writeheader()

        for trial_idx in range(config.n_trials):
            trial = study.ask()
            outer_params = (
                dict(config.fixed_outer_params)
                if config.fixed_outer_params is not None
                else _suggest_outer_params(
                    trial, registry_params,
                    range_overrides=config.outer_param_ranges,
                )
            )
            inner_params = _suggest_inner_params(trial, inner_cfg)

            print(f"\n--- Trial {trial_idx + 1}/{config.n_trials} ---")
            print(f"  Replaying full span ({(span_end - span_start).days}j)...")
            n_rows = await _replay_full_span(
                db, config, outer_params, span_start, span_end, parquet_path,
            )
            if n_rows < 100:
                print(f"  SKIP: only {n_rows} feature rows produced")
                study.tell(trial, -1000.0)
                continue
            full_df = pl.read_parquet(parquet_path)
            print(f"  Got {len(full_df)} feature rows; running {n_windows} windows...")

            window_pnls: list[float] = []
            window_n_trades: list[int] = []
            for wi, w in enumerate(windows):
                train_df = _slice_features(full_df, w.train_start, w.train_end)
                trainer = _train_from_slice(train_df, inner_params, config)
                if trainer is None:
                    window_pnls.append(0.0)
                    window_n_trades.append(0)
                    print(f"  W{wi}: SKIP (degenerate train)")
                    continue
                pnl, n_tr, wr, _ = await _backtest_slice(
                    trainer, w.test_start, w.test_end,
                    outer_params, inner_params, config, db,
                )
                window_pnls.append(pnl)
                window_n_trades.append(n_tr)
                print(f"  W{wi}: test pnl={pnl:+.1f}€ n={n_tr} WR={wr*100:.0f}%")

            aggs = compute_aggregates(window_pnls, window_n_trades)
            print(f"  -> {config.objective}={aggs[config.objective]:+.3f}"
                  f" | sum={aggs['sum_pnl']:+.1f}EUR"
                  f" | %pos={aggs['pct_positive']:.0%}"
                  f" | min={aggs['min_pnl']:+.1f}"
                  f" | max={aggs['max_pnl']:+.1f}")

            study.tell(trial, aggs[config.objective])

            row: dict[str, Any] = {"trial_idx": trial_idx, **{k: aggs[k] for k in AGGREGATE_KEYS}}
            for p in registry_params:
                row[f"outer__{p.name}"] = outer_params.get(p.name)
            for k, v in inner_params.items():
                row[f"inner__{k}"] = v
            for i, wp in enumerate(window_pnls):
                row[f"w{i}_pnl"] = round(float(wp), 4)
            for i, n_tr in enumerate(window_n_trades):
                row[f"w{i}_n_trades"] = n_tr
            writer.writerow(row)
            f.flush()

            result.trial_records.append(TrialAggregate(
                trial_idx=trial_idx,
                outer_params=dict(outer_params),
                inner_params=dict(inner_params),
                window_pnls=list(window_pnls),
                window_n_trades=list(window_n_trades),
                aggregates=dict(aggs),
            ))

    # Cleanup the parquet file
    if parquet_path.exists():
        parquet_path.unlink()

    if not result.trial_records:
        msg = "No completed trials — cannot pick a best."
        raise RuntimeError(msg)

    # --- Pick best trial + validation pass ---
    best = max(result.trial_records, key=lambda r: r.aggregates[config.objective])
    result.best_trial_idx = best.trial_idx
    result.best_outer_params = dict(best.outer_params)
    result.best_inner_params = dict(best.inner_params)
    result.best_aggregates = dict(best.aggregates)
    result.best_window_pnls = list(best.window_pnls)
    result.best_window_n_trades = list(best.window_n_trades)

    print(f"\n{'=' * 60}")
    print(f"BEST TRIAL: #{best.trial_idx}  "
          f"({config.objective}={best.aggregates[config.objective]:+.3f})")
    print(f"{'=' * 60}")
    print("Re-replaying full span with best params for validation pass...")
    n_rows = await _replay_full_span(
        db, config, best.outer_params, span_start, span_end, parquet_path,
    )
    if n_rows < 100:
        msg = f"Best trial replay produced only {n_rows} rows; cannot validate."
        raise RuntimeError(msg)
    full_df = pl.read_parquet(parquet_path)

    val_pnls: list[float] = []
    val_trades_cnt: list[int] = []
    train_pnls: list[float] = []
    robust_pnls: list[float] = []

    for wi, w in enumerate(windows):
        train_df = _slice_features(full_df, w.train_start, w.train_end)
        trainer = _train_from_slice(train_df, best.inner_params, config)
        if trainer is None:
            val_pnls.append(0.0)
            val_trades_cnt.append(0)
            train_pnls.append(0.0)
            robust_pnls.append(0.0)
            print(f"  W{wi}: SKIP")
            continue

        val_pnl, val_n, val_wr, _ = await _backtest_slice(
            trainer, w.val_start, w.val_end,
            best.outer_params, best.inner_params, config, db,
        )
        val_pnls.append(val_pnl)
        val_trades_cnt.append(val_n)

        if config.compute_train_robust:
            train_pnl, _, _, _ = await _backtest_slice(
                trainer, w.train_start, w.train_end,
                best.outer_params, best.inner_params, config, db,
            )
            test_pnl_w = best.window_pnls[wi]
            train_days = (w.train_end - w.train_start).days
            test_days = (w.test_end - w.test_start).days
            robust = compute_robust_score(
                test_pnl_w, train_pnl, train_days, test_days,
            )
            train_pnls.append(train_pnl)
            robust_pnls.append(robust)
            print(f"  W{wi}: val={val_pnl:+.1f}EUR n={val_n}"
                  f" | test_pnl={test_pnl_w:+.1f} train_pnl={train_pnl:+.1f}"
                  f" robust={robust:+.1f}")
        else:
            train_pnls.append(0.0)
            robust_pnls.append(0.0)
            print(f"  W{wi}: val pnl={val_pnl:+.1f}EUR n={val_n} WR={val_wr*100:.0f}%")

    if parquet_path.exists():
        parquet_path.unlink()

    val_aggs = compute_aggregates(val_pnls, val_trades_cnt)
    print("\nValidation aggregates:")
    for k in AGGREGATE_KEYS:
        print(f"  {k:<18} = {val_aggs[k]:+.3f}")

    if config.compute_train_robust:
        robust_aggs = compute_aggregates(robust_pnls, val_trades_cnt)
        print("\nRobust score aggregates (best trial, computed on test+train):")
        for k in AGGREGATE_KEYS:
            print(f"  {k:<18} = {robust_aggs[k]:+.3f}")
        result.robust_aggregates = dict(robust_aggs)

    result.val_window_pnls = list(val_pnls)
    result.val_window_n_trades = list(val_trades_cnt)
    result.val_aggregates = dict(val_aggs)
    result.train_window_pnls = list(train_pnls)
    result.robust_window_pnls = list(robust_pnls)

    val_path = Path(f"{output_prefix}_validation.csv")
    val_fields = ["w_idx", "val_pnl", "val_n_trades",
                  "test_pnl", "train_pnl", "robust_score"]
    with open(val_path, "w", newline="") as vf:
        vw = csv.DictWriter(vf, fieldnames=val_fields)
        vw.writeheader()
        for i, (vp, n_tr) in enumerate(zip(val_pnls, val_trades_cnt, strict=True)):
            vw.writerow({
                "w_idx": i,
                "val_pnl": round(float(vp), 4),
                "val_n_trades": n_tr,
                "test_pnl": round(float(best.window_pnls[i]), 4),
                "train_pnl": round(float(train_pnls[i]), 4),
                "robust_score": round(float(robust_pnls[i]), 4),
            })

    params_path = Path(f"{output_prefix}_best_params.json")
    with open(params_path, "w") as pf:
        json.dump({
            "trial_idx": best.trial_idx,
            "objective": config.objective,
            "objective_value": best.aggregates[config.objective],
            "aggregates": best.aggregates,
            "val_aggregates": val_aggs,
            "robust_aggregates": (
                result.robust_aggregates if config.compute_train_robust else {}
            ),
            "outer_params": best.outer_params,
            "inner_params": best.inner_params,
        }, pf, indent=2, default=str)

    return result
