"""Global (panel) Optuna optimizer for the Midas engine.

Unlike ``optimizer.py`` which optimises per-window (one Optuna study per
test slice), this optimiser runs a SINGLE Optuna study across N disjoint
windows. Each trial suggests ONE set of hyperparameters applied to all
N windows: the model is retrained on each window's train slice and
evaluated on each window's test slice. The Optuna objective is a single
scalar aggregate (e.g. Sharpe) computed over the N per-window PnLs,
which forces hyperparameters to generalise across regimes.

At the end of the study, the best trial is re-evaluated on each window's
validation slice (never used during Optuna) to confirm the selection
survives truly unseen data.

Window layout (disjoint):
    |- train 14j -|- test 1j -|- val 1j -|   ← window 0
                                            |- train 14j -|- test 1j -|- val 1j -|  ← window 1
    ...
"""

from __future__ import annotations

import contextlib
import csv
import json
import statistics
import tempfile
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

    Disjoint by construction: ``train_end == test_start``,
    ``test_end == val_start``, and the next window's ``train_start``
    comes after this window's ``val_end``.
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
    """Aggregate metric to maximise. Must be in ``AGGREGATE_KEYS``."""
    sample_on_candle: bool = True
    sample_rate: int = 1
    # Per-window scoring + deficit policy (kept lenient; aggregation does the work)
    min_daily_trades: int = 0
    trade_deficit_penalty: float = 0.0
    atr_column: str = ATR_COLUMN_DEFAULT
    # Inner param ranges (reused from nested optimiser)
    sl_range: tuple[float, float] = (1.5, 8.0)
    tp_range: tuple[float, float] = (1.5, 8.0)
    k_sl_range: tuple[float, float] = (0.5, 3.0)
    k_tp_range: tuple[float, float] = (0.5, 3.0)
    gamma_range: tuple[float, float] = (0.5, 3.0)
    max_margin_proba_range: tuple[float, float] = (0.70, 0.95)
    min_risk_pct_range: tuple[float, float] = (0.001, 0.02)
    # Outer search ranges / fixed values
    fixed_outer_params: dict[str, Any] | None = None
    outer_param_ranges: dict[str, tuple[float, float]] | None = None
    fixed_inner_params: dict[str, Any] | None = None
    # Simulator
    slippage_min_pts: float = 0.0
    slippage_max_pts: float = 0.0
    slippage_seed: int | None = None
    # Feature importance filtering + meta-labelling
    importance_threshold: float = 0.0
    use_meta_labeling: bool = False


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
    trial_records: list[TrialAggregate] = field(default_factory=list)


def generate_disjoint_windows(
    data_start: datetime,
    data_end: datetime,
    train_days: int = 14,
    test_days: int = 1,
    val_days: int = 1,
    step_days: int | None = None,
    n_windows: int | None = None,
) -> list[Window]:
    """Build disjoint windows covering ``data_start..data_end``.

    Each window is ``train_days + test_days + val_days`` long. Windows
    are non-overlapping: the next window starts ``step_days`` after the
    current one's start, defaulting to one full window length so that
    windows are back-to-back with no overlap.

    Args:
        data_start: Earliest date available.
        data_end: Latest date available (exclusive end).
        train_days: Length of each train slice in days.
        test_days: Length of each test slice in days.
        val_days: Length of each validation slice in days.
        step_days: Stride between consecutive window starts. Defaults
            to ``train_days + test_days + val_days`` (fully disjoint).
        n_windows: Max number of windows to produce. If ``None``, fit
            as many as possible in the available range.
    """
    window_span = train_days + test_days + val_days
    if step_days is None:
        step_days = window_span
    if step_days < window_span:
        msg = (
            f"step_days ({step_days}) must be >= "
            f"train+test+val ({window_span}) to keep windows disjoint"
        )
        raise ValueError(msg)

    windows: list[Window] = []
    cursor = data_start
    while cursor + timedelta(days=window_span) <= data_end:
        train_start = cursor
        train_end = train_start + timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + timedelta(days=test_days)
        val_start = test_end
        val_end = val_start + timedelta(days=val_days)
        windows.append(Window(
            train_start=train_start, train_end=train_end,
            test_start=test_start, test_end=test_end,
            val_start=val_start, val_end=val_end,
        ))
        if n_windows is not None and len(windows) >= n_windows:
            break
        cursor = cursor + timedelta(days=step_days)
    return windows


def compute_aggregates(
    window_pnls: list[float],
    window_n_trades: list[int] | None = None,
) -> dict[str, float]:
    """Compute all per-trial aggregates from N per-window PnLs.

    Returns a dict with the keys in ``AGGREGATE_KEYS``. Designed to be
    robust when all windows are zero (sharpe=0, not NaN).
    """
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

    # Sharpe over windows (not time): mean / stdev across N windows.
    # Guard against zero stdev (all identical) → sharpe = 0.
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


async def _train_and_evaluate_window(
    window: Window,
    outer_params: dict[str, Any],
    inner_params: dict[str, Any],
    config: GlobalOptunaConfig,
    db: Any,
    *,
    eval_slice: str = "test",
) -> tuple[float, int, float, list[MidasTrade]]:
    """Train a model on window.train, evaluate on window.test or .val.

    Returns ``(pnl, n_trades, win_rate, trades)``. A silent failure
    (insufficient features, label degenerate, LightGBM error) returns
    ``(0.0, 0, 0.0, [])``.
    """
    def registry_factory() -> Any:
        return build_default_registry(instrument=config.instrument)

    # --- 1. Replay train window to get features ---
    registry = registry_factory()
    registry.configure_all(outer_params)

    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "features.parquet"
        replay = ReplayEngine(
            db, registry,
            ReplayConfig(
                instrument=config.instrument,
                start=window.train_start, end=window.train_end,
                sample_on_candle=config.sample_on_candle,
                sample_rate=config.sample_rate,
                output_path=parquet_path,
            ),
        )
        replay_result = await replay.run()
        if not parquet_path.exists() or replay_result.feature_rows < 100:
            return 0.0, 0, 0.0, []
        df = pl.read_parquet(parquet_path)

    # --- 2. Relabel + filter ---
    label_result = relabel_dataframe(
        df,
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
        return 0.0, 0, 0.0, []
    df_filtered = df.filter(pl.Series(mask))
    target_filtered = target[mask]
    buy_pnls = [label_result.buy_pnls[i] for i, m in enumerate(mask) if m]
    sell_pnls = [label_result.sell_pnls[i] for i, m in enumerate(mask) if m]
    weights = MidasTrainer.build_sample_weights(
        buy_pnls, sell_pnls, target_filtered,
    )

    # --- 3. Train entry + optional meta + exit ---
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
        return 0.0, 0, 0.0, []

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

    # --- 4. Backtest on the selected slice ---
    eval_start = window.test_start if eval_slice == "test" else window.val_start
    eval_end = window.test_end if eval_slice == "test" else window.val_end
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
    # Build an OptimizerConfig shim for _evaluate_oos_async
    eval_cfg = OptimizerConfig(
        instrument=config.instrument,
        train_start=window.train_start, train_end=window.train_end,
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


async def run_global_optuna(
    config: GlobalOptunaConfig,
    db: Database,
    output_prefix: str,
) -> GlobalOptunaResult:
    """Run one global Optuna study across all windows.

    Writes:
      - ``{output_prefix}_trials.csv`` — per trial: params + aggregates + per-window PnLs
      - ``{output_prefix}_validation.csv`` — best trial's val PnL per window

    Args:
        config: Global optimisation config (windows, n_trials, objective, …).
        db: Database connection.
        output_prefix: File path prefix for logs.

    Returns:
        GlobalOptunaResult with best params, best aggregates, and
        validation-window metrics.
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

    # Sample registry to enumerate tunable params
    sample_registry = build_default_registry(instrument=config.instrument)
    registry_params = sample_registry.all_tunable_params()

    # Optimizer shim to reuse _suggest_inner_params
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
    for i, w in enumerate(windows):
        print(f"  W{i}: train {w.train_start.date()}→{w.train_end.date()}"
              f" | test {w.test_start.date()}→{w.test_end.date()}"
              f" | val {w.val_start.date()}→{w.val_end.date()}")

    study = optuna.create_study(
        direction="maximize",
        study_name="midas_global",
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
    )

    # Open per-trial CSV writer
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
            window_pnls: list[float] = []
            window_n_trades: list[int] = []
            for wi, w in enumerate(windows):
                pnl, n_tr, wr, _ = await _train_and_evaluate_window(
                    w, outer_params, inner_params, config, db,
                    eval_slice="test",
                )
                window_pnls.append(pnl)
                window_n_trades.append(n_tr)
                print(f"  W{wi}: test pnl={pnl:+.1f}€ n={n_tr} WR={wr*100:.0f}%")

            aggs = compute_aggregates(window_pnls, window_n_trades)
            print(f"  → {config.objective}={aggs[config.objective]:+.3f}"
                  f" | sum={aggs['sum_pnl']:+.1f}€"
                  f" | %pos={aggs['pct_positive']:.0%}"
                  f" | min={aggs['min_pnl']:+.1f}"
                  f" | max={aggs['max_pnl']:+.1f}")

            objective_score = aggs[config.objective]
            study.tell(trial, objective_score)

            # Log this trial
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

    # --- Pick best trial + validate on held-out val slices ---
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
    print("Running validation on val slices…")

    val_pnls: list[float] = []
    val_trades_cnt: list[int] = []
    for wi, w in enumerate(windows):
        pnl, n_tr, wr, _ = await _train_and_evaluate_window(
            w, best.outer_params, best.inner_params, config, db,
            eval_slice="val",
        )
        val_pnls.append(pnl)
        val_trades_cnt.append(n_tr)
        print(f"  W{wi} val: pnl={pnl:+.1f}€ n={n_tr} WR={wr*100:.0f}%")
    val_aggs = compute_aggregates(val_pnls, val_trades_cnt)
    print("\nValidation aggregates:")
    for k in AGGREGATE_KEYS:
        print(f"  {k:<18} = {val_aggs[k]:+.3f}")

    result.val_window_pnls = list(val_pnls)
    result.val_window_n_trades = list(val_trades_cnt)
    result.val_aggregates = dict(val_aggs)

    # Write validation CSV
    val_path = Path(f"{output_prefix}_validation.csv")
    val_fields = ["w_idx", "val_pnl", "val_n_trades"]
    with open(val_path, "w", newline="") as vf:
        vw = csv.DictWriter(vf, fieldnames=val_fields)
        vw.writeheader()
        for i, (vp, n_tr) in enumerate(zip(val_pnls, val_trades_cnt, strict=True)):
            vw.writerow({
                "w_idx": i,
                "val_pnl": round(float(vp), 4),
                "val_n_trades": n_tr,
            })

    # Dump best params as JSON
    params_path = Path(f"{output_prefix}_best_params.json")
    with open(params_path, "w") as pf:
        json.dump({
            "trial_idx": best.trial_idx,
            "objective": config.objective,
            "objective_value": best.aggregates[config.objective],
            "aggregates": best.aggregates,
            "val_aggregates": val_aggs,
            "outer_params": best.outer_params,
            "inner_params": best.inner_params,
        }, pf, indent=2, default=str)

    return result
