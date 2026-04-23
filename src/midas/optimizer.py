"""Nested Optuna optimizer for the Midas scalping engine.

Outer loop: tunes feature extractor params (expensive, re-replays ticks).
Inner loop: tunes SL/TP + LightGBM hyperparams (cheap, relabels in-memory).

Score = walk-forward OOS metric (pnl with trade deficit penalty by default).
"""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl

from src.midas.labeler import build_exit_dataset, relabel_dataframe
from src.midas.replay_engine import (
    ReplayConfig,
    ReplayEngine,
    build_default_registry,
)
from src.midas.trade_simulator import MidasTrade, SimConfig, TradeSimulator
from src.midas.trainer import MidasTrainer, TrainerConfig
from src.midas.types import ATR_COLUMN_DEFAULT

if TYPE_CHECKING:
    from src.common.db import Database
    from src.midas.types import Tick


INNER_PARAM_KEYS: frozenset[str] = frozenset({
    "n_estimators", "learning_rate", "max_depth", "num_leaves",
    "min_child_samples", "subsample", "colsample_bytree",
    "entry_threshold", "exit_threshold",
    "k_sl", "k_tp", "sl_fallback", "tp_fallback",
    "sl_points", "tp_points", "label_timeout",
    "gamma", "max_margin_proba", "min_risk_pct",
})
"""Parameter names that belong to the inner Optuna loop.

Used by CLI scripts to separate inner vs outer params when loading
a YAML file with ``--fix-outer-params``.
"""


def load_fixed_outer_params(path: str) -> dict[str, Any]:
    """Load outer params from a YAML file, filtering out inner params.

    Args:
        path: Path to YAML file with all params.

    Returns:
        Dict of outer-only params (extractor params).
    """
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        msg = f"Expected a YAML mapping in {path}, got {type(raw).__name__}"
        raise ValueError(msg)
    return {
        k: v for k, v in raw.items()
        if not k.startswith("_") and k not in INNER_PARAM_KEYS
    }


def load_fixed_inner_params(path: str) -> dict[str, Any]:
    """Load fixed inner params from a YAML file.

    Only keys that are in ``INNER_PARAM_KEYS`` are kept.

    Args:
        path: Path to YAML file with ``{name: value}`` entries.

    Returns:
        Dict of inner params to fix during optimization.
    """
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        msg = f"Expected a YAML mapping in {path}, got {type(raw).__name__}"
        raise ValueError(msg)
    return {k: v for k, v in raw.items() if k in INNER_PARAM_KEYS}


def load_outer_param_ranges(path: str) -> dict[str, tuple[float, float]]:
    """Load restricted outer param ranges from a YAML file.

    Expected format::

        atr_period: [10, 16]
        liq_lookback: [100, 200]

    Args:
        path: Path to YAML file with ``{name: [low, high]}`` entries.

    Returns:
        Dict of ``{name: (low, high)}`` range overrides.
    """
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        msg = f"Expected a YAML mapping in {path}, got {type(raw).__name__}"
        raise ValueError(msg)
    ranges: dict[str, tuple[float, float]] = {}
    for k, v in raw.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            ranges[k] = (float(v[0]), float(v[1]))
    return ranges


def _count_trading_days(start: datetime, end: datetime) -> int:
    """Count weekdays (Mon-Fri) between start (inclusive) and end (exclusive).

    Uses O(1) formula instead of day-by-day iteration.

    Args:
        start: Window start datetime.
        end: Window end datetime.

    Returns:
        Number of trading days (at least 1).
    """
    total_days = (end.date() - start.date()).days
    if total_days <= 0:
        return 1
    full_weeks, remainder = divmod(total_days, 7)
    days = full_weeks * 5
    start_weekday = start.weekday()
    for offset in range(remainder):
        if (start_weekday + offset) % 7 < 5:
            days += 1
    return max(days, 1)


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """Nested Optuna configuration.

    Args:
        instrument: Instrument to optimize.
        train_start: Training data start.
        train_end: Training data end.
        test_start: OOS test data start.
        test_end: OOS test data end.
        outer_trials: Number of outer loop trials (extractor params).
        inner_trials: Number of inner loop trials (SL/TP + LightGBM).
        sample_on_candle: Extract features on candle close (default).
        sample_rate: Legacy tick-based sampling.
        score_metric: Metric to optimize (composite uses PnL + trade deficit).
        min_daily_trades: Minimum expected trades per trading day.
        trade_deficit_penalty: Penalty per missing trade below minimum.
        validation_start: Validation window start (None = no validation).
        validation_end: Validation window end (None = no validation).
        fixed_inner_params: Fixed inner params (skip suggestion for these).
        sl_range: SL search range in points (fixed mode fallback).
        tp_range: TP search range in points (fixed mode fallback).
        k_sl_range: k_sl multiplier search range (ATR mode).
        k_tp_range: k_tp multiplier search range (ATR mode).
        atr_column: Column name for ATR values.
        fixed_outer_params: Fixed extractor params (skip outer search).
        slippage_min_pts: Min adverse slippage per market order (points).
        slippage_max_pts: Max adverse slippage per market order (points).
        slippage_seed: RNG seed for reproducible slippage sequences.
    """

    instrument: str = "XAUUSD"
    train_start: datetime = field(default_factory=datetime.now)
    train_end: datetime = field(default_factory=datetime.now)
    test_start: datetime = field(default_factory=datetime.now)
    test_end: datetime = field(default_factory=datetime.now)
    outer_trials: int = 30
    inner_trials: int = 30
    sample_on_candle: bool = True
    sample_rate: int = 1
    score_metric: str = "composite"
    min_daily_trades: int = 10
    trade_deficit_penalty: float = 10.0
    validation_start: datetime | None = None
    validation_end: datetime | None = None
    fixed_inner_params: dict[str, Any] | None = None
    sl_range: tuple[float, float] = (1.5, 8.0)
    tp_range: tuple[float, float] = (1.5, 8.0)
    k_sl_range: tuple[float, float] = (0.5, 3.0)
    k_tp_range: tuple[float, float] = (0.5, 3.0)
    gamma_range: tuple[float, float] = (0.5, 3.0)
    max_margin_proba_range: tuple[float, float] = (0.70, 0.95)
    min_risk_pct_range: tuple[float, float] = (0.001, 0.02)
    atr_column: str = ATR_COLUMN_DEFAULT
    fixed_outer_params: dict[str, Any] | None = None
    outer_param_ranges: dict[str, tuple[float, float]] | None = None
    slippage_min_pts: float = 0.0
    slippage_max_pts: float = 0.0
    slippage_seed: int | None = None
    importance_threshold: float = 0.0
    use_meta_labeling: bool = False
    track_train_score: bool = False


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """Best-inner result for a single outer trial.

    One record per outer trial, capturing the best inner params and
    all OOS trades for that configuration.
    """

    window_idx: int
    outer_idx: int
    score: float
    n_trades: int
    win_rate: float
    pnl: float
    outer_params: dict[str, Any]
    inner_params: dict[str, Any]
    trades: list[MidasTrade]
    val_score: float | None = None
    val_n_trades: int = 0
    val_win_rate: float = 0.0
    val_pnl: float = 0.0
    train_score: float | None = None
    train_n_trades: int = 0
    train_win_rate: float = 0.0
    train_pnl: float = 0.0


@dataclass
class OptimizationResult:
    """Result of a nested optimization run."""

    best_outer_params: dict[str, Any] = field(default_factory=dict)
    best_inner_params: dict[str, Any] = field(default_factory=dict)
    best_trainer: MidasTrainer | None = None
    best_score: float = 0.0
    total_outer_trials: int = 0
    total_inner_trials: int = 0
    best_n_trades: int = 0
    best_win_rate: float = 0.0
    best_pnl: float = 0.0
    best_trades: list[MidasTrade] = field(default_factory=list)
    trial_records: list[TrialRecord] = field(default_factory=list)
    val_score: float | None = None
    val_n_trades: int = 0
    val_win_rate: float = 0.0
    val_pnl: float = 0.0


def _suggest_outer_params(
    trial: optuna.Trial,
    registry_params: list[Any],
    range_overrides: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Suggest extractor params only for the outer loop.

    Args:
        trial: Optuna trial.
        registry_params: Default param descriptors from the registry.
        range_overrides: Optional ``{name: (low, high)}`` to restrict
            search ranges (from ``--outer-ranges-from``).
    """
    params: dict[str, Any] = {}
    for p in registry_params:
        low, high = p.low, p.high
        if range_overrides and p.name in range_overrides:
            low, high = range_overrides[p.name]
        if p.param_type == "int":
            params[p.name] = trial.suggest_int(p.name, int(low), int(high))
        else:
            params[p.name] = trial.suggest_float(p.name, low, high)
    return params


def _suggest_inner_params(
    trial: optuna.Trial,
    config: OptimizerConfig,
) -> dict[str, Any]:
    """Suggest k_sl/k_tp + sizing + LightGBM + entry/exit thresholds.

    Params listed in ``config.fixed_inner_params`` are not suggested —
    the fixed value is used instead, reducing the search space.
    """
    fixed = config.fixed_inner_params or {}

    def _float(name: str, low: float, high: float, **kw: Any) -> float:
        if name in fixed:
            return float(fixed[name])
        return trial.suggest_float(name, low, high, **kw)

    def _int(name: str, low: int, high: int) -> int:
        if name in fixed:
            return int(fixed[name])
        return trial.suggest_int(name, low, high)

    return {
        # ATR-based SL/TP multipliers (relabeled per inner trial)
        "k_sl": _float("k_sl", config.k_sl_range[0], config.k_sl_range[1]),
        "k_tp": _float("k_tp", config.k_tp_range[0], config.k_tp_range[1]),
        "sl_fallback": _float(
            "sl_fallback", config.sl_range[0], config.sl_range[1],
        ),
        "tp_fallback": _float(
            "tp_fallback", config.tp_range[0], config.tp_range[1],
        ),
        "label_timeout": _float("label_timeout", 60.0, 600.0),
        # Dynamic sizing
        "gamma": _float(
            "gamma", config.gamma_range[0], config.gamma_range[1],
        ),
        "max_margin_proba": _float(
            "max_margin_proba",
            config.max_margin_proba_range[0],
            config.max_margin_proba_range[1],
        ),
        "min_risk_pct": _float(
            "min_risk_pct",
            config.min_risk_pct_range[0],
            config.min_risk_pct_range[1],
        ),
        # LightGBM hyperparams (shared entry + exit).
        # Ranges sized for ~1M-row training sets (650j x M1 panel mode).
        # num_leaves <= 2^max_depth keeps depth as the effective constraint;
        # min_child_samples scales with dataset size to avoid leaf overfit.
        "n_estimators": _int("n_estimators", 100, 1500),
        "learning_rate": _float("learning_rate", 0.005, 0.2, log=True),
        "max_depth": _int("max_depth", 4, 12),
        "num_leaves": _int("num_leaves", 31, 255),
        "min_child_samples": _int("min_child_samples", 50, 1000),
        "subsample": _float("subsample", 0.5, 1.0),
        "colsample_bytree": _float("colsample_bytree", 0.5, 1.0),
        "entry_threshold": _float("entry_threshold", 0.25, 0.60),
        # Exit model threshold
        "exit_threshold": _float("exit_threshold", 0.30, 0.80),
    }


def compute_robust_score(
    test_pnl: float,
    train_pnl: float,
    train_days: float,
    test_days: float,
) -> float:
    """Apply an overfit penalty to the test PnL.

    A trial is "overfit" when its train PnL is much higher than the
    pro-rata expected level given the test result. Concretely:

        expected_ratio = train_days / test_days       (e.g. 14/2 = 7)
        actual_ratio   = train_pnl / test_pnl
        penalty        = max(0, actual_ratio - expected_ratio) / expected_ratio
        robust_score   = test_pnl / (1 + penalty)

    The penalty is only applied when both PnLs are positive — a trial
    where the train was negative or the test was non-positive is left
    unchanged (the score is just ``test_pnl``). This penalises configs
    that look unrealistically good on train (likely overfit) while
    leaving "modest, balanced" trials intact.

    Args:
        test_pnl: PnL on the OOS test window.
        train_pnl: PnL on the train window (replay through the model).
        train_days: Number of trading days in the train window.
        test_days: Number of trading days in the test window.

    Returns:
        The penalised score.
    """
    if test_pnl <= 0 or train_pnl <= 0 or test_days <= 0:
        return test_pnl
    expected_ratio = train_days / test_days
    actual_ratio = train_pnl / test_pnl
    if actual_ratio <= expected_ratio:
        return test_pnl
    penalty = (actual_ratio - expected_ratio) / expected_ratio
    return test_pnl / (1.0 + penalty)


async def _evaluate_oos_async(
    trainer: MidasTrainer,
    sim_config: SimConfig,
    db: Any,
    config: OptimizerConfig,
    registry_factory: Any,
    extractor_params: dict[str, Any],
    *,
    eval_start: datetime | None = None,
    eval_end: datetime | None = None,
) -> tuple[float, int, float, float, list[MidasTrade]]:
    """Run OOS test and return (score, n_trades, win_rate, total_pnl, trades).

    By default evaluates on ``config.test_start`` → ``config.test_end``.
    Pass ``eval_start``/``eval_end`` to evaluate on a different window
    (e.g. the train window for a train-vs-OOS correlation study).
    """
    simulator = TradeSimulator(sim_config)
    _trainer = trainer
    _simulator = simulator
    _atr_col = config.atr_column
    latest_features: dict[str, float] = {}

    def callback(
        tick: Tick,
        features: dict[str, float],
        *,
        tr: MidasTrainer = _trainer,
        sim: TradeSimulator = _simulator,
        atr_col: str = _atr_col,
        feat: dict[str, float] = latest_features,
    ) -> None:
        feat.update(features)
        # Exit model at sample time (candle close when sample_on_candle=True)
        if tr.has_exit_model and sim.open_count > 0:
            ctx = sim.get_position_context(tick)
            if ctx is not None:
                should_close, _ = tr.predict_exit(
                    feat,
                    pos_unrealized_pnl=ctx["pos_unrealized_pnl"],
                    pos_duration_sec=ctx["pos_duration_sec"],
                    pos_direction=ctx["pos_direction"],
                )
                if should_close:
                    sim.early_close(tick)
        signal, confidence = tr.predict(features)
        sim.on_signal(
            tick, signal,
            atr=features.get(atr_col, 0.0),
            proba=confidence,
        )

    last_tick_holder: list[Tick | None] = [None]

    def exit_hook(
        tick: Tick,
        *,
        sim: TradeSimulator = _simulator,
        holder: list[Tick | None] = last_tick_holder,
    ) -> None:
        # SL/TP mechanical exits (must stay per-tick)
        sim.on_tick(tick)
        holder[0] = tick

    registry = registry_factory()
    registry.configure_all(extractor_params)

    start = eval_start if eval_start is not None else config.test_start
    end = eval_end if eval_end is not None else config.test_end
    engine = ReplayEngine(
        db, registry,
        ReplayConfig(
            instrument=config.instrument,
            start=start,
            end=end,
            sample_on_candle=config.sample_on_candle,
            sample_rate=config.sample_rate,
        ),
        tick_callback=callback,
        every_tick_hook=exit_hook,
    )
    await engine.run()

    # Force-close remaining positions at last market price
    if simulator.open_count > 0 and last_tick_holder[0] is not None:
        simulator.close_all(last_tick_holder[0])

    trades = simulator.closed_trades
    n_trades = len(trades)

    if n_trades == 0:
        total_pnl = 0.0
        win_rate = 0.0
    else:
        total_pnl = sum(t.pnl for t in trades)
        wins = sum(1 for t in trades if t.is_win)
        win_rate = wins / n_trades

    # Trade deficit penalty (applies to composite and pnl metrics)
    trading_days = _count_trading_days(start, end)
    min_trades = config.min_daily_trades * trading_days
    deficit_penalty = (
        max(0, min_trades - n_trades) * config.trade_deficit_penalty
    )

    if config.score_metric == "win_rate":
        score = win_rate
    elif config.score_metric == "pnl_per_trade":
        score = total_pnl / n_trades if n_trades > 0 else -1000.0
    else:
        # "composite", "pnl", and "robust" all start from PnL - deficit.
        # "robust" applies an additional train-overfit penalty in the
        # inner loop after computing train_pnl.
        score = total_pnl - deficit_penalty

    return score, n_trades, win_rate, total_pnl, trades


async def run_nested_optuna(
    config: OptimizerConfig,
    db: Database,
    *,
    window_idx: int = 0,
) -> OptimizationResult:
    """Run the nested Optuna optimization.

    Outer loop: extractor params → replay ticks → features DataFrame.
    Inner loop: SL/TP → relabel DataFrame → train LightGBM → evaluate OOS.

    Args:
        config: Optimizer configuration.
        db: Database connection.
        window_idx: Walk-forward window index (0 for standalone runs).
    """
    result = OptimizationResult()

    def registry_factory() -> Any:
        return build_default_registry(instrument=config.instrument)

    sample_registry = registry_factory()
    registry_params = sample_registry.all_tunable_params()

    has_validation = (
        config.validation_start is not None and config.validation_end is not None
    )

    n_fixed_inner = len(config.fixed_inner_params) if config.fixed_inner_params else 0
    n_inner_params = 17 - n_fixed_inner

    print(f"\nNested Optuna: {config.outer_trials} outer x "
          f"{config.inner_trials} inner trials")
    print(f"  Outer: {len(registry_params)} extractor params")
    print(f"  Inner: {n_inner_params} params"
          f"{f' ({n_fixed_inner} fixed)' if n_fixed_inner else ''}")
    print(f"  ATR column: {config.atr_column}")
    if config.outer_param_ranges:
        print(f"  Outer ranges restricted: {config.outer_param_ranges}")
    if config.fixed_inner_params:
        print(f"  Fixed inner: {config.fixed_inner_params}")
    print(f"  Train:      {config.train_start.date()}"
          f" → {config.train_end.date()}")
    print(f"  Selection:  {config.test_start.date()}"
          f" → {config.test_end.date()}")
    if has_validation:
        print(f"  Validation: {config.validation_start.date()}"  # type: ignore[union-attr]
              f" → {config.validation_end.date()}")  # type: ignore[union-attr]

    best_score = -float("inf")
    best_outer: dict[str, Any] = {}
    best_inner: dict[str, Any] = {}
    best_trades = 0
    best_wr = 0.0
    best_pnl = 0.0

    outer_study = optuna.create_study(
        direction="maximize",
        study_name="midas_outer",
        sampler=optuna.samplers.TPESampler(n_startup_trials=5),
    )

    for outer_i in range(config.outer_trials):
        outer_trial = outer_study.ask()
        if config.fixed_outer_params is not None:
            extractor_params = dict(config.fixed_outer_params)
        else:
            extractor_params = _suggest_outer_params(
                outer_trial, registry_params,
                range_overrides=config.outer_param_ranges,
            )

        print(f"\n--- Outer trial {outer_i + 1}/{config.outer_trials}"
              f"{' [FIXED]' if config.fixed_outer_params else ''} ---")

        # --- Outer: replay to extract features (no labeling) ---
        registry = registry_factory()
        registry.configure_all(extractor_params)

        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "features.parquet"

            replay = ReplayEngine(
                db, registry,
                ReplayConfig(
                    instrument=config.instrument,
                    start=config.train_start,
                    end=config.train_end,
                    sample_on_candle=config.sample_on_candle,
                    sample_rate=config.sample_rate,
                    output_path=parquet_path,
                ),
            )
            replay_result = await replay.run()

            if not parquet_path.exists() or replay_result.feature_rows < 100:
                print(f"  SKIP: insufficient features "
                      f"({replay_result.feature_rows})")
                outer_study.tell(outer_trial, -1000.0)
                continue

            df = pl.read_parquet(parquet_path)

        print(f"  Features: {len(df)} rows")

        # --- Inner: relabel + train + evaluate for each SL/TP combo ---
        inner_study = optuna.create_study(
            direction="maximize",
            study_name=f"midas_inner_{outer_i}",
            sampler=optuna.samplers.TPESampler(n_startup_trials=5),
        )

        best_inner_score = -float("inf")
        best_inner_trainer: MidasTrainer | None = None
        best_inner_trades = 0
        best_inner_wr = 0.0
        best_inner_pnl = 0.0
        best_inner_train_pnl = 0.0
        best_inner_train_n_trades = 0
        best_inner_train_wr = 0.0
        best_inner_params_local: dict[str, Any] = {}
        best_inner_trades_list: list[MidasTrade] = []

        for _inner_i in range(config.inner_trials):
            inner_trial = inner_study.ask()
            inner_params = _suggest_inner_params(inner_trial, config)

            k_sl = inner_params["k_sl"]
            k_tp = inner_params["k_tp"]
            sl_fallback = inner_params["sl_fallback"]
            tp_fallback = inner_params["tp_fallback"]
            timeout = inner_params["label_timeout"]

            # Relabel in-memory with ATR-based SL/TP (fast, no DB)
            label_result = relabel_dataframe(
                df,
                sl_points=sl_fallback,
                tp_points=tp_fallback,
                timeout_seconds=timeout,
                k_sl=k_sl,
                k_tp=k_tp,
                atr_column=config.atr_column,
            )

            # Build target + filter
            target = MidasTrainer.build_target(
                label_result.buy_labels,
                label_result.sell_labels,
            )
            buy_arr = np.asarray(label_result.buy_labels)
            sell_arr = np.asarray(label_result.sell_labels)
            mask = ~((buy_arr == -1) & (sell_arr == -1))

            if mask.sum() < 100:
                inner_study.tell(inner_trial, -1000.0)
                continue

            df_filtered = df.filter(pl.Series(mask))
            target_filtered = target[mask]

            # PnL weights
            buy_pnls = [label_result.buy_pnls[i] for i, m in enumerate(mask) if m]
            sell_pnls = [label_result.sell_pnls[i] for i, m in enumerate(mask) if m]
            weights = MidasTrainer.build_sample_weights(
                buy_pnls, sell_pnls, target_filtered,
            )

            # Train entry model
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
                inner_study.tell(inner_trial, -1000.0)
                continue

            # Train meta model (Lopez de Prado gate on primary signal)
            if config.use_meta_labeling and trainer._entry_model is not None:
                # Use the feature list the primary actually trained on
                # (importance_threshold may have filtered some out).
                primary_features = trainer._entry_features
                x_train = df_filtered.select(primary_features).to_numpy()
                proba = np.asarray(trainer._entry_model.predict(x_train))
                p_buy = proba[:, 1]
                p_sell = proba[:, 2]
                thr = trainer_config.entry_threshold
                # Primary signal per row: 0=PASS, 1=BUY, 2=SELL
                primary = np.where(
                    (p_buy >= thr) & (p_buy > p_sell), 1,
                    np.where((p_sell >= thr) & (p_sell > p_buy), 2, 0),
                ).astype(np.int32)
                signal_mask = primary != 0
                if signal_mask.sum() >= 50:
                    meta_df = df_filtered.filter(pl.Series(signal_mask))
                    meta_proba = np.where(
                        primary[signal_mask] == 1,
                        p_buy[signal_mask],
                        p_sell[signal_mask],
                    ).astype(np.float32)
                    meta_direction = primary[signal_mask]
                    # meta_label: 1 if primary == target (correct direction)
                    meta_label = (
                        primary[signal_mask]
                        == target_filtered[signal_mask]
                    ).astype(np.int32)
                    if meta_label.sum() > 0 and meta_label.sum() < len(meta_label):
                        with contextlib.suppress(ValueError, lgb.basic.LightGBMError):
                            trainer.train_meta(
                                meta_df, meta_proba, meta_direction, meta_label,
                            )

            # Train exit model (optimal-close labeling)
            exit_ds = build_exit_dataset(
                df_filtered,
                target_filtered,
                sl_points=sl_fallback,
                tp_points=tp_fallback,
                timeout_seconds=timeout,
                k_sl=k_sl,
                k_tp=k_tp,
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

            # Evaluate on OOS with ATR-based SL/TP + dynamic sizing
            sim_config = SimConfig(
                sl_points=sl_fallback,
                tp_points=tp_fallback,
                k_sl=k_sl,
                k_tp=k_tp,
                max_spread=2.0,
                gamma=inner_params["gamma"],
                max_margin_proba=inner_params["max_margin_proba"],
                sizing_threshold=inner_params["entry_threshold"],
                min_risk_pct=inner_params["min_risk_pct"],
                slippage_min_pts=config.slippage_min_pts,
                slippage_max_pts=config.slippage_max_pts,
                slippage_seed=config.slippage_seed,
            )
            score, n_tr, wr, pnl, trades_list = await _evaluate_oos_async(
                trainer, sim_config, db, config,
                registry_factory, extractor_params,
            )

            inner_train_pnl = 0.0
            inner_train_n_trades = 0
            inner_train_wr = 0.0
            if config.score_metric == "robust":
                # Compute train backtest for the overfit penalty.
                (
                    _, inner_train_n_trades, inner_train_wr,
                    inner_train_pnl, _,
                ) = await _evaluate_oos_async(
                    trainer, sim_config, db, config,
                    registry_factory, extractor_params,
                    eval_start=config.train_start,
                    eval_end=config.train_end,
                )
                train_days = _count_trading_days(
                    config.train_start, config.train_end,
                )
                test_days = _count_trading_days(
                    config.test_start, config.test_end,
                )
                score = compute_robust_score(
                    pnl, inner_train_pnl, train_days, test_days,
                )

            inner_study.tell(inner_trial, score)

            if score > best_inner_score:
                best_inner_score = score
                best_inner_trainer = trainer
                best_inner_params_local = dict(inner_params)
                best_inner_trades = n_tr
                best_inner_wr = wr
                best_inner_pnl = pnl
                best_inner_trades_list = trades_list
                best_inner_train_pnl = inner_train_pnl
                best_inner_train_n_trades = inner_train_n_trades
                best_inner_train_wr = inner_train_wr

        # Report to outer
        outer_study.tell(outer_trial, best_inner_score)
        result.total_inner_trials += config.inner_trials

        # Train backtest: evaluate the best-inner model on the TRAIN window.
        # Lets callers measure train/OOS correlation to detect overfitting.
        t_sc: float | None = None
        t_nt = 0
        t_wr_val = 0.0
        t_pnl_val = 0.0
        # If robust scoring already computed train_pnl in the inner loop,
        # reuse it instead of replaying.
        if config.score_metric == "robust" and best_inner_trainer is not None:
            t_sc = best_inner_score
            t_pnl_val = best_inner_train_pnl
            t_nt = best_inner_train_n_trades
            t_wr_val = best_inner_train_wr
        elif (config.track_train_score
                and best_inner_trainer is not None
                and best_inner_params_local):
            t_sim = SimConfig(
                sl_points=best_inner_params_local.get("sl_fallback", 3.0),
                tp_points=best_inner_params_local.get("tp_fallback", 3.0),
                k_sl=best_inner_params_local.get("k_sl", 1.0),
                k_tp=best_inner_params_local.get("k_tp", 1.0),
                max_spread=2.0,
                gamma=best_inner_params_local.get("gamma", 1.0),
                max_margin_proba=best_inner_params_local.get(
                    "max_margin_proba", 0.80,
                ),
                sizing_threshold=best_inner_params_local.get(
                    "entry_threshold", 0.5,
                ),
                min_risk_pct=best_inner_params_local.get(
                    "min_risk_pct", 0.005,
                ),
                slippage_min_pts=config.slippage_min_pts,
                slippage_max_pts=config.slippage_max_pts,
                slippage_seed=config.slippage_seed,
            )
            t_sc, t_nt, t_wr_val, t_pnl_val, _ = await _evaluate_oos_async(
                best_inner_trainer, t_sim, db, config,
                registry_factory, extractor_params,
                eval_start=config.train_start,
                eval_end=config.train_end,
            )

        # Validation pass for this outer trial
        v_sc: float | None = None
        v_nt = 0
        v_wr_val = 0.0
        v_pnl_val = 0.0
        if (has_validation
                and best_inner_trainer is not None
                and best_inner_params_local):
            assert config.validation_start is not None
            assert config.validation_end is not None
            vt_config = dataclasses.replace(
                config,
                test_start=config.validation_start,
                test_end=config.validation_end,
                validation_start=None,
                validation_end=None,
            )
            vt_sim = SimConfig(
                sl_points=best_inner_params_local.get("sl_fallback", 3.0),
                tp_points=best_inner_params_local.get("tp_fallback", 3.0),
                k_sl=best_inner_params_local.get("k_sl", 1.0),
                k_tp=best_inner_params_local.get("k_tp", 1.0),
                max_spread=2.0,
                gamma=best_inner_params_local.get("gamma", 1.0),
                max_margin_proba=best_inner_params_local.get(
                    "max_margin_proba", 0.80,
                ),
                sizing_threshold=best_inner_params_local.get(
                    "entry_threshold", 0.5,
                ),
                min_risk_pct=best_inner_params_local.get(
                    "min_risk_pct", 0.005,
                ),
                slippage_min_pts=config.slippage_min_pts,
                slippage_max_pts=config.slippage_max_pts,
                slippage_seed=config.slippage_seed,
            )
            v_sc, v_nt, v_wr_val, v_pnl_val, _ = await _evaluate_oos_async(
                best_inner_trainer, vt_sim, db, vt_config,
                registry_factory, extractor_params,
            )

        # Record best-inner result for this outer trial
        result.trial_records.append(TrialRecord(
            window_idx=window_idx,
            outer_idx=outer_i,
            score=best_inner_score,
            n_trades=best_inner_trades,
            win_rate=best_inner_wr,
            pnl=best_inner_pnl,
            outer_params=dict(extractor_params),
            inner_params=dict(best_inner_params_local),
            trades=list(best_inner_trades_list),
            val_score=v_sc,
            val_n_trades=v_nt,
            val_win_rate=v_wr_val,
            val_pnl=v_pnl_val,
            train_score=t_sc,
            train_n_trades=t_nt,
            train_win_rate=t_wr_val,
            train_pnl=t_pnl_val,
        ))

        val_str = ""
        if v_sc is not None:
            val_str = (f", val_trades={v_nt}, "
                       f"val_WR={v_wr_val*100:.0f}%, "
                       f"val_PnL={v_pnl_val:+.1f}")
        print(f"  Best inner: score={best_inner_score:+.2f}, "
              f"k_sl={best_inner_params_local.get('k_sl', 0):.2f}, "
              f"k_tp={best_inner_params_local.get('k_tp', 0):.2f}, "
              f"gamma={best_inner_params_local.get('gamma', 0):.2f}, "
              f"sel_trades={best_inner_trades}, "
              f"WR={best_inner_wr*100:.0f}%, "
              f"PnL={best_inner_pnl:+.1f}{val_str}, "
              f"threshold={best_inner_params_local.get('entry_threshold', '?')}")

        if best_inner_score > best_score:
            best_score = best_inner_score
            best_outer = dict(extractor_params)
            best_inner = best_inner_params_local
            result.best_trainer = best_inner_trainer
            result.best_trades = best_inner_trades_list
            best_trades = best_inner_trades
            best_wr = best_inner_wr
            best_pnl = best_inner_pnl

    result.best_outer_params = best_outer
    result.best_inner_params = best_inner
    result.best_score = best_score
    result.total_outer_trials = config.outer_trials
    result.best_n_trades = best_trades
    result.best_win_rate = best_wr
    result.best_pnl = best_pnl

    # Pick validation results from the best trial record
    if has_validation and result.trial_records:
        best_rec = max(result.trial_records, key=lambda r: r.score)
        result.val_score = best_rec.val_score
        result.val_n_trades = best_rec.val_n_trades
        result.val_win_rate = best_rec.val_win_rate
        result.val_pnl = best_rec.val_pnl

    _print_result(result)
    return result


def default_output_prefix() -> str:
    """Generate a timestamped output prefix for trial log files.

    Returns:
        Prefix like ``config/midas_optuna_20260412_143022``.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"config/midas_optuna_{stamp}"


def write_trial_logs(
    records: list[TrialRecord],
    prefix: str,
) -> tuple[Path, Path]:
    """Write trial CSV and trades CSV from collected TrialRecords.

    Args:
        records: List of trial records (one per outer trial per window).
        prefix: Path prefix (e.g. ``config/midas_optuna_20260412_143022``).
            Files created: ``{prefix}_trials.csv``, ``{prefix}_trades.csv``.

    Returns:
        (trials_path, trades_path) — the two files written.
    """
    trials_path = Path(f"{prefix}_trials.csv")
    trades_path = Path(f"{prefix}_trades.csv")
    trials_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all param keys across records (outer + inner may vary)
    outer_keys: set[str] = set()
    inner_keys: set[str] = set()
    for r in records:
        outer_keys.update(r.outer_params.keys())
        inner_keys.update(r.inner_params.keys())
    outer_sorted = sorted(outer_keys)
    inner_sorted = sorted(inner_keys)

    # --- Trials CSV ---
    has_val = any(r.val_score is not None for r in records)
    has_train = any(r.train_score is not None for r in records)
    trial_fields = (
        ["window_idx", "outer_idx", "score", "n_trades", "win_rate", "pnl"]
        + (["train_score", "train_n_trades", "train_win_rate", "train_pnl"]
           if has_train else [])
        + (["val_score", "val_n_trades", "val_win_rate", "val_pnl"]
           if has_val else [])
        + [f"outer__{k}" for k in outer_sorted]
        + [f"inner__{k}" for k in inner_sorted]
    )
    with open(trials_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=trial_fields)
        writer.writeheader()
        for r in records:
            row: dict[str, Any] = {
                "window_idx": r.window_idx,
                "outer_idx": r.outer_idx,
                "score": r.score,
                "n_trades": r.n_trades,
                "win_rate": round(r.win_rate, 4),
                "pnl": round(r.pnl, 2),
            }
            if has_train:
                row["train_score"] = (
                    round(r.train_score, 2)
                    if r.train_score is not None else ""
                )
                row["train_n_trades"] = r.train_n_trades
                row["train_win_rate"] = round(r.train_win_rate, 4)
                row["train_pnl"] = round(r.train_pnl, 2)
            if has_val:
                row["val_score"] = (
                    round(r.val_score, 2) if r.val_score is not None else ""
                )
                row["val_n_trades"] = r.val_n_trades
                row["val_win_rate"] = round(r.val_win_rate, 4)
                row["val_pnl"] = round(r.val_pnl, 2)
            for k in outer_sorted:
                row[f"outer__{k}"] = r.outer_params.get(k, "")
            for k in inner_sorted:
                row[f"inner__{k}"] = r.inner_params.get(k, "")
            writer.writerow(row)

    # --- Trades CSV ---
    trade_fields = [
        "window_idx", "outer_idx", "trade_id", "direction",
        "entry_time", "exit_time", "entry_price", "exit_price",
        "sl_price", "tp_price", "size", "proba",
        "pnl", "pnl_points", "is_win",
    ]
    n_trades = 0
    with open(trades_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=trade_fields)
        writer.writeheader()
        for r in records:
            n_trades += len(r.trades)
            for t in r.trades:
                writer.writerow({
                    "window_idx": r.window_idx,
                    "outer_idx": r.outer_idx,
                    "trade_id": t.trade_id,
                    "direction": t.direction,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "sl_price": t.sl_price,
                    "tp_price": t.tp_price,
                    "size": t.size,
                    "proba": t.proba,
                    "pnl": round(t.pnl, 4),
                    "pnl_points": round(t.pnl_points, 4),
                    "is_win": t.is_win,
                })

    print(f"\nTrial logs: {trials_path} ({len(records)} trials)")
    print(f"Trade logs: {trades_path} ({n_trades} trades)")
    return trials_path, trades_path


def _print_result(result: OptimizationResult) -> None:
    """Print optimization results."""
    print(f"\n{'='*60}")
    print("NESTED OPTUNA RESULTS")
    print(f"{'='*60}")
    print(f"  Best score: {result.best_score:+.2f}")
    print(f"  Trades: {result.best_n_trades}, "
          f"WR: {result.best_win_rate*100:.1f}%, "
          f"PnL: {result.best_pnl:+.2f}")
    if result.val_score is not None:
        print(f"  Validation: score={result.val_score:+.2f}, "
              f"trades={result.val_n_trades}, "
              f"WR={result.val_win_rate*100:.1f}%, "
              f"PnL={result.val_pnl:+.2f}")
    print(f"  Outer trials: {result.total_outer_trials}, "
          f"Inner trials: {result.total_inner_trials}")
    print("\n  Best outer params (extractor):")
    for k, v in sorted(result.best_outer_params.items()):
        print(f"    {k}: {v}")
    print("\n  Best inner params (SL/TP + LightGBM):")
    for k, v in sorted(result.best_inner_params.items()):
        print(f"    {k}: {v}")
