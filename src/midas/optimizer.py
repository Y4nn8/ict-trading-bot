"""Nested Optuna optimizer for the Midas scalping engine.

Outer loop: tunes feature extractor params (expensive, re-replays ticks).
Inner loop: tunes SL/TP + LightGBM hyperparams (cheap, relabels in-memory).

Score = walk-forward OOS metric (composite, pnl, win_rate, pnl_per_trade).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl

from src.midas.labeler import relabel_dataframe
from src.midas.replay_engine import (
    ReplayConfig,
    ReplayEngine,
    build_default_registry,
)
from src.midas.trade_simulator import SimConfig, TradeSimulator
from src.midas.trainer import MidasTrainer, TrainerConfig

if TYPE_CHECKING:
    from src.common.db import Database
    from src.midas.types import Tick


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
        score_metric: Metric to optimize.
        sl_range: SL search range in points (fixed mode fallback).
        tp_range: TP search range in points (fixed mode fallback).
        k_sl_range: k_sl multiplier search range (ATR mode).
        k_tp_range: k_tp multiplier search range (ATR mode).
        atr_column: Column name for ATR values.
        fixed_outer_params: Fixed extractor params (skip outer search).
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
    sl_range: tuple[float, float] = (1.5, 8.0)
    tp_range: tuple[float, float] = (1.5, 8.0)
    k_sl_range: tuple[float, float] = (0.5, 3.0)
    k_tp_range: tuple[float, float] = (0.5, 3.0)
    atr_column: str = "scalp__m1_atr"
    fixed_outer_params: dict[str, Any] | None = None


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


def _suggest_outer_params(
    trial: optuna.Trial,
    registry_params: list[Any],
) -> dict[str, Any]:
    """Suggest extractor params only for the outer loop."""
    params: dict[str, Any] = {}
    for p in registry_params:
        if p.param_type == "int":
            params[p.name] = trial.suggest_int(p.name, int(p.low), int(p.high))
        else:
            params[p.name] = trial.suggest_float(p.name, p.low, p.high)
    return params


def _suggest_inner_params(
    trial: optuna.Trial,
    config: OptimizerConfig,
) -> dict[str, Any]:
    """Suggest k_sl/k_tp + LightGBM hyperparams + entry threshold."""
    return {
        # ATR-based SL/TP multipliers (relabeled per inner trial)
        "k_sl": trial.suggest_float(
            "k_sl", config.k_sl_range[0], config.k_sl_range[1],
        ),
        "k_tp": trial.suggest_float(
            "k_tp", config.k_tp_range[0], config.k_tp_range[1],
        ),
        "sl_fallback": trial.suggest_float(
            "sl_fallback", config.sl_range[0], config.sl_range[1],
        ),
        "label_timeout": trial.suggest_float(
            "label_timeout", 60.0, 600.0,
        ),
        # LightGBM hyperparams
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.3, log=True,
        ),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.5, 1.0,
        ),
        "entry_threshold": trial.suggest_float(
            "entry_threshold", 0.25, 0.60,
        ),
    }


async def _evaluate_oos_async(
    trainer: MidasTrainer,
    sim_config: SimConfig,
    db: Any,
    config: OptimizerConfig,
    registry_factory: Any,
    extractor_params: dict[str, Any],
) -> tuple[float, int, float, float]:
    """Run OOS test and return (score, n_trades, win_rate, total_pnl)."""
    simulator = TradeSimulator(sim_config)
    _trainer = trainer
    _simulator = simulator
    _atr_col = config.atr_column

    def callback(
        tick: Tick,
        features: dict[str, float],
        *,
        tr: MidasTrainer = _trainer,
        sim: TradeSimulator = _simulator,
        atr_col: str = _atr_col,
    ) -> None:
        signal, _ = tr.predict(features)
        sim.on_signal(tick, signal, atr=features.get(atr_col, 0.0))

    last_tick_holder: list[Tick | None] = [None]

    def exit_hook(
        tick: Tick,
        *,
        sim: TradeSimulator = _simulator,
        holder: list[Tick | None] = last_tick_holder,
    ) -> None:
        sim.on_tick(tick)
        holder[0] = tick

    registry = registry_factory()
    registry.configure_all(extractor_params)

    engine = ReplayEngine(
        db, registry,
        ReplayConfig(
            instrument=config.instrument,
            start=config.test_start,
            end=config.test_end,
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
        return -1000.0, 0, 0.0, 0.0

    total_pnl = sum(t.pnl for t in trades)
    wins = sum(1 for t in trades if t.is_win)
    win_rate = wins / n_trades

    if config.score_metric == "composite":
        if n_trades < 10:
            return -1000.0, n_trades, win_rate, total_pnl
        score = total_pnl * (win_rate**0.5) * (n_trades**0.5)
    elif config.score_metric == "win_rate":
        score = win_rate
    elif config.score_metric == "pnl_per_trade":
        score = total_pnl / n_trades
    else:
        score = total_pnl

    return score, n_trades, win_rate, total_pnl


async def run_nested_optuna(
    config: OptimizerConfig,
    db: Database,
) -> OptimizationResult:
    """Run the nested Optuna optimization.

    Outer loop: extractor params → replay ticks → features DataFrame.
    Inner loop: SL/TP → relabel DataFrame → train LightGBM → evaluate OOS.
    """
    result = OptimizationResult()

    def registry_factory() -> Any:
        return build_default_registry(instrument=config.instrument)

    sample_registry = registry_factory()
    registry_params = sample_registry.all_tunable_params()

    print(f"\nNested Optuna: {config.outer_trials} outer x "
          f"{config.inner_trials} inner trials")
    print(f"  Outer: {len(registry_params)} extractor params")
    print("  Inner: k_sl/k_tp/sl_fallback/timeout + 8 LightGBM params = 12 params")
    print(f"  ATR column: {config.atr_column}")
    print(f"  Train: {config.train_start.date()} → {config.train_end.date()}")
    print(f"  Test:  {config.test_start.date()} → {config.test_end.date()}")

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
        best_inner_params_local: dict[str, Any] = {}
        best_inner_k_sl = 0.0
        best_inner_k_tp = 0.0
        best_inner_sl_fb = 0.0

        for _inner_i in range(config.inner_trials):
            inner_trial = inner_study.ask()
            inner_params = _suggest_inner_params(inner_trial, config)

            k_sl = inner_params["k_sl"]
            k_tp = inner_params["k_tp"]
            sl_fallback = inner_params["sl_fallback"]
            timeout = inner_params["label_timeout"]

            # Relabel in-memory with ATR-based SL/TP (fast, no DB)
            label_result = relabel_dataframe(
                df,
                sl_points=sl_fallback,
                tp_points=sl_fallback,  # symmetric fallback
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

            # Train
            trainer_config = TrainerConfig(
                n_estimators=inner_params["n_estimators"],
                learning_rate=inner_params["learning_rate"],
                max_depth=inner_params["max_depth"],
                num_leaves=inner_params["num_leaves"],
                min_child_samples=inner_params["min_child_samples"],
                subsample=inner_params["subsample"],
                colsample_bytree=inner_params["colsample_bytree"],
                entry_threshold=inner_params["entry_threshold"],
                early_stopping_rounds=30,
            )

            trainer = MidasTrainer(trainer_config)
            try:
                trainer.train(df_filtered, target_filtered, sample_weights=weights)
            except (ValueError, lgb.basic.LightGBMError):
                inner_study.tell(inner_trial, -1000.0)
                continue

            # Evaluate on OOS with ATR-based SL/TP
            sim_config = SimConfig(
                sl_points=sl_fallback,
                tp_points=sl_fallback,
                k_sl=k_sl,
                k_tp=k_tp,
                max_spread=2.0,
            )
            score, n_tr, wr, pnl = await _evaluate_oos_async(
                trainer, sim_config, db, config,
                registry_factory, extractor_params,
            )

            inner_study.tell(inner_trial, score)

            if score > best_inner_score:
                best_inner_score = score
                best_inner_trainer = trainer
                best_inner_params_local = dict(inner_params)
                best_inner_trades = n_tr
                best_inner_wr = wr
                best_inner_pnl = pnl
                best_inner_k_sl = k_sl
                best_inner_k_tp = k_tp
                best_inner_sl_fb = sl_fallback

        # Report to outer
        outer_study.tell(outer_trial, best_inner_score)
        result.total_inner_trials += config.inner_trials

        print(f"  Best inner: score={best_inner_score:+.2f}, "
              f"k_sl={best_inner_k_sl:.2f}, k_tp={best_inner_k_tp:.2f}, "
              f"trades={best_inner_trades}, "
              f"WR={best_inner_wr*100:.0f}%, "
              f"PnL={best_inner_pnl:+.1f}, "
              f"threshold={best_inner_params_local.get('entry_threshold', '?')}")

        if best_inner_score > best_score:
            best_score = best_inner_score
            best_outer = dict(extractor_params)
            best_inner = best_inner_params_local
            result.best_trainer = best_inner_trainer

            if best_inner_trainer is not None:
                _, best_trades, best_wr, best_pnl = await _evaluate_oos_async(
                    best_inner_trainer,
                    SimConfig(
                        sl_points=best_inner_sl_fb,
                        tp_points=best_inner_sl_fb,
                        k_sl=best_inner_k_sl,
                        k_tp=best_inner_k_tp,
                        max_spread=2.0,
                    ),
                    db, config, registry_factory, extractor_params,
                )

    result.best_outer_params = best_outer
    result.best_inner_params = best_inner
    result.best_score = best_score
    result.total_outer_trials = config.outer_trials
    result.best_n_trades = best_trades
    result.best_win_rate = best_wr
    result.best_pnl = best_pnl

    _print_result(result)
    return result


def _print_result(result: OptimizationResult) -> None:
    """Print optimization results."""
    print(f"\n{'='*60}")
    print("NESTED OPTUNA RESULTS")
    print(f"{'='*60}")
    print(f"  Best score: {result.best_score:+.2f}")
    print(f"  Trades: {result.best_n_trades}, "
          f"WR: {result.best_win_rate*100:.1f}%, "
          f"PnL: {result.best_pnl:+.2f}")
    print(f"  Outer trials: {result.total_outer_trials}, "
          f"Inner trials: {result.total_inner_trials}")
    print("\n  Best outer params (extractor):")
    for k, v in sorted(result.best_outer_params.items()):
        print(f"    {k}: {v}")
    print("\n  Best inner params (SL/TP + LightGBM):")
    for k, v in sorted(result.best_inner_params.items()):
        print(f"    {k}: {v}")
