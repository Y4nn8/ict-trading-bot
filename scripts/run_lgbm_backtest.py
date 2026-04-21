"""LightGBM scalping backtest on M1 candles with walk-forward validation.

Two strategy modes:
  - Mean Reversion (MR): enter against deviation from SMA
  - Trend Following (TF): enter with momentum direction

Three trading presets (all fixed, no Optuna):
  A - Conservative: threshold=0.70, SL=15, TP=5
  B - Balanced:     threshold=0.60, SL=10, TP=10
  C - Aggressive:   threshold=0.55, SL=5,  TP=15

Walk-forward: train LightGBM on N days, test on 1 day, step 1 day.
LightGBM hyperparams are fixed (shallow trees, high regularization).
Features computed with pandas-ta.

Usage:
    uv run python -m scripts.run_lgbm_backtest \\
        --instrument XAUUSD --start 2026-01-07 --end 2026-04-07 \\
        --train-days 30 --test-days 1 --output config/lgbm_wf
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import math
import random
import time as time_mod
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv

from src.common.config import load_config
from src.common.db import Database
from src.common.logging import setup_logging
from src.midas.candle_builder import CandleBuilder
from src.midas.types import Tick


# ──────────────────────────── Constants ────────────────────────────────────

LGB_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "n_estimators": 150,
    "max_depth": 4,
    "num_leaves": 15,
    "learning_rate": 0.05,
    "min_child_samples": 200,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
}

FEATURE_NAMES = [
    "rsi_14", "zscore_60", "boll_pctb_20", "roc_6",
    "atr_30", "atr_ratio", "wick_ratio", "body_ratio",
    "consecutive", "hour_sin", "hour_cos", "adx_14",
]

WARMUP = 60
LABEL_HORIZON = 30  # 30 M1 candles = 30 min


@dataclass(frozen=True, slots=True)
class Preset:
    name: str
    threshold: float
    sl_points: float
    tp_points: float


PRESETS = [
    Preset("A_highWR", 0.60, 7.0, 4.0),
    Preset("B_balanced", 0.60, 5.0, 5.0),
    Preset("C_highRR", 0.60, 4.0, 7.0),
    Preset("A_flip", -0.60, 7.0, 4.0),
    Preset("B_flip", -0.60, 5.0, 5.0),
    Preset("C_flip", -0.60, 4.0, 7.0),
]


@dataclass(frozen=True, slots=True)
class Candle:
    time: datetime
    open: float
    high: float
    low: float
    close: float
    close_bid: float
    close_ask: float


@dataclass(frozen=True, slots=True)
class SimResult:
    n_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    final_capital: float
    return_pct: float


# ──────────────────────── Data Loading ─────────────────────────────────────


async def load_m1_candles(
    db: Database,
    instrument: str,
    start: datetime,
    end: datetime,
    chunk_size: int = 100_000,
    log: logging.Logger | None = None,
) -> list[Candle]:
    """Stream ticks from DB, aggregate to M1 candles."""
    builder = CandleBuilder(bucket_seconds=60)
    candles: list[Candle] = []
    n_ticks = 0
    t0 = time_mod.monotonic()
    log_every = 10_000_000

    async with db.pool.acquire() as conn, conn.transaction():
        stmt = await conn.prepare(
            "SELECT time, bid, ask FROM ticks "
            "WHERE instrument = $1 AND time >= $2 AND time < $3 "
            "ORDER BY time ASC",
        )
        cursor = await stmt.cursor(instrument, start, end)

        while True:
            rows = await cursor.fetch(chunk_size)
            if not rows:
                break

            for row in rows:
                tick = Tick(
                    time=row["time"],
                    bid=float(row["bid"]),
                    ask=float(row["ask"]),
                )
                n_ticks += 1
                closed = builder.process_tick(tick)
                if closed is not None:
                    candles.append(Candle(
                        time=closed["time"],
                        open=float(closed["open"]),
                        high=float(closed["high"]),
                        low=float(closed["low"]),
                        close=float(closed["close"]),
                        close_bid=float(closed["bid"]),
                        close_ask=float(closed["ask"]),
                    ))

                if n_ticks % log_every == 0 and log is not None:
                    elapsed = time_mod.monotonic() - t0
                    rate = n_ticks / elapsed if elapsed > 0 else 0
                    log.info(
                        "load_progress ticks=%s candles=%s "
                        "elapsed=%.1fs rate=%.0f ticks/s",
                        f"{n_ticks:,}", f"{len(candles):,}",
                        elapsed, rate,
                    )

    flushed = builder.flush()
    if flushed is not None:
        candles.append(Candle(
            time=flushed["time"],
            open=float(flushed["open"]),
            high=float(flushed["high"]),
            low=float(flushed["low"]),
            close=float(flushed["close"]),
            close_bid=float(flushed["bid"]),
            close_ask=float(flushed["ask"]),
        ))

    if log is not None:
        elapsed = time_mod.monotonic() - t0
        log.info(
            "load_complete ticks=%s candles=%s elapsed=%.1fs",
            f"{n_ticks:,}", f"{len(candles):,}", elapsed,
        )
    return candles


# ──────────────────────── Feature Computation (pandas-ta) ─────────────────


def compute_features(
    candles: list[Candle],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute all features using pandas-ta.

    Returns:
        (feature_matrix [n, 12], sma60 array, roc6 array).
    """
    df = pd.DataFrame({
        "open": [c.open for c in candles],
        "high": [c.high for c in candles],
        "low": [c.low for c in candles],
        "close": [c.close for c in candles],
    })

    # 1. RSI(14)
    df["rsi_14"] = ta.rsi(df["close"], length=14)

    # 2. Z-score(close, 60)
    sma60 = ta.sma(df["close"], length=60)
    std60 = ta.stdev(df["close"], length=60)
    df["zscore_60"] = (df["close"] - sma60) / std60.replace(0, np.nan)

    # 3. Bollinger %B(20, 2)
    bbands = ta.bbands(df["close"], length=20, std=2)
    df["boll_pctb_20"] = bbands.iloc[:, -1]  # BBP column (percent B)

    # 4. ROC(6)
    df["roc_6"] = ta.roc(df["close"], length=6)

    # 5. ATR(30)
    df["atr_30"] = ta.atr(df["high"], df["low"], df["close"], length=30)

    # 6. ATR ratio = ATR(6) / ATR(30)
    atr6 = ta.atr(df["high"], df["low"], df["close"], length=6)
    df["atr_ratio"] = atr6 / df["atr_30"].replace(0, np.nan)

    # 7. Wick ratio
    ranges = df["high"] - df["low"]
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    df["wick_ratio"] = ((upper_wick + lower_wick) / ranges.replace(0, np.nan)).fillna(0)

    # 8. Body ratio
    body = (df["close"] - df["open"]).abs()
    df["body_ratio"] = (body / ranges.replace(0, np.nan)).fillna(0)

    # 9. Consecutive candles (same direction)
    direction = np.sign(df["close"].values - df["open"].values)
    consec = np.ones(len(df))
    for i in range(1, len(df)):
        if direction[i] == direction[i - 1] and direction[i] != 0:
            consec[i] = consec[i - 1] + 1
    df["consecutive"] = consec

    # 10-11. Hour sin/cos
    hours = np.array([
        c.time.hour + c.time.minute / 60.0 for c in candles
    ])
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    # 12. ADX(14)
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["adx_14"] = adx_df.iloc[:, 0]  # ADX column

    features = df[FEATURE_NAMES].values.astype(np.float64)
    return features, sma60.values.astype(np.float64), df["roc_6"].values.astype(np.float64)


# ──────────────────────── Direction & Labels ──────────────────────────────


def compute_directions(
    closes: np.ndarray,
    sma60: np.ndarray,
    roc6: np.ndarray,
    strategy: str,
) -> np.ndarray:
    """Compute trade direction: +1 LONG, -1 SHORT, 0 no trade."""
    n = len(closes)
    dirs = np.zeros(n, dtype=np.int8)

    if strategy == "mr":
        deviation = closes - sma60
        for i in range(n):
            if np.isnan(sma60[i]):
                continue
            if deviation[i] > 0.01:
                dirs[i] = -1  # SHORT: expect reversion down
            elif deviation[i] < -0.01:
                dirs[i] = 1   # LONG: expect reversion up
    elif strategy == "tf":
        for i in range(n):
            if np.isnan(roc6[i]):
                continue
            if roc6[i] > 0.001:
                dirs[i] = 1   # LONG: with momentum
            elif roc6[i] < -0.001:
                dirs[i] = -1  # SHORT: with momentum

    return dirs


def compute_labels(
    candles: list[Candle],
    directions: np.ndarray,
    sl_points: float,
    tp_points: float,
    horizon: int = LABEL_HORIZON,
) -> np.ndarray:
    """Compute binary labels: 1 if TP hit before SL within horizon candles."""
    n = len(candles)
    labels = np.full(n, np.nan)

    for i in range(n - horizon - 1):
        d = directions[i]
        if d == 0:
            continue
        entry = candles[i + 1].open
        hit = False
        for j in range(i + 1, min(i + 1 + horizon, n)):
            if d == 1:  # LONG
                if candles[j].low <= entry - sl_points:
                    labels[i] = 0
                    hit = True
                    break
                if candles[j].high >= entry + tp_points:
                    labels[i] = 1
                    hit = True
                    break
            else:  # SHORT
                if candles[j].high >= entry + sl_points:
                    labels[i] = 0
                    hit = True
                    break
                if candles[j].low <= entry - tp_points:
                    labels[i] = 1
                    hit = True
                    break
        if not hit and d != 0:
            labels[i] = 0

    return labels


# ──────────────────────── Trading Simulation ──────────────────────────────


def simulate_trades(
    candles: list[Candle],
    probabilities: np.ndarray,
    directions: np.ndarray,
    threshold: float,
    sl_points: float,
    tp_points: float,
    initial_capital: float = 5_000.0,
    max_spread: float = 2.0,
    margin_pct_per_trade: float = 0.5,
    broker_margin_pct: float = 0.05,
    min_lot_size: float = 0.1,
    value_per_point: float = 1.0,
    slippage_max: float = 0.5,
    slippage_seed: int = 42,
) -> SimResult:
    """Simulate trades using model predictions and fixed SL/TP."""
    rng = random.Random(slippage_seed)
    capital = initial_capital
    in_pos = False
    pos_dir = 0
    pos_entry = 0.0
    pos_sl = 0.0
    pos_tp = 0.0
    pos_size = 0.0
    pnls: list[float] = []

    def _close_at(exit_price: float) -> None:
        nonlocal in_pos, pos_dir, capital
        pts = (exit_price - pos_entry) if pos_dir == 1 else (pos_entry - exit_price)
        pnl = pts * pos_size * value_per_point
        capital += pnl
        pnls.append(pnl)
        in_pos = False
        pos_dir = 0

    for i in range(len(candles) - 1):
        c = candles[i]

        # Check exits
        if in_pos:
            if pos_dir == 1:
                if c.low <= pos_sl:
                    _close_at(pos_sl)
                elif c.high >= pos_tp:
                    _close_at(pos_tp)
            else:
                if c.high >= pos_sl:
                    _close_at(pos_sl)
                elif c.low <= pos_tp:
                    _close_at(pos_tp)

        # Entry check
        if not in_pos and directions[i] != 0:
            prob = probabilities[i]
            if np.isnan(prob):
                continue
            inverted = threshold < 0
            actual_threshold = abs(threshold)
            trigger = prob >= actual_threshold
            if trigger:
                nc = candles[i + 1]
                spread = nc.close_ask - nc.close_bid
                if spread > max_spread:
                    continue

                margin_per_lot = nc.open * broker_margin_pct
                if margin_per_lot <= 0:
                    continue
                raw = margin_pct_per_trade * capital / margin_per_lot
                size = math.floor(raw / min_lot_size) * min_lot_size
                if size < min_lot_size:
                    continue

                half_spread = spread / 2.0
                slip = rng.uniform(0, slippage_max)
                d = -directions[i] if inverted else directions[i]

                if d == 1:
                    entry = nc.open + half_spread + slip
                    pos_sl = entry - sl_points
                    pos_tp = entry + tp_points
                else:
                    entry = nc.open - half_spread - slip
                    pos_sl = entry + sl_points
                    pos_tp = entry - tp_points

                pos_entry = entry
                pos_dir = d
                pos_size = size
                in_pos = True

    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    total = sum(pnls)
    equity = initial_capital
    peak = initial_capital
    max_dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    return SimResult(
        n_trades=n,
        wins=wins,
        losses=n - wins,
        win_rate=wins / n if n > 0 else 0.0,
        total_pnl=total,
        max_drawdown=max_dd,
        final_capital=initial_capital + total,
        return_pct=total / initial_capital * 100 if initial_capital > 0 else 0.0,
    )


# ──────────────────────── Walk-Forward ────────────────────────────────────


def _slice_indices(
    candles: list[Candle], start: datetime, end: datetime,
) -> tuple[int, int]:
    """Return (start_idx, end_idx) for candles in [start, end)."""
    lo = 0
    hi = len(candles)
    for i, c in enumerate(candles):
        if c.time >= start and lo == 0 and (i == 0 or candles[i - 1].time < start):
            lo = i
        if c.time >= end:
            hi = i
            break
    return lo, hi


def run_walk_forward(
    candles: list[Candle],
    features: np.ndarray,
    sma60: np.ndarray,
    roc6: np.ndarray,
    *,
    full_start: datetime,
    full_end: datetime,
    train_days: int,
    test_days: int,
    step_days: int,
    initial_capital: float,
    value_per_point: float,
    margin_pct_per_trade: float,
    broker_margin_pct: float,
    min_lot_size: float,
    max_spread: float,
    slippage_max: float,
    output_prefix: Path | None,
    log: logging.Logger,
) -> list[dict[str, Any]]:
    """Walk-forward: train LightGBM on train, predict on test, step forward."""
    all_records: list[dict[str, Any]] = []
    window_idx = 0
    train_start = full_start

    strategies = ["mr", "tf"]

    while True:
        train_end = train_start + timedelta(days=train_days)
        test_end = train_end + timedelta(days=test_days)
        if test_end > full_end:
            break

        train_lo, train_hi = _slice_indices(candles, train_start, train_end)
        test_lo, test_hi = _slice_indices(candles, train_end, test_end)

        train_candles = candles[train_lo:train_hi]
        test_candles = candles[test_lo:test_hi]
        train_feat = features[train_lo:train_hi]
        test_feat = features[test_lo:test_hi]
        train_sma = sma60[train_lo:train_hi]
        test_sma = sma60[test_lo:test_hi]
        train_roc = roc6[train_lo:train_hi]
        test_roc = roc6[test_lo:test_hi]
        train_closes = np.array([c.close for c in train_candles])
        test_closes = np.array([c.close for c in test_candles])

        if len(train_candles) < WARMUP * 2 or len(test_candles) < 10:
            train_start += timedelta(days=step_days)
            window_idx += 1
            continue

        log.info(
            "──── W%d: train %s→%s (%d)  test %s→%s (%d) ────",
            window_idx,
            train_start.date(), train_end.date(), len(train_candles),
            train_end.date(), test_end.date(), len(test_candles),
        )

        for strategy in strategies:
            train_dirs = compute_directions(train_closes, train_sma, train_roc, strategy)
            test_dirs = compute_directions(test_closes, test_sma, test_roc, strategy)

            for preset in PRESETS:
                labels = compute_labels(
                    train_candles, train_dirs,
                    sl_points=preset.sl_points,
                    tp_points=preset.tp_points,
                )

                valid = (
                    ~np.isnan(labels)
                    & ~np.any(np.isnan(train_feat), axis=1)
                )
                X_train = train_feat[valid]
                y_train = labels[valid]

                if len(X_train) < 100 or y_train.sum() < 10 or (1 - y_train).sum() < 10:
                    log.info(
                        "  %s/%s: skipped (insufficient: %d valid, %.0f pos)",
                        strategy.upper(), preset.name,
                        len(X_train), y_train.sum() if len(y_train) > 0 else 0,
                    )
                    train_start += timedelta(days=step_days)
                    continue

                model = lgb.LGBMClassifier(**LGB_PARAMS)
                model.fit(X_train, y_train)

                valid_test = ~np.any(np.isnan(test_feat), axis=1)
                probas = np.full(len(test_candles), np.nan)
                if valid_test.sum() > 0:
                    probas[valid_test] = model.predict_proba(test_feat[valid_test])[:, 1]

                result = simulate_trades(
                    test_candles, probas, test_dirs,
                    threshold=preset.threshold,
                    sl_points=preset.sl_points,
                    tp_points=preset.tp_points,
                    initial_capital=initial_capital,
                    max_spread=max_spread,
                    margin_pct_per_trade=margin_pct_per_trade,
                    broker_margin_pct=broker_margin_pct,
                    min_lot_size=min_lot_size,
                    value_per_point=value_per_point,
                    slippage_max=slippage_max,
                )

                train_wr = y_train.mean() if len(y_train) > 0 else 0.0
                log.info(
                    "  %s/%s: train=%d pos_rate=%.1f%%  "
                    "OOS n=%d wr=%.1f%% pnl=%+.2f mdd=%.2f",
                    strategy.upper(), preset.name,
                    len(X_train), train_wr * 100,
                    result.n_trades, result.win_rate * 100,
                    result.total_pnl, result.max_drawdown,
                )

                all_records.append({
                    "window_idx": window_idx,
                    "train_start": train_start.date().isoformat(),
                    "train_end": train_end.date().isoformat(),
                    "test_start": train_end.date().isoformat(),
                    "test_end": test_end.date().isoformat(),
                    "strategy": strategy,
                    "preset": preset.name,
                    "train_samples": len(X_train),
                    "train_pos_rate": float(train_wr),
                    "oos_n_trades": result.n_trades,
                    "oos_wr": result.win_rate,
                    "oos_pnl": result.total_pnl,
                    "oos_return_pct": result.return_pct,
                    "oos_mdd": result.max_drawdown,
                })

        train_start += timedelta(days=step_days)
        window_idx += 1

    # Write CSV
    if output_prefix is not None and all_records:
        csv_path = Path(str(output_prefix) + "_wf_results.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
            w.writeheader()
            w.writerows(all_records)
        log.info("wrote %s", csv_path)

    return all_records


# ──────────────────────── CLI & Main ──────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LightGBM scalping WF backtest")
    p.add_argument("--instrument", type=str, default="XAUUSD")
    p.add_argument(
        "--start", type=lambda s: datetime.fromisoformat(s).replace(tzinfo=UTC),
        required=True,
    )
    p.add_argument(
        "--end", type=lambda s: datetime.fromisoformat(s).replace(tzinfo=UTC),
        required=True,
    )
    p.add_argument("--train-days", type=int, default=30)
    p.add_argument("--test-days", type=int, default=1)
    p.add_argument("--step-days", type=int, default=1)
    p.add_argument("--initial-capital", type=float, default=5_000.0)
    p.add_argument("--value-per-point", type=float, default=0.10)
    p.add_argument("--margin-pct-per-trade", type=float, default=0.5)
    p.add_argument("--broker-margin-pct", type=float, default=0.05)
    p.add_argument("--min-lot-size", type=float, default=0.1)
    p.add_argument("--max-spread", type=float, default=2.0)
    p.add_argument("--slippage-max", type=float, default=0.5)
    p.add_argument("--output", type=str, default=None)
    return p


async def main(args: argparse.Namespace) -> None:
    config = load_config()
    setup_logging(config.logging.level, json_format=False)
    log = logging.getLogger("lgbm_backtest")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(h)
        log.propagate = False

    db = Database(config.database)
    await db.connect()

    try:
        log.info(
            "instrument=%s start=%s end=%s train=%dd test=%dd step=%dd",
            args.instrument, args.start.date(), args.end.date(),
            args.train_days, args.test_days, args.step_days,
        )

        log.info("loading M1 candles from ticks …")
        candles = await load_m1_candles(
            db, args.instrument, args.start, args.end, log=log,
        )
        if not candles:
            log.error("no candles produced, aborting")
            return

        log.info("computing features (pandas-ta) on %d candles …", len(candles))
        features, sma60, roc6 = compute_features(candles)

        output_prefix = Path(args.output) if args.output else None
        records = run_walk_forward(
            candles, features, sma60, roc6,
            full_start=args.start,
            full_end=args.end,
            train_days=args.train_days,
            test_days=args.test_days,
            step_days=args.step_days,
            initial_capital=args.initial_capital,
            value_per_point=args.value_per_point,
            margin_pct_per_trade=args.margin_pct_per_trade,
            broker_margin_pct=args.broker_margin_pct,
            min_lot_size=args.min_lot_size,
            max_spread=args.max_spread,
            slippage_max=args.slippage_max,
            output_prefix=output_prefix,
            log=log,
        )

        # Summary per (strategy, preset)
        print("\n══════════ WALK-FORWARD SUMMARY ══════════")
        for strategy in ["mr", "tf"]:
            for preset in PRESETS:
                subset = [
                    r for r in records
                    if r["strategy"] == strategy and r["preset"] == preset.name
                ]
                if not subset:
                    continue
                total_pnl = sum(r["oos_pnl"] for r in subset)
                total_trades = sum(r["oos_n_trades"] for r in subset)
                pos_windows = sum(1 for r in subset if r["oos_pnl"] > 0)
                mean_wr = (
                    np.mean([r["oos_wr"] for r in subset if r["oos_n_trades"] > 0])
                    if any(r["oos_n_trades"] > 0 for r in subset) else 0.0
                )
                print(
                    f"  {strategy.upper():2s} / {preset.name:15s}  "
                    f"windows={len(subset):3d}  "
                    f"pos={pos_windows:2d}/{len(subset):2d}  "
                    f"trades={total_trades:5d}  "
                    f"wr={mean_wr:5.1f}%  "
                    f"pnl={total_pnl:+9.2f} €"
                )
        print()

    finally:
        await db.disconnect()


if __name__ == "__main__":
    load_dotenv()
    parser = build_parser()
    asyncio.run(main(parser.parse_args()))
