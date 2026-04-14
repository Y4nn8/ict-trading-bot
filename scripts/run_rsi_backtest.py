"""Standalone RSI scalping backtest on M1 candles (tick replay).

Ports the Pine Script RSI crossover strategy:
  - LONG on crossover(RSI, oversold)
  - SHORT on crossunder(RSI, overbought)
  - Fixed SL/TP in price points
  - Reversal: opposite signal closes current position then opens new one

Uses Wilder RSI (matches Pine's ta.rsi default). Streams ticks from the
ticks hypertable via an asyncpg cursor, aggregates into M1 candles with
the existing CandleBuilder.

Two modes:
  - Tick-level single backtest (default, ``--optuna-trials 0``): full
    tick-by-tick replay via the Midas TradeSimulator — accurate but slow
    (~5 min per year of XAUUSD ticks).
  - Fast Optuna search (``--optuna-trials N > 0``): caches M1 candles
    in RAM once, each trial runs a candle-level simulation
    (SL/TP via H/L) — seconds per trial. Top-K validated in tick-mode.

Usage:
    # Single tick-level backtest
    uv run python -m scripts.run_rsi_backtest \\
        --instrument XAUUSD --start 2025-04-07 --end 2026-04-07 \\
        --rsi-length 14 --overbought 74 --oversold 24 \\
        --sl-points 7 --tp-points 4

    # Optuna search (fast candle-level)
    uv run python -m scripts.run_rsi_backtest \\
        --instrument XAUUSD --start 2025-04-07 --end 2026-04-07 \\
        --optuna-trials 100 --output config/rsi_optuna
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import optuna
from dotenv import load_dotenv

from src.common.config import load_config
from src.common.db import Database
from src.common.logging import setup_logging
from src.midas.candle_builder import CandleBuilder
from src.midas.trade_simulator import MidasTrade, SimConfig, TradeSimulator
from src.midas.types import Tick

# ─────────────────────────────── RSI ──────────────────────────────────────────


class WilderRSI:
    """Incremental Wilder RSI (matches Pine ta.rsi)."""

    def __init__(self, length: int) -> None:
        if length < 2:
            msg = "RSI length must be >= 2"
            raise ValueError(msg)
        self._length = length
        self._prev_close: float | None = None
        self._avg_gain: float | None = None
        self._avg_loss: float | None = None
        self._warmup_gains: list[float] = []
        self._warmup_losses: list[float] = []

    def update(self, close: float) -> float | None:
        if self._prev_close is None:
            self._prev_close = close
            return None

        diff = close - self._prev_close
        self._prev_close = close
        gain = diff if diff > 0 else 0.0
        loss = -diff if diff < 0 else 0.0

        if self._avg_gain is None:
            self._warmup_gains.append(gain)
            self._warmup_losses.append(loss)
            if len(self._warmup_gains) < self._length:
                return None
            self._avg_gain = sum(self._warmup_gains) / self._length
            self._avg_loss = sum(self._warmup_losses) / self._length
        else:
            assert self._avg_loss is not None
            n = self._length
            self._avg_gain = (self._avg_gain * (n - 1) + gain) / n
            self._avg_loss = (self._avg_loss * (n - 1) + loss) / n

        if self._avg_loss == 0.0:
            return 100.0
        rs = self._avg_gain / self._avg_loss
        return 100.0 - 100.0 / (1.0 + rs)


@dataclass
class _RsiState:
    prev_rsi: float | None = None
    pending_signal: int = 0  # 0=none, 1=BUY, 2=SELL


def detect_signal(
    state: _RsiState,
    rsi: float | None,
    overbought: float,
    oversold: float,
) -> int:
    signal = 0
    if state.prev_rsi is not None and rsi is not None:
        if state.prev_rsi <= oversold < rsi:
            signal = 1
        elif state.prev_rsi >= overbought > rsi:
            signal = 2
    state.prev_rsi = rsi
    return signal


# ─────────────────────────── M1 candle cache ─────────────────────────────────


@dataclass(frozen=True, slots=True)
class M1Candle:
    """M1 candle with mid OHLC + last-tick bid/ask at close."""

    time: datetime
    open: float
    high: float
    low: float
    close: float
    close_bid: float
    close_ask: float


async def load_m1_candles(
    db: Database,
    instrument: str,
    start: datetime,
    end: datetime,
    chunk_size: int = 100_000,
    log: logging.Logger | None = None,
) -> list[M1Candle]:
    """Stream ticks from DB, aggregate to M1, return list in memory."""
    builder = CandleBuilder(bucket_seconds=60)
    candles: list[M1Candle] = []
    n_ticks = 0
    t0 = time.monotonic()
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
                    candles.append(M1Candle(
                        time=closed["time"],
                        open=float(closed["open"]),
                        high=float(closed["high"]),
                        low=float(closed["low"]),
                        close=float(closed["close"]),
                        close_bid=float(closed["bid"]),
                        close_ask=float(closed["ask"]),
                    ))

                if n_ticks % log_every == 0 and log is not None:
                    elapsed = time.monotonic() - t0
                    rate = n_ticks / elapsed if elapsed > 0 else 0
                    log.info(
                        "load_progress ticks=%s candles=%s "
                        "elapsed=%.1fs rate=%.0f ticks/s",
                        f"{n_ticks:,}", f"{len(candles):,}",
                        elapsed, rate,
                    )

    flushed = builder.flush()
    if flushed is not None:
        candles.append(M1Candle(
            time=flushed["time"],
            open=float(flushed["open"]),
            high=float(flushed["high"]),
            low=float(flushed["low"]),
            close=float(flushed["close"]),
            close_bid=float(flushed["bid"]),
            close_ask=float(flushed["ask"]),
        ))

    if log is not None:
        elapsed = time.monotonic() - t0
        log.info(
            "load_complete ticks=%s candles=%s elapsed=%.1fs",
            f"{n_ticks:,}", f"{len(candles):,}", elapsed,
        )
    return candles


# ────────────────────── Candle-level fast backtest ───────────────────────────


@dataclass
class FastBacktestResult:
    """Summary of a candle-level RSI backtest."""

    n_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    total_pnl_pts: float
    max_drawdown: float
    final_capital: float
    return_pct: float


def simulate_fast(
    candles: list[M1Candle],
    *,
    rsi_length: int,
    overbought: float,
    oversold: float,
    sl_points: float,
    tp_points: float,
    slippage_min: float = 0.0,
    slippage_max: float = 0.0,
    slippage_seed: int = 42,
    initial_capital: float = 5_000.0,
    size: float = 0.1,
    value_per_point: float = 1.0,
    max_spread: float = 2.0,
    margin_pct_per_trade: float | None = None,
    broker_margin_pct: float = 0.05,
    min_lot_size: float = 0.1,
) -> FastBacktestResult:
    """Candle-level RSI backtest using OHLC for SL/TP resolution.

    Entry at NEXT candle's open (with spread + slippage).
    Exit via H/L of subsequent candles (pessimistic: SL before TP).

    Sizing:
      - ``margin_pct_per_trade = None``: flat ``size`` lots per trade.
      - ``margin_pct_per_trade = 0.5``: use 50% of current capital as
        margin (size = floor(0.5 * capital / margin_per_lot / min_lot)
        * min_lot). Capital compounds as trades close.
    """
    import math

    rng = random.Random(slippage_seed)
    rsi = WilderRSI(rsi_length)
    state = _RsiState()

    def _sample_slip() -> float:
        if slippage_max <= 0.0:
            return 0.0
        lo = max(min(slippage_min, slippage_max), 0.0)
        return rng.uniform(lo, slippage_max)

    capital = initial_capital
    in_pos = False
    pos_dir = 0  # 1=BUY, 2=SELL
    pos_entry = 0.0
    pos_sl = 0.0
    pos_tp = 0.0
    pos_size = 0.0
    pnls: list[float] = []
    pts_sum = 0.0

    def _compute_size(price: float) -> float:
        if margin_pct_per_trade is None:
            return size
        margin_per_lot = price * broker_margin_pct
        if margin_per_lot <= 0 or capital <= 0:
            return 0.0
        eps = 1e-9
        target_margin = margin_pct_per_trade * capital
        lot_steps = target_margin / margin_per_lot / min_lot_size
        return math.floor(lot_steps + eps) * min_lot_size

    def _close_at(exit_price: float) -> None:
        nonlocal in_pos, pos_dir, capital, pts_sum
        pts = (exit_price - pos_entry) if pos_dir == 1 else (pos_entry - exit_price)
        pnl = pts * pos_size * value_per_point
        capital += pnl
        pnls.append(pnl)
        pts_sum += pts
        in_pos = False
        pos_dir = 0

    def _try_open(sig: int, next_candle: M1Candle) -> None:
        nonlocal in_pos, pos_dir, pos_entry, pos_sl, pos_tp, pos_size
        spread = next_candle.close_ask - next_candle.close_bid
        if spread > max_spread:
            return
        size_ = _compute_size(next_candle.open)
        if size_ < min_lot_size:
            return
        half_spread = spread / 2.0
        slip = _sample_slip()
        if sig == 1:
            pos_entry = next_candle.open + half_spread + slip
            pos_sl = pos_entry - sl_points
            pos_tp = pos_entry + tp_points
        else:
            pos_entry = next_candle.open - half_spread - slip
            pos_sl = pos_entry + sl_points
            pos_tp = pos_entry - tp_points
        pos_dir = sig
        pos_size = size_
        in_pos = True

    for c in candles:
        # Check exits using mid H/L (small bias vs tick-level ok for search)
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

        # Execute pending signal at this candle's open (computed from i-1 close)
        if state.pending_signal != 0:
            sig = state.pending_signal
            state.pending_signal = 0
            # Reversal: close opposite at this candle open
            if in_pos and pos_dir != sig:
                spread = c.close_ask - c.close_bid
                half_spread = spread / 2.0
                # BUY exits at bid ≈ mid - half_spread, etc.
                exit_price = (
                    c.open - half_spread if pos_dir == 1
                    else c.open + half_spread
                )
                _close_at(exit_price)
            if not in_pos:
                _try_open(sig, c)

        # RSI + new signal at candle close
        r = rsi.update(c.close)
        new_sig = detect_signal(state, r, overbought, oversold)
        if new_sig != 0:
            state.pending_signal = new_sig

    # Metrics
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

    return FastBacktestResult(
        n_trades=n,
        wins=wins,
        losses=n - wins,
        win_rate=(wins / n) if n else 0.0,
        total_pnl=total,
        total_pnl_pts=pts_sum,
        max_drawdown=max_dd,
        final_capital=initial_capital + total,
        return_pct=(total / initial_capital * 100.0) if initial_capital else 0.0,
    )


# ─────────────────────────── Tick-level backtest ─────────────────────────────


async def run_tick_backtest(
    db: Database,
    *,
    instrument: str,
    start: datetime,
    end: datetime,
    rsi_length: int,
    overbought: float,
    oversold: float,
    sim_config: SimConfig,
    signal_proba: float = 0.0,
    log: logging.Logger | None = None,
    chunk_size: int = 100_000,
) -> list[MidasTrade]:
    """Stream ticks, build M1, compute RSI, simulate trades tick-by-tick."""
    builder = CandleBuilder(bucket_seconds=60)
    rsi = WilderRSI(rsi_length)
    state = _RsiState()
    sim = TradeSimulator(sim_config)

    n_ticks = 0
    n_candles = 0
    n_buy = 0
    n_sell = 0
    n_reversals = 0
    t0 = time.monotonic()

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
                sim.on_tick(tick)

                if state.pending_signal != 0:
                    sig = state.pending_signal
                    state.pending_signal = 0
                    ctx = sim.get_position_context(tick, 0)
                    if ctx is not None:
                        pos_dir = ctx["pos_direction"]
                        if (sig == 1 and pos_dir < 0) or (sig == 2 and pos_dir > 0):
                            sim.close_all(tick)
                            n_reversals += 1
                    sim.on_signal(tick, sig, proba=signal_proba)
                    if sig == 1:
                        n_buy += 1
                    else:
                        n_sell += 1

                closed = builder.process_tick(tick)
                if closed is not None:
                    n_candles += 1
                    r = rsi.update(float(closed["close"]))
                    ns = detect_signal(state, r, overbought, oversold)
                    if ns != 0:
                        state.pending_signal = ns

            if log is not None and n_ticks % 10_000_000 < chunk_size:
                elapsed = time.monotonic() - t0
                log.info(
                    "tick_backtest_progress ticks=%s candles=%s trades=%s "
                    "elapsed=%.1fs",
                    f"{n_ticks:,}", f"{n_candles:,}",
                    f"{len(sim.closed_trades):,}", elapsed,
                )

    builder.flush()
    if log is not None:
        log.info(
            "tick_backtest_done ticks=%s candles=%s signals_buy=%d "
            "signals_sell=%d reversals=%d",
            f"{n_ticks:,}", f"{n_candles:,}", n_buy, n_sell, n_reversals,
        )
    return sim.closed_trades


def compute_summary_from_trades(
    trades: list[MidasTrade],
    initial_capital: float,
    size: float,
    value_per_point: float,
) -> FastBacktestResult:
    """Compute a FastBacktestResult-equivalent summary from MidasTrades."""
    n = len(trades)
    if n == 0:
        return FastBacktestResult(
            0, 0, 0, 0.0, 0.0, 0.0, 0.0, initial_capital, 0.0,
        )
    wins = sum(1 for t in trades if t.is_win)
    total_pnl = sum(t.pnl for t in trades)
    total_pts = sum(t.pnl_points for t in trades)
    equity = initial_capital
    peak = initial_capital
    max_dd = 0.0
    for t in trades:
        equity += t.pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return FastBacktestResult(
        n_trades=n,
        wins=wins,
        losses=n - wins,
        win_rate=wins / n,
        total_pnl=total_pnl,
        total_pnl_pts=total_pts,
        max_drawdown=max_dd,
        final_capital=initial_capital + total_pnl,
        return_pct=total_pnl / initial_capital * 100.0,
    )


# ───────────────────────────── Output helpers ────────────────────────────────


def print_summary(tag: str, s: FastBacktestResult, initial_capital: float) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {tag}")
    print(f"{'=' * 60}")
    print(f"  Initial capital : {initial_capital:>10.2f} €")
    print(f"  Final capital   : {s.final_capital:>10.2f} €")
    print(f"  Return          : {s.return_pct:>+10.2f} %")
    print(f"  Total PnL       : {s.total_pnl:>+10.2f} €")
    print(f"  Total PnL (pts) : {s.total_pnl_pts:>+10.2f}")
    print(f"  Max drawdown    : {s.max_drawdown:>10.2f} €")
    print(f"  N trades        : {s.n_trades:>10}")
    print(f"  Wins / Losses   : {s.wins} / {s.losses}")
    print(f"  Win rate        : {s.win_rate * 100:>10.2f} %")
    print(f"{'=' * 60}")


def write_trades_csv(trades: list[MidasTrade], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "trade_id", "direction", "entry_time", "exit_time",
            "entry_price", "exit_price", "sl_price", "tp_price",
            "size", "pnl", "pnl_points", "is_win",
        ])
        for t in trades:
            w.writerow([
                t.trade_id, t.direction,
                t.entry_time.isoformat(), t.exit_time.isoformat(),
                f"{t.entry_price:.5f}", f"{t.exit_price:.5f}",
                f"{t.sl_price:.5f}", f"{t.tp_price:.5f}",
                f"{t.size:.4f}", f"{t.pnl:.4f}",
                f"{t.pnl_points:.4f}", int(t.is_win),
            ])


def write_summary_csv(summary: FastBacktestResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "n_trades", "wins", "losses", "win_rate",
            "total_pnl", "total_pnl_pts", "max_drawdown",
            "final_capital", "return_pct",
        ])
        w.writerow([
            summary.n_trades, summary.wins, summary.losses,
            f"{summary.win_rate:.6f}",
            f"{summary.total_pnl:.4f}", f"{summary.total_pnl_pts:.4f}",
            f"{summary.max_drawdown:.4f}",
            f"{summary.final_capital:.4f}", f"{summary.return_pct:.4f}",
        ])


# ──────────────────────────────── Optuna ─────────────────────────────────────


def run_optuna(
    candles: list[M1Candle],
    n_trials: int,
    *,
    slippage_min: float,
    slippage_max: float,
    initial_capital: float,
    size: float,
    value_per_point: float,
    margin_pct_per_trade: float | None = None,
    broker_margin_pct: float = 0.05,
    min_lot_size: float = 0.1,
    trials_csv: Path | None = None,
    log: logging.Logger,
) -> list[dict[str, Any]]:
    """TPE search on RSI/OB/OS/SL/TP. Returns per-trial records sorted by PnL."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    records: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        rsi_len = trial.suggest_int("rsi_length", 5, 30)
        ob = trial.suggest_float("overbought", 55.0, 95.0)
        os_ = trial.suggest_float("oversold", 5.0, 45.0)
        sl = trial.suggest_float("sl_points", 1.0, 20.0)
        tp = trial.suggest_float("tp_points", 1.0, 20.0)
        if os_ >= ob:
            return -1e9  # invalid

        r = simulate_fast(
            candles,
            rsi_length=rsi_len,
            overbought=ob, oversold=os_,
            sl_points=sl, tp_points=tp,
            slippage_min=slippage_min, slippage_max=slippage_max,
            slippage_seed=42,
            initial_capital=initial_capital,
            size=size, value_per_point=value_per_point,
            margin_pct_per_trade=margin_pct_per_trade,
            broker_margin_pct=broker_margin_pct,
            min_lot_size=min_lot_size,
        )

        # Penalize low trade count (don't reward overfit "3 lucky trades")
        penalty = 0.0
        if r.n_trades < 50:
            penalty = (50 - r.n_trades) * 2.0
        score = r.total_pnl - penalty

        records.append({
            "trial": trial.number,
            "rsi_length": rsi_len,
            "overbought": ob,
            "oversold": os_,
            "sl_points": sl,
            "tp_points": tp,
            "n_trades": r.n_trades,
            "win_rate": r.win_rate,
            "total_pnl": r.total_pnl,
            "max_drawdown": r.max_drawdown,
            "score": score,
        })
        log.info(
            "trial %3d/%d  rsi=%2d ob=%.1f os=%.1f sl=%.2f tp=%.2f "
            "| n=%4d wr=%.1f%% pnl=%+.2f dd=%.2f",
            trial.number + 1, n_trials,
            rsi_len, ob, os_, sl, tp,
            r.n_trades, r.win_rate * 100,
            r.total_pnl, r.max_drawdown,
        )
        return score

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=10)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    if trials_csv is not None:
        trials_csv.parent.mkdir(parents=True, exist_ok=True)
        with trials_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            w.writeheader()
            w.writerows(records)

    records.sort(key=lambda r: r["score"], reverse=True)
    return records


# ───────────────────────────── Walk-forward ──────────────────────────────────


def _add_months(dt: datetime, months: int) -> datetime:
    """Add calendar months to a datetime (handles year wrap)."""
    m = dt.month - 1 + months
    y = dt.year + m // 12
    return dt.replace(year=y, month=m % 12 + 1)


def _slice_candles(
    candles: list[M1Candle], start: datetime, end: datetime,
) -> list[M1Candle]:
    """Return candles in [start, end)."""
    return [c for c in candles if start <= c.time < end]


def run_walk_forward(
    candles: list[M1Candle],
    *,
    full_start: datetime,
    full_end: datetime,
    train_days: int,
    test_days: int,
    step_days: int,
    n_trials: int,
    slippage_min: float,
    slippage_max: float,
    initial_capital: float,
    size: float,
    value_per_point: float,
    margin_pct_per_trade: float | None,
    broker_margin_pct: float,
    min_lot_size: float,
    output_prefix: Path | None,
    log: logging.Logger,
) -> list[dict[str, Any]]:
    """Walk-forward: Optuna on train → apply best on OOS test, step forward."""
    window_records: list[dict[str, Any]] = []
    window_idx = 0
    train_start = full_start

    while True:
        train_end = train_start + timedelta(days=train_days)
        test_end = train_end + timedelta(days=test_days)
        if test_end > full_end:
            break

        train_candles = _slice_candles(candles, train_start, train_end)
        test_candles = _slice_candles(candles, train_end, test_end)
        if not train_candles or not test_candles:
            log.warning(
                "window %d skipped (empty slice)", window_idx,
            )
            train_start = train_start + timedelta(days=step_days)
            window_idx += 1
            continue

        log.info(
            "────────── WINDOW %d: train %s→%s (%d candles)  "
            "test %s→%s (%d candles) ──────────",
            window_idx,
            train_start.date(), train_end.date(), len(train_candles),
            train_end.date(), test_end.date(), len(test_candles),
        )

        # Train: Optuna on train_candles
        train_records = run_optuna(
            train_candles,
            n_trials=n_trials,
            slippage_min=slippage_min,
            slippage_max=slippage_max,
            initial_capital=initial_capital,
            size=size,
            value_per_point=value_per_point,
            margin_pct_per_trade=margin_pct_per_trade,
            broker_margin_pct=broker_margin_pct,
            min_lot_size=min_lot_size,
            trials_csv=None,
            log=log,
        )
        best = train_records[0]

        # OOS: apply best on test_candles
        oos = simulate_fast(
            test_candles,
            rsi_length=best["rsi_length"],
            overbought=best["overbought"],
            oversold=best["oversold"],
            sl_points=best["sl_points"],
            tp_points=best["tp_points"],
            slippage_min=slippage_min, slippage_max=slippage_max,
            slippage_seed=42,
            initial_capital=initial_capital,
            size=size, value_per_point=value_per_point,
            margin_pct_per_trade=margin_pct_per_trade,
            broker_margin_pct=broker_margin_pct,
            min_lot_size=min_lot_size,
        )

        rec = {
            "window_idx": window_idx,
            "train_start": train_start.date().isoformat(),
            "train_end": train_end.date().isoformat(),
            "test_start": train_end.date().isoformat(),
            "test_end": test_end.date().isoformat(),
            "rsi_length": best["rsi_length"],
            "overbought": best["overbought"],
            "oversold": best["oversold"],
            "sl_points": best["sl_points"],
            "tp_points": best["tp_points"],
            "train_pnl": best["total_pnl"],
            "train_n_trades": best["n_trades"],
            "train_wr": best["win_rate"],
            "oos_pnl": oos.total_pnl,
            "oos_n_trades": oos.n_trades,
            "oos_wr": oos.win_rate,
            "oos_return_pct": oos.return_pct,
            "oos_mdd": oos.max_drawdown,
        }
        window_records.append(rec)
        log.info(
            "WINDOW %d  best: rsi=%d ob=%.1f os=%.1f sl=%.2f tp=%.2f  "
            "TRAIN n=%d wr=%.1f%% pnl=%+.0f  |  OOS n=%d wr=%.1f%% pnl=%+.0f "
            "(%+.1f%%)  mdd=%.0f",
            window_idx,
            best["rsi_length"], best["overbought"], best["oversold"],
            best["sl_points"], best["tp_points"],
            best["n_trades"], best["win_rate"] * 100, best["total_pnl"],
            oos.n_trades, oos.win_rate * 100, oos.total_pnl,
            oos.return_pct, oos.max_drawdown,
        )

        train_start = train_start + timedelta(days=step_days)
        window_idx += 1

    if output_prefix is not None:
        csv_path = Path(str(output_prefix) + "_wf_windows.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            if window_records:
                w = csv.DictWriter(f, fieldnames=list(window_records[0].keys()))
                w.writeheader()
                w.writerows(window_records)
        log.info("wrote %s", csv_path)

    return window_records


# ─────────────────────────────── Main ────────────────────────────────────────


def _build_sim_config(args: argparse.Namespace, sl_pts: float, tp_pts: float) -> SimConfig:
    """Build a SimConfig with either flat size or margin-based dynamic sizing.

    Margin mode uses the gamma-ramp trick: gamma=1, threshold=0,
    max_margin_proba=1, and ``proba=margin_pct_per_trade`` at signal time
    produces a size using exactly that fraction of available margin.
    """
    if args.margin_pct_per_trade is not None:
        return SimConfig(
            sl_points=sl_pts, tp_points=tp_pts,
            initial_capital=args.initial_capital,
            size=args.size,  # unused in dynamic mode
            value_per_point=args.value_per_point,
            max_open_positions=1,
            max_spread=args.max_spread,
            gamma=1.0,
            sizing_threshold=0.0,
            max_margin_proba=1.0,
            margin_pct=args.broker_margin_pct,
            min_lot_size=args.min_lot_size,
            slippage_min_pts=args.slippage_min,
            slippage_max_pts=args.slippage_max,
            slippage_seed=args.slippage_seed,
        )
    return SimConfig(
        sl_points=sl_pts, tp_points=tp_pts,
        initial_capital=args.initial_capital,
        size=args.size,
        value_per_point=args.value_per_point,
        max_open_positions=1,
        max_spread=args.max_spread,
        slippage_min_pts=args.slippage_min,
        slippage_max_pts=args.slippage_max,
        slippage_seed=args.slippage_seed,
    )


async def main(args: argparse.Namespace) -> None:
    config = load_config()
    setup_logging(config.logging.level, json_format=False)
    log = logging.getLogger("rsi_backtest")
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
            "range instrument=%s start=%s end=%s",
            args.instrument, args.start.date(), args.end.date(),
        )

        if args.walk_forward:
            log.info("loading m1 candles from ticks (walk-forward) …")
            candles = await load_m1_candles(
                db, args.instrument, args.start, args.end, log=log,
            )
            if not candles:
                log.error("no candles produced, aborting")
                return

            output_prefix = Path(args.output) if args.output else None
            records = run_walk_forward(
                candles,
                full_start=args.start,
                full_end=args.end,
                train_days=args.train_days,
                test_days=args.test_days,
                step_days=args.step_days if args.step_days else args.test_days,
                n_trials=args.optuna_trials or 50,
                slippage_min=args.slippage_min,
                slippage_max=args.slippage_max,
                initial_capital=args.initial_capital,
                size=args.size,
                value_per_point=args.value_per_point,
                margin_pct_per_trade=args.margin_pct_per_trade,
                broker_margin_pct=args.broker_margin_pct,
                min_lot_size=args.min_lot_size,
                output_prefix=output_prefix,
                log=log,
            )

            # Aggregate
            print("\n── WALK-FORWARD SUMMARY ──")
            total_oos = sum(r["oos_pnl"] for r in records)
            pos_wins = sum(1 for r in records if r["oos_pnl"] > 0)
            mean_wr = (
                sum(r["oos_wr"] for r in records) / len(records)
                if records else 0.0
            )
            total_trades = sum(r["oos_n_trades"] for r in records)
            for r in records:
                print(
                    f"  W{r['window_idx']:2d}  "
                    f"train={r['train_start']}→{r['train_end']}  "
                    f"test={r['test_start']}→{r['test_end']}  "
                    f"n={r['oos_n_trades']:4d} wr={r['oos_wr'] * 100:5.1f}% "
                    f"pnl={r['oos_pnl']:+9.2f}",
                )
            print(
                f"\n  Windows       : {len(records)}  "
                f"({pos_wins} positive, {len(records) - pos_wins} negative)",
            )
            print(f"  Total OOS PnL : {total_oos:+.2f} €")
            print(f"  Total trades  : {total_trades}")
            print(f"  Mean OOS WR   : {mean_wr * 100:.2f} %")
            return

        if args.optuna_trials > 0:
            # ─── Optuna mode: cache candles, run N trials ───
            log.info("loading m1 candles from ticks …")
            candles = await load_m1_candles(
                db, args.instrument, args.start, args.end, log=log,
            )
            if not candles:
                log.error("no candles produced, aborting")
                return

            log.info("running optuna n_trials=%d", args.optuna_trials)
            trials_csv = None
            if args.output:
                trials_csv = Path(args.output + "_trials.csv")
            records = run_optuna(
                candles,
                n_trials=args.optuna_trials,
                slippage_min=args.slippage_min,
                slippage_max=args.slippage_max,
                initial_capital=args.initial_capital,
                size=args.size,
                value_per_point=args.value_per_point,
                margin_pct_per_trade=args.margin_pct_per_trade,
                broker_margin_pct=args.broker_margin_pct,
                min_lot_size=args.min_lot_size,
                trials_csv=trials_csv,
                log=log,
            )

            print("\n── TOP 5 TRIALS (by score) ──")
            for r in records[:5]:
                print(
                    f"  #{r['trial']:3d}  rsi={r['rsi_length']:2d} "
                    f"ob={r['overbought']:.1f} os={r['oversold']:.1f} "
                    f"sl={r['sl_points']:.2f} tp={r['tp_points']:.2f}  "
                    f"n={r['n_trades']:4d} wr={r['win_rate'] * 100:.1f}% "
                    f"pnl={r['total_pnl']:+.2f}",
                )

            if args.validate_top > 0:
                log.info(
                    "validating top-%d in tick-level mode", args.validate_top,
                )
                proba = args.margin_pct_per_trade or 0.0
                for r in records[: args.validate_top]:
                    sim_cfg = _build_sim_config(
                        args, r["sl_points"], r["tp_points"],
                    )
                    trades = await run_tick_backtest(
                        db,
                        instrument=args.instrument,
                        start=args.start, end=args.end,
                        rsi_length=r["rsi_length"],
                        overbought=r["overbought"],
                        oversold=r["oversold"],
                        sim_config=sim_cfg,
                        signal_proba=proba,
                        log=log,
                    )
                    s = compute_summary_from_trades(
                        trades, args.initial_capital,
                        args.size, args.value_per_point,
                    )
                    print_summary(
                        f"TICK-LEVEL VALIDATION trial #{r['trial']}",
                        s, args.initial_capital,
                    )
            return

        # ─── Single tick-level backtest ───
        sim_cfg = _build_sim_config(args, args.sl_points, args.tp_points)
        mpt = args.margin_pct_per_trade
        log.info(
            "config rsi=%d ob=%.1f os=%.1f sl=%.2f tp=%.2f slip=%.2f-%.2f "
            "sizing=%s",
            args.rsi_length, args.overbought, args.oversold,
            args.sl_points, args.tp_points,
            args.slippage_min, args.slippage_max,
            (f"margin={mpt * 100:.0f}%" if mpt is not None
             else f"flat={args.size}"),
        )
        trades = await run_tick_backtest(
            db,
            instrument=args.instrument,
            start=args.start, end=args.end,
            rsi_length=args.rsi_length,
            overbought=args.overbought, oversold=args.oversold,
            sim_config=sim_cfg,
            signal_proba=mpt or 0.0,
            log=log,
        )
        s = compute_summary_from_trades(
            trades, args.initial_capital, args.size, args.value_per_point,
        )
        print_summary("RSI BACKTEST (tick-level)", s, args.initial_capital)

        if args.output:
            prefix = Path(args.output)
            write_trades_csv(
                trades, prefix.with_name(prefix.name + "_trades.csv"),
            )
            write_summary_csv(
                s, prefix.with_name(prefix.name + "_summary.csv"),
            )
            log.info("wrote %s_{trades,summary}.csv", prefix)

    finally:
        await db.disconnect()


def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RSI M1 scalping backtest")
    p.add_argument("--instrument", default="XAUUSD")
    p.add_argument("--start", type=_parse_dt, required=True)
    p.add_argument("--end", type=_parse_dt, required=True)
    # Strategy params (used in single-backtest mode)
    p.add_argument("--rsi-length", type=int, default=14)
    p.add_argument("--overbought", type=float, default=74.0)
    p.add_argument("--oversold", type=float, default=24.0)
    p.add_argument("--sl-points", type=float, default=7.0)
    p.add_argument("--tp-points", type=float, default=4.0)
    # Sim
    p.add_argument("--initial-capital", type=float, default=5_000.0)
    p.add_argument("--size", type=float, default=0.1)
    p.add_argument("--value-per-point", type=float, default=1.0)
    p.add_argument("--max-spread", type=float, default=2.0)
    p.add_argument("--slippage-min", type=float, default=0.0)
    p.add_argument("--slippage-max", type=float, default=0.0)
    p.add_argument("--slippage-seed", type=int, default=42)
    # Margin-based sizing (overrides --size when set)
    p.add_argument(
        "--margin-pct-per-trade", type=float, default=None,
        help="Fraction of current capital to commit as margin per trade "
             "(e.g. 0.5 = 50%%). When set, overrides --size.",
    )
    p.add_argument(
        "--broker-margin-pct", type=float, default=0.05,
        help="Broker margin requirement as fraction of notional (default 0.05 = 5%%).",
    )
    p.add_argument(
        "--min-lot-size", type=float, default=0.1,
        help="Minimum tradeable lot (default 0.1).",
    )
    # Optuna
    p.add_argument(
        "--optuna-trials", type=int, default=0,
        help="0 = single tick-level backtest; >0 enables Optuna search "
             "on cached M1 candles.",
    )
    p.add_argument(
        "--validate-top", type=int, default=0,
        help="After Optuna, re-validate top-K trials in tick-level mode.",
    )
    # Walk-forward
    p.add_argument(
        "--walk-forward", action="store_true",
        help="Enable walk-forward: Optuna on train, apply best on OOS test, "
             "step forward.",
    )
    p.add_argument("--train-days", type=int, default=180)
    p.add_argument("--test-days", type=int, default=30)
    p.add_argument("--step-days", type=int, default=None,
                   help="Step size in days (default: same as --test-days)")
    # Output
    p.add_argument("--output", type=str, default=None)
    return p


if __name__ == "__main__":
    load_dotenv()
    parser = build_parser()
    asyncio.run(main(parser.parse_args()))
