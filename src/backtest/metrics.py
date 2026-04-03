"""Backtest performance metrics.

Computes Sharpe, Sortino, MDD, profit factor, win rate, average R:R.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.common.models import Trade


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Computed performance metrics for a set of trades."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    avg_pnl: float
    avg_winner: float
    avg_loser: float
    avg_r_multiple: float
    avg_risk_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float


def compute_metrics(
    trades: list[Trade],
    initial_capital: float = 10000.0,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252.0,
) -> PerformanceMetrics:
    """Compute performance metrics from a list of trades.

    Args:
        trades: List of closed Trade objects.
        initial_capital: Starting capital for drawdown calculation.
        risk_free_rate: Annual risk-free rate for Sharpe/Sortino.
        periods_per_year: Number of trading periods per year.

    Returns:
        PerformanceMetrics with all computed values.
    """
    if not trades:
        return _empty_metrics()

    pnls = [t.pnl or 0.0 for t in trades]
    r_multiples = [t.r_multiple or 0.0 for t in trades]

    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]

    total_trades = len(trades)
    winning_trades = len(winners)
    losing_trades = len(losers)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

    gross_profit = sum(winners) if winners else 0.0
    gross_loss = abs(sum(losers)) if losers else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    total_pnl = sum(pnls)
    avg_pnl = total_pnl / total_trades
    avg_winner = sum(winners) / len(winners) if winners else 0.0
    avg_loser = sum(losers) / len(losers) if losers else 0.0
    avg_r_multiple = sum(r_multiples) / len(r_multiples) if r_multiples else 0.0

    # Average risk % of capital per trade
    risk_pcts: list[float] = []
    running_capital = initial_capital
    for t in trades:
        entry = t.entry_price or 0.0
        sl = t.stop_loss or 0.0
        size = t.size or 0.0
        risk_amount = abs(entry - sl) * size
        if running_capital > 0:
            risk_pcts.append(risk_amount / running_capital * 100)
        running_capital += t.pnl or 0.0
    avg_risk_pct = sum(risk_pcts) / len(risk_pcts) if risk_pcts else 0.0

    # Drawdown calculation
    equity_curve = _build_equity_curve(pnls, initial_capital)
    max_dd, max_dd_pct = _compute_max_drawdown(equity_curve)

    # Sharpe ratio
    pnl_array = np.array(pnls)
    mean_return = float(np.mean(pnl_array))
    std_return = float(np.std(pnl_array, ddof=1)) if len(pnl_array) > 1 else 0.0

    daily_rf = risk_free_rate / periods_per_year
    sharpe = (
        (mean_return - daily_rf) / std_return * np.sqrt(periods_per_year)
        if std_return > 0
        else 0.0
    )

    # Sortino ratio (downside deviation)
    negative_returns = pnl_array[pnl_array < 0]
    downside_std = (
        float(np.std(negative_returns, ddof=1)) if len(negative_returns) > 1 else 0.0
    )
    sortino = (
        (mean_return - daily_rf) / downside_std * np.sqrt(periods_per_year)
        if downside_std > 0
        else 0.0
    )

    # Calmar ratio
    annualized_return = mean_return * periods_per_year
    calmar = annualized_return / abs(max_dd_pct) if max_dd_pct != 0 else 0.0

    return PerformanceMetrics(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_pnl=total_pnl,
        avg_pnl=avg_pnl,
        avg_winner=avg_winner,
        avg_loser=avg_loser,
        avg_r_multiple=avg_r_multiple,
        avg_risk_pct=avg_risk_pct,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
        calmar_ratio=float(calmar),
    )


def _build_equity_curve(pnls: list[float], initial_capital: float) -> list[float]:
    """Build cumulative equity curve from PnL list."""
    curve = [initial_capital]
    for pnl in pnls:
        curve.append(curve[-1] + pnl)
    return curve


def _compute_max_drawdown(equity_curve: list[float]) -> tuple[float, float]:
    """Compute maximum drawdown in absolute and percentage terms."""
    if len(equity_curve) < 2:
        return 0.0, 0.0

    peak = equity_curve[0]
    max_dd = 0.0
    max_dd_pct = 0.0

    for equity in equity_curve[1:]:
        if equity > peak:
            peak = equity
        dd = peak - equity
        dd_pct = dd / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd_pct

    return max_dd, max_dd_pct


def _empty_metrics() -> PerformanceMetrics:
    return PerformanceMetrics(
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate=0.0,
        profit_factor=0.0,
        total_pnl=0.0,
        avg_pnl=0.0,
        avg_winner=0.0,
        avg_loser=0.0,
        avg_r_multiple=0.0,
        avg_risk_pct=0.0,
        max_drawdown=0.0,
        max_drawdown_pct=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        calmar_ratio=0.0,
    )
