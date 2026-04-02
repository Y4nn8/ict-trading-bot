"""Walk-forward validation engine.

Splits data into train/test windows and aggregates metrics
across all windows to prevent overfitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from src.common.logging import get_logger

if TYPE_CHECKING:
    from datetime import datetime

    from src.backtest.metrics import PerformanceMetrics
    from src.common.models import Trade

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    """A single walk-forward train/test window."""

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_metrics: PerformanceMetrics
    test_metrics: PerformanceMetrics


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward validation results."""

    windows: list[WalkForwardWindow] = field(default_factory=list)
    mean_sharpe: float = 0.0
    mean_sortino: float = 0.0
    mean_profit_factor: float = 0.0
    mean_win_rate: float = 0.0
    mean_avg_r: float = 0.0
    worst_window_mdd: float = 0.0
    total_test_trades: int = 0


def generate_windows(
    start: datetime,
    end: datetime,
    train_months: int = 4,
    test_months: int = 1,
    step_months: int = 1,
) -> list[tuple[datetime, datetime, datetime, datetime]]:
    """Generate walk-forward window boundaries.

    Args:
        start: Data start date.
        end: Data end date.
        train_months: Training period in months.
        test_months: Testing period in months.
        step_months: Step size between windows in months.

    Returns:
        List of (train_start, train_end, test_start, test_end) tuples.
    """
    windows: list[tuple[datetime, datetime, datetime, datetime]] = []

    current = start
    while True:
        train_start = current
        train_end = _add_months(train_start, train_months)
        test_start = train_end
        test_end = _add_months(test_start, test_months)

        if test_end > end:
            break

        windows.append((train_start, train_end, test_start, test_end))
        current = _add_months(current, step_months)

    return windows


def aggregate_walk_forward(
    windows: list[WalkForwardWindow],
) -> WalkForwardResult:
    """Aggregate metrics across walk-forward windows.

    Args:
        windows: List of completed walk-forward windows.

    Returns:
        WalkForwardResult with aggregated metrics.
    """
    if not windows:
        return WalkForwardResult()

    test_metrics = [w.test_metrics for w in windows]

    sharpes = [m.sharpe_ratio for m in test_metrics]
    sortinos = [m.sortino_ratio for m in test_metrics]
    pfs = [m.profit_factor for m in test_metrics if m.profit_factor != float("inf")]
    win_rates = [m.win_rate for m in test_metrics]
    avg_rs = [m.avg_r_multiple for m in test_metrics]
    mdds = [m.max_drawdown_pct for m in test_metrics]
    total_trades = sum(m.total_trades for m in test_metrics)

    return WalkForwardResult(
        windows=windows,
        mean_sharpe=float(np.mean(sharpes)) if sharpes else 0.0,
        mean_sortino=float(np.mean(sortinos)) if sortinos else 0.0,
        mean_profit_factor=float(np.mean(pfs)) if pfs else 0.0,
        mean_win_rate=float(np.mean(win_rates)) if win_rates else 0.0,
        mean_avg_r=float(np.mean(avg_rs)) if avg_rs else 0.0,
        worst_window_mdd=max(mdds) if mdds else 0.0,
        total_test_trades=total_trades,
    )


def split_trades_by_time(
    trades: list[Trade],
    start: datetime,
    end: datetime,
) -> list[Trade]:
    """Filter trades that fall within a time window.

    Args:
        trades: All trades.
        start: Window start (inclusive).
        end: Window end (exclusive).

    Returns:
        Filtered list of trades.
    """
    return [
        t for t in trades
        if t.opened_at >= start and t.opened_at < end
    ]


def _add_months(dt: datetime, months: int) -> datetime:
    """Add months to a datetime, handling year rollover."""
    month = dt.month + months
    year = dt.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    day = min(dt.day, 28)  # Safe for all months
    return dt.replace(year=year, month=month, day=day)
