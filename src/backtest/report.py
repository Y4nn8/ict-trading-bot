"""Backtest report generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.backtest.metrics import compute_metrics

if TYPE_CHECKING:
    from src.common.models import Trade


def generate_report(
    trades: list[Trade],
    initial_capital: float = 10000.0,
) -> dict[str, Any]:
    """Generate a backtest report from trade results.

    Args:
        trades: List of closed trades.
        initial_capital: Starting capital.

    Returns:
        Dict with report sections.
    """
    metrics = compute_metrics(trades, initial_capital)

    return {
        "summary": {
            "total_trades": metrics.total_trades,
            "winning_trades": metrics.winning_trades,
            "losing_trades": metrics.losing_trades,
            "win_rate": round(metrics.win_rate * 100, 2),
            "profit_factor": round(metrics.profit_factor, 2),
            "total_pnl": round(metrics.total_pnl, 2),
            "initial_capital": initial_capital,
            "final_capital": round(initial_capital + metrics.total_pnl, 2),
            "return_pct": round(metrics.total_pnl / initial_capital * 100, 2),
        },
        "risk_metrics": {
            "sharpe_ratio": round(metrics.sharpe_ratio, 3),
            "sortino_ratio": round(metrics.sortino_ratio, 3),
            "calmar_ratio": round(metrics.calmar_ratio, 3),
            "max_drawdown": round(metrics.max_drawdown, 2),
            "max_drawdown_pct": round(metrics.max_drawdown_pct * 100, 2),
        },
        "trade_metrics": {
            "avg_pnl": round(metrics.avg_pnl, 2),
            "avg_winner": round(metrics.avg_winner, 2),
            "avg_loser": round(metrics.avg_loser, 2),
            "avg_r_multiple": round(metrics.avg_r_multiple, 2),
        },
    }


def format_report(report: dict[str, Any]) -> str:
    """Format a report dict as a readable string.

    Args:
        report: Report dict from generate_report.

    Returns:
        Formatted multi-line string.
    """
    lines = ["=" * 50, "BACKTEST REPORT", "=" * 50, ""]

    summary = report["summary"]
    lines.append("SUMMARY")
    lines.append(f"  Total trades:    {summary['total_trades']}")
    lines.append(f"  Win rate:        {summary['win_rate']}%")
    lines.append(f"  Profit factor:   {summary['profit_factor']}")
    lines.append(f"  Total PnL:       {summary['total_pnl']}")
    lines.append(f"  Return:          {summary['return_pct']}%")
    lines.append(f"  Final capital:   {summary['final_capital']}")
    lines.append("")

    risk = report["risk_metrics"]
    lines.append("RISK METRICS")
    lines.append(f"  Sharpe ratio:    {risk['sharpe_ratio']}")
    lines.append(f"  Sortino ratio:   {risk['sortino_ratio']}")
    lines.append(f"  Max drawdown:    {risk['max_drawdown_pct']}%")
    lines.append("")

    trade = report["trade_metrics"]
    lines.append("TRADE METRICS")
    lines.append(f"  Avg PnL:         {trade['avg_pnl']}")
    lines.append(f"  Avg winner:      {trade['avg_winner']}")
    lines.append(f"  Avg loser:       {trade['avg_loser']}")
    lines.append(f"  Avg R:R:         {trade['avg_r_multiple']}")

    return "\n".join(lines)
