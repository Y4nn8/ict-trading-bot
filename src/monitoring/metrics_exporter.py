"""Prometheus metrics exporter for Grafana monitoring."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Trading metrics
TRADES_TOTAL = Counter(
    "trading_bot_trades_total",
    "Total number of trades executed",
    ["instrument", "direction", "result"],
)

TRADE_PNL = Histogram(
    "trading_bot_trade_pnl",
    "PnL distribution per trade",
    ["instrument"],
    buckets=[-100, -50, -20, -10, 0, 10, 20, 50, 100, 200, 500],
)

# Portfolio metrics
PORTFOLIO_CAPITAL = Gauge(
    "trading_bot_portfolio_capital",
    "Current portfolio capital",
)

PORTFOLIO_EQUITY = Gauge(
    "trading_bot_portfolio_equity",
    "Current portfolio equity (capital + unrealized PnL)",
)

OPEN_POSITIONS = Gauge(
    "trading_bot_open_positions",
    "Number of open positions",
)

DAILY_PNL = Gauge(
    "trading_bot_daily_pnl",
    "Today's cumulative PnL",
)

# System metrics
CANDLE_PROCESSING_TIME = Histogram(
    "trading_bot_candle_processing_seconds",
    "Time to process a single candle",
)

NEWS_EVENTS_RECEIVED = Counter(
    "trading_bot_news_events_total",
    "Total news events received",
    ["source"],
)

CIRCUIT_BREAKER_TRIPS = Counter(
    "trading_bot_circuit_breaker_trips_total",
    "Number of circuit breaker activations",
    ["type"],
)


def start_metrics_server(port: int = 9090) -> None:
    """Start the Prometheus metrics HTTP server.

    Args:
        port: Port to listen on.
    """
    start_http_server(port)


def record_trade(
    instrument: str,
    direction: str,
    pnl: float,
) -> None:
    """Record a completed trade in metrics.

    Args:
        instrument: Instrument name.
        direction: Trade direction.
        pnl: Trade PnL.
    """
    result = "win" if pnl > 0 else "loss"
    TRADES_TOTAL.labels(instrument=instrument, direction=direction, result=result).inc()
    TRADE_PNL.labels(instrument=instrument).observe(pnl)


def update_portfolio_metrics(
    capital: float,
    equity: float,
    positions: int,
    daily_pnl: float,
) -> None:
    """Update portfolio Prometheus gauges.

    Args:
        capital: Current capital.
        equity: Current equity.
        positions: Open position count.
        daily_pnl: Today's PnL.
    """
    PORTFOLIO_CAPITAL.set(capital)
    PORTFOLIO_EQUITY.set(equity)
    OPEN_POSITIONS.set(positions)
    DAILY_PNL.set(daily_pnl)
