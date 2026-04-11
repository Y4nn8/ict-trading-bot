"""Custom exception hierarchy for the trading bot."""

from __future__ import annotations


class TradingBotError(Exception):
    """Base exception for all trading bot errors."""


class ConfigError(TradingBotError):
    """Raised when configuration is invalid or missing."""


class DatabaseError(TradingBotError):
    """Raised when a database operation fails."""


class BrokerError(TradingBotError):
    """Raised when a broker API call fails."""


class BrokerAuthError(BrokerError):
    """Raised when broker authentication fails."""


class BrokerRateLimitError(BrokerError):
    """Raised when broker API rate limit is exceeded."""


class MarketDataError(TradingBotError):
    """Raised when market data ingestion or processing fails."""


class StructureDetectionError(TradingBotError):
    """Raised when a structure detection algorithm fails."""


class StrategyError(TradingBotError):
    """Raised when strategy evaluation fails."""


class ExecutionError(TradingBotError):
    """Raised when order execution fails."""


class RiskLimitError(ExecutionError):
    """Raised when a risk limit or circuit breaker is triggered."""


class NewsError(TradingBotError):
    """Raised when news ingestion or interpretation fails."""


class BacktestError(TradingBotError):
    """Raised when the backtest engine encounters an error."""


class ImprovementError(TradingBotError):
    """Raised when the improvement loop fails."""


class MidasError(TradingBotError):
    """Raised when the Midas tick-level engine encounters an error."""
