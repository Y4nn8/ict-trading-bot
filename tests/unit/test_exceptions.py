"""Tests for the exception hierarchy."""

from __future__ import annotations

from src.common.exceptions import (
    BacktestError,
    BrokerAuthError,
    BrokerError,
    ConfigError,
    DatabaseError,
    ExecutionError,
    MarketDataError,
    RiskLimitError,
    TradingBotError,
)


class TestExceptionHierarchy:
    """Tests for exception inheritance."""

    def test_base_exception(self) -> None:
        exc = TradingBotError("test")
        assert str(exc) == "test"
        assert isinstance(exc, Exception)

    def test_config_error_is_trading_bot_error(self) -> None:
        assert issubclass(ConfigError, TradingBotError)

    def test_database_error_is_trading_bot_error(self) -> None:
        assert issubclass(DatabaseError, TradingBotError)

    def test_broker_auth_is_broker_error(self) -> None:
        assert issubclass(BrokerAuthError, BrokerError)
        assert issubclass(BrokerAuthError, TradingBotError)

    def test_risk_limit_is_execution_error(self) -> None:
        assert issubclass(RiskLimitError, ExecutionError)
        assert issubclass(RiskLimitError, TradingBotError)

    def test_market_data_error(self) -> None:
        assert issubclass(MarketDataError, TradingBotError)

    def test_backtest_error(self) -> None:
        assert issubclass(BacktestError, TradingBotError)
