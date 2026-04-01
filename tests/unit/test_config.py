"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.common.config import AppConfig, load_config
from src.common.exceptions import ConfigError


class TestLoadConfig:
    """Tests for the config loader."""

    def test_load_default_config(self) -> None:
        config = load_config()
        assert isinstance(config, AppConfig)
        assert config.logging.level == "INFO"
        assert config.database.min_connections == 2

    def test_config_has_instruments(self) -> None:
        config = load_config()
        assert len(config.instruments) > 0
        assert config.instruments[0].name == "EUR/USD"

    def test_config_risk_defaults(self) -> None:
        config = load_config()
        assert config.risk.max_daily_drawdown_pct == 3.0
        assert config.risk.max_total_drawdown_pct == 10.0
        assert config.risk.max_simultaneous_positions == 5

    def test_config_walk_forward_defaults(self) -> None:
        config = load_config()
        assert config.backtest.walk_forward.train_months == 4
        assert config.backtest.walk_forward.test_months == 1

    def test_missing_config_dir_raises(self) -> None:
        with pytest.raises(ConfigError, match="Default config not found"):
            load_config(config_path=Path("/nonexistent/path"))

    def test_database_url_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABASE_URL", "postgresql://custom:custom@db:5432/custom")
        config = load_config()
        assert config.database.url == "postgresql://custom:custom@db:5432/custom"

    def test_confluence_risk_map(self) -> None:
        config = load_config()
        assert config.risk.confluence_risk_map.low == 0.5
        assert config.risk.confluence_risk_map.medium == 1.0
        assert config.risk.confluence_risk_map.high == 2.0

    def test_broker_config_defaults(self) -> None:
        config = load_config()
        assert config.broker.acc_type == "DEMO"

    def test_market_data_config(self) -> None:
        config = load_config()
        assert config.market_data.base_timeframe == "M5"
        assert "H1" in config.market_data.higher_timeframes
        assert config.market_data.default_history_days == 180
