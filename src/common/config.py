"""Configuration loader and validation using Pydantic."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from src.common.exceptions import ConfigError

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    json_format: bool = True


class DatabaseConfig(BaseModel):
    """Database connection configuration."""

    url: str = "postgresql://trader:trader_secret@localhost:5432/trading_bot"
    min_connections: int = 2
    max_connections: int = 10


class BrokerConfig(BaseModel):
    """IG Markets broker configuration."""

    api_key: str = ""
    username: str = ""
    password: str = ""
    acc_number: str = ""
    acc_type: str = "DEMO"


class MarketDataConfig(BaseModel):
    """Market data ingestion configuration."""

    base_timeframe: str = "M5"
    higher_timeframes: list[str] = Field(default_factory=lambda: ["H1", "H4", "D1"])
    default_history_days: int = 180


class InstrumentConfig(BaseModel):
    """Per-instrument configuration."""

    name: str
    epic: str
    asset_class: str
    leverage: int
    min_size: float = 0.5
    value_per_point: float = 1.0
    point_currency: str = "EUR"  # Currency of the value_per_point
    min_spread: float = 0.0
    avg_spread: float = 0.0


class ConfluenceRiskMap(BaseModel):
    """Maps confluence levels to risk percentages."""

    low: float = 0.5
    medium: float = 1.0
    high: float = 2.0


class RiskConfig(BaseModel):
    """Risk management configuration."""

    max_risk_per_trade_pct: float = 1.0
    max_daily_drawdown_pct: float = 3.0
    max_total_drawdown_pct: float = 10.0
    max_simultaneous_positions: int = 5
    confluence_risk_map: ConfluenceRiskMap = Field(default_factory=ConfluenceRiskMap)


class StrategyConfig(BaseModel):
    """Strategy configuration."""

    min_confluence_score: float = 0.4
    entry_timeout_candles: int = 6


class WalkForwardConfig(BaseModel):
    """Walk-forward validation configuration."""

    train_months: int = 4
    test_months: int = 1
    step_months: int = 1


class SimulationConfig(BaseModel):
    """Backtest simulation configuration."""

    slippage_max_pips: float = 2.0
    order_rejection_rate: float = 0.01


class BacktestConfig(BaseModel):
    """Backtest configuration."""

    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)


class OptunaConfig(BaseModel):
    """Optuna optimizer configuration."""

    n_trials: int = 100
    min_improvement_pct: float = 2.0
    max_mdd_degradation_pct: float = 5.0


class LLMImprovementConfig(BaseModel):
    """LLM improvement loop configuration."""

    max_iterations: int = 5
    max_sharpe_jump_pct: float = 50.0


class ImprovementConfig(BaseModel):
    """Improvement loop configuration."""

    optuna: OptunaConfig = Field(default_factory=OptunaConfig)
    llm: LLMImprovementConfig = Field(default_factory=LLMImprovementConfig)


class NewsConfig(BaseModel):
    """News configuration."""

    pre_event_pause_minutes: int = 30
    post_event_resume_minutes: int = 15
    finnhub_api_key: str = ""
    anthropic_api_key: str = ""


class AppConfig(BaseModel):
    """Root application configuration."""

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    market_data: MarketDataConfig = Field(default_factory=MarketDataConfig)
    instruments: list[InstrumentConfig] = Field(default_factory=list)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    improvement: ImprovementConfig = Field(default_factory=ImprovementConfig)
    news: NewsConfig = Field(default_factory=NewsConfig)

    def get_instrument(self, name: str) -> InstrumentConfig | None:
        """Look up an instrument config by name.

        Args:
            name: Instrument name (e.g. "EUR/USD").

        Returns:
            InstrumentConfig if found, None otherwise.
        """
        for inst in self.instruments:
            if inst.name == name:
                return inst
        return None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    config_path: Path | None = None,
    instrument_overrides: list[str] | None = None,
) -> AppConfig:
    """Load and validate configuration from YAML files.

    Loads default.yml, then optionally merges instrument-specific overrides.
    Environment variables override the database URL (DATABASE_URL).

    Args:
        config_path: Path to the config directory. Defaults to project config/.
        instrument_overrides: List of instrument config filenames to merge.

    Returns:
        Validated AppConfig instance.

    Raises:
        ConfigError: If configuration files are missing or invalid.
    """
    config_dir = config_path or CONFIG_DIR
    default_file = config_dir / "default.yml"

    if not default_file.exists():
        raise ConfigError(f"Default config not found: {default_file}")

    with open(default_file) as f:
        config_data: dict[str, Any] = yaml.safe_load(f) or {}

    # Merge instrument-specific overrides
    if instrument_overrides:
        instruments_dir = config_dir / "instruments"
        for override_file in instrument_overrides:
            override_path = instruments_dir / override_file
            if override_path.exists():
                with open(override_path) as f:
                    override_data: dict[str, Any] = yaml.safe_load(f) or {}
                config_data = _deep_merge(config_data, override_data)

    # Environment variable overrides
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        config_data.setdefault("database", {})["url"] = db_url

    # Broker env overrides
    _env_broker_map = {
        "IG_API_KEY": "api_key",
        "IG_USERNAME": "username",
        "IG_PASSWORD": "password",
        "IG_ACC_NUMBER": "acc_number",
        "IG_ACC_TYPE": "acc_type",
    }
    for env_var, config_key in _env_broker_map.items():
        value = os.environ.get(env_var)
        if value:
            config_data.setdefault("broker", {})[config_key] = value

    # API key env overrides
    _api_key_overrides = [
        ("FINNHUB_API_KEY", "news", "finnhub_api_key"),
        ("ANTHROPIC_API_KEY", "news", "anthropic_api_key"),
    ]
    for api_env, api_section, api_key in _api_key_overrides:
        api_value = os.environ.get(api_env)
        if api_value:
            config_data.setdefault(api_section, {})[api_key] = api_value

    try:
        return AppConfig(**config_data)
    except Exception as e:
        raise ConfigError(f"Invalid configuration: {e}") from e
