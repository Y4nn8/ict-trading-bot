"""IG Markets API client wrapper for market data retrieval."""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import polars as pl
from trading_ig import IGService

from src.common.exceptions import BrokerAuthError, BrokerError, MarketDataError
from src.common.logging import get_logger
from src.market_data.storage import CANDLE_SCHEMA

if TYPE_CHECKING:
    from src.common.config import BrokerConfig

logger = get_logger(__name__)


@dataclass
class MarketSpecs:
    """Contract specifications for an IG instrument."""

    epic: str
    name: str
    instrument_type: str
    value_of_one_pip: float | None
    one_pip_means: float | None
    contract_size: float | None
    lot_size: float
    min_deal_size: float
    min_step_distance: float
    scaling_factor: int
    margin_factor: float
    margin_factor_unit: str


def _parse_currency_value(raw: str | None) -> float | None:
    """Parse IG's currency-prefixed value like '$1' or '€5' into a float."""
    if raw is None:
        return None
    match = re.search(r"[\d.]+", raw)
    return float(match.group()) if match else None


_RESOLUTION_MAP: dict[str, str] = {
    "M5": "5Min",
    "H1": "1Hour",
    "H4": "4Hour",
    "D1": "1Day",
}


class IGClient:
    """Wrapper around the trading_ig library for market data operations.

    Args:
        config: Broker configuration with API credentials.
    """

    def __init__(self, config: BrokerConfig) -> None:
        self._config = config
        self._service: IGService | None = None

    @property
    def service(self) -> IGService:
        """Get the authenticated IG service."""
        if self._service is None:
            raise BrokerError("Not connected. Call connect() first.")
        return self._service

    def connect(self) -> None:
        """Authenticate with IG Markets API."""
        try:
            self._service = IGService(
                username=self._config.username,
                password=self._config.password,
                api_key=self._config.api_key,
                acc_type=self._config.acc_type,
                acc_number=self._config.acc_number,
            )
            self._service.create_session()
            logger.info("ig_connected", acc_type=self._config.acc_type)
        except Exception as e:
            raise BrokerAuthError(f"Failed to connect to IG: {e}") from e

    def disconnect(self) -> None:
        """Close the IG session."""
        if self._service is not None:
            with contextlib.suppress(Exception):
                self._service.logout()
            self._service = None
            logger.info("ig_disconnected")

    def fetch_market_details(self, epic: str) -> MarketSpecs:
        """Fetch contract specifications for an instrument from IG API.

        Uses GET /markets/{epic} (v3) to retrieve dealing rules,
        instrument details, and snapshot data.

        Args:
            epic: IG instrument epic identifier.

        Returns:
            MarketSpecs with contract specifications.

        Raises:
            BrokerError: If the request fails.
        """
        try:
            data = self.service.fetch_market_by_epic(epic)

            instrument = data.get("instrument", {})
            dealing_rules = data.get("dealingRules", {})
            snapshot = data.get("snapshot", {})

            min_deal = dealing_rules.get("minDealSize", {})
            min_step = dealing_rules.get("minStepDistance", {})

            specs = MarketSpecs(
                epic=epic,
                name=instrument.get("name", ""),
                instrument_type=instrument.get("type", ""),
                value_of_one_pip=_parse_currency_value(
                    instrument.get("valueOfOnePip"),
                ),
                one_pip_means=_parse_currency_value(
                    instrument.get("onePipMeans"),
                ),
                contract_size=_parse_currency_value(
                    instrument.get("contractSize"),
                ),
                lot_size=float(instrument.get("lotSize", 1.0)),
                min_deal_size=float(min_deal.get("value", 0.5)),
                min_step_distance=float(min_step.get("value", 0.5)),
                scaling_factor=int(snapshot.get("scalingFactor", 1)),
                margin_factor=float(instrument.get("marginFactor", 0)),
                margin_factor_unit=instrument.get("marginFactorUnit", ""),
            )

            logger.info(
                "market_details_fetched",
                epic=epic,
                name=specs.name,
                value_per_pip=specs.value_of_one_pip,
                min_deal_size=specs.min_deal_size,
                contract_size=specs.contract_size,
            )
            return specs

        except BrokerError:
            raise
        except Exception as e:
            raise BrokerError(
                f"Failed to fetch market details for {epic}: {e}"
            ) from e

    def fetch_historical_candles(
        self,
        epic: str,
        resolution: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Fetch historical OHLCV data from IG Markets.

        Args:
            epic: IG instrument epic identifier.
            resolution: Timeframe string (M5, H1, H4, D1).
            start: Start datetime (UTC).
            end: End datetime (UTC).

        Returns:
            Polars DataFrame with columns: time, open, high, low, close, volume.

        Raises:
            MarketDataError: If data retrieval fails.
        """
        ig_resolution = _RESOLUTION_MAP.get(resolution)
        if ig_resolution is None:
            raise MarketDataError(f"Unsupported resolution: {resolution}")

        try:
            # Use API v1 which passes dates as query params (v2 puts them in URL path)
            start_str = start.strftime("%Y-%m-%dT%H:%M:%S")
            end_str = end.strftime("%Y-%m-%dT%H:%M:%S")

            response = self.service.fetch_historical_prices_by_epic_and_date_range(
                epic=epic,
                resolution=ig_resolution,
                start_date=start_str,
                end_date=end_str,
                version="1",
            )

            prices = response["prices"]
            if prices is None or prices.empty:
                logger.warning("no_data_returned", epic=epic, resolution=resolution)
                return pl.DataFrame(schema=CANDLE_SCHEMA)

            # IG returns multi-level columns (bid/ask/last), use bid prices
            records: list[dict[str, object]] = []
            for idx, row in prices.iterrows():
                records.append({
                    "time": idx.to_pydatetime().replace(tzinfo=UTC)
                    if idx.tzinfo is None
                    else idx.to_pydatetime(),
                    "open": float(row[("bid", "Open")]),
                    "high": float(row[("bid", "High")]),
                    "low": float(row[("bid", "Low")]),
                    "close": float(row[("bid", "Close")]),
                    "volume": float(row.get(("last", "Volume"), 0)),
                })

            df = pl.DataFrame(records)
            logger.info(
                "fetched_candles",
                epic=epic,
                resolution=resolution,
                count=len(df),
            )
            return df

        except BrokerError:
            raise
        except Exception as e:
            raise MarketDataError(
                f"Failed to fetch candles for {epic}/{resolution}: {e}"
            ) from e
