"""IG Markets API client wrapper for market data retrieval."""

from __future__ import annotations

import contextlib
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
            start_str = start.strftime("%Y-%m-%dT%H:%M:%S")
            end_str = end.strftime("%Y-%m-%dT%H:%M:%S")

            response = self.service.fetch_historical_prices_by_epic_and_date_range(
                epic=epic,
                resolution=ig_resolution,
                start_date=start_str,
                end_date=end_str,
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
