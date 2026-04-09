"""Tests for IG Markets client wrapper."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest

from src.common.config import BrokerConfig
from src.common.exceptions import BrokerError, MarketDataError
from src.market_data.ig_client import IGClient, _parse_currency_value


@pytest.fixture
def broker_config() -> BrokerConfig:
    return BrokerConfig(
        api_key="test_key",
        username="test_user",
        password="test_pass",
        acc_number="test_acc",
        acc_type="DEMO",
    )


@pytest.fixture
def ig_client(broker_config: BrokerConfig) -> IGClient:
    return IGClient(broker_config)


class TestIGClient:
    """Tests for IGClient."""

    def test_service_raises_when_not_connected(self, ig_client: IGClient) -> None:
        with pytest.raises(BrokerError, match="Not connected"):
            _ = ig_client.service

    def test_unsupported_resolution_raises(self, ig_client: IGClient) -> None:
        ig_client._service = MagicMock()
        with pytest.raises(MarketDataError, match="Unsupported resolution"):
            ig_client.fetch_historical_candles(
                epic="CS.D.EURUSD.CFD.IP",
                resolution="M2",
                start=datetime(2024, 1, 1, tzinfo=UTC),
                end=datetime(2024, 1, 2, tzinfo=UTC),
            )

    @patch("src.market_data.ig_client.IGService")
    def test_connect_creates_session(
        self, mock_ig_service_cls: MagicMock, ig_client: IGClient
    ) -> None:
        mock_instance = MagicMock()
        mock_ig_service_cls.return_value = mock_instance
        ig_client.connect()
        mock_instance.create_session.assert_called_once()

    def test_fetch_returns_empty_df_when_no_data(self, ig_client: IGClient) -> None:
        mock_service = MagicMock()
        mock_service.fetch_historical_prices_by_epic_and_date_range.return_value = {
            "prices": pd.DataFrame()
        }
        ig_client._service = mock_service

        result = ig_client.fetch_historical_candles(
            epic="CS.D.EURUSD.CFD.IP",
            resolution="M5",
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 2, tzinfo=UTC),
        )
        assert isinstance(result, pl.DataFrame)
        assert result.is_empty()

    def test_fetch_parses_ig_response(self, ig_client: IGClient) -> None:
        # Build a mock IG response with multi-level columns
        index = pd.DatetimeIndex([
            pd.Timestamp("2024-01-15 10:00:00", tz="UTC"),
            pd.Timestamp("2024-01-15 10:05:00", tz="UTC"),
        ])
        columns = pd.MultiIndex.from_tuples([
            ("bid", "Open"), ("bid", "High"), ("bid", "Low"), ("bid", "Close"),
            ("last", "Volume"),
        ])
        data = [
            [1.0800, 1.0810, 1.0795, 1.0805, 150.0],
            [1.0805, 1.0815, 1.0800, 1.0812, 120.0],
        ]
        prices_df = pd.DataFrame(data, index=index, columns=columns)

        mock_service = MagicMock()
        mock_service.fetch_historical_prices_by_epic_and_date_range.return_value = {
            "prices": prices_df
        }
        ig_client._service = mock_service

        result = ig_client.fetch_historical_candles(
            epic="CS.D.EURUSD.CFD.IP",
            resolution="M5",
            start=datetime(2024, 1, 15, 10, 0, tzinfo=UTC),
            end=datetime(2024, 1, 15, 10, 10, tzinfo=UTC),
        )

        assert len(result) == 2
        assert result["open"][0] == 1.0800
        assert result["high"][1] == 1.0815
        assert result["volume"][0] == 150.0

    def test_disconnect_clears_service(self, ig_client: IGClient) -> None:
        ig_client._service = MagicMock()
        ig_client.disconnect()
        assert ig_client._service is None

    def test_fetch_market_details(self, ig_client: IGClient) -> None:
        mock_service = MagicMock()
        mock_service.fetch_market_by_epic.return_value = {
            "instrument": {
                "name": "EUR/USD",
                "type": "CURRENCIES",
                "valueOfOnePip": "$1",
                "onePipMeans": "0.0001",
                "contractSize": "$100000",
                "lotSize": 1.0,
                "marginFactor": 3.33,
                "marginFactorUnit": "PERCENTAGE",
            },
            "dealingRules": {
                "minDealSize": {"value": 0.5},
                "minStepDistance": {"value": 0.1},
            },
            "snapshot": {"scalingFactor": 5},
        }
        ig_client._service = mock_service

        specs = ig_client.fetch_market_details("CS.D.EURUSD.CFD.IP")
        assert specs.epic == "CS.D.EURUSD.CFD.IP"
        assert specs.name == "EUR/USD"
        assert specs.value_of_one_pip == 1.0
        assert specs.one_pip_means == 0.0001
        assert specs.contract_size == 100000.0
        assert specs.min_deal_size == 0.5
        assert specs.min_step_distance == 0.1
        assert specs.scaling_factor == 5
        assert specs.margin_factor == 3.33

    def test_fetch_market_details_error(self, ig_client: IGClient) -> None:
        mock_service = MagicMock()
        mock_service.fetch_market_by_epic.side_effect = Exception("API error")
        ig_client._service = mock_service

        with pytest.raises(BrokerError, match="Failed to fetch market details"):
            ig_client.fetch_market_details("INVALID_EPIC")


class TestParseCurrencyValue:
    """Tests for _parse_currency_value helper."""

    def test_dollar_prefix(self) -> None:
        assert _parse_currency_value("$1") == 1.0

    def test_euro_prefix(self) -> None:
        assert _parse_currency_value("€5.50") == 5.50

    def test_plain_number(self) -> None:
        assert _parse_currency_value("0.0001") == 0.0001

    def test_none_returns_none(self) -> None:
        assert _parse_currency_value(None) is None

    def test_no_number_returns_none(self) -> None:
        assert _parse_currency_value("USD") is None
