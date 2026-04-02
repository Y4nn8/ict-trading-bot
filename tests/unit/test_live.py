"""Tests for live execution: order manager, portfolio, telegram, monitoring."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.common.models import Direction
from src.execution.order_manager import OrderManager, OrderResult
from src.execution.portfolio import LivePosition, Portfolio
from src.monitoring.health import HealthChecker
from src.monitoring.metrics_exporter import record_trade, update_portfolio_metrics
from src.monitoring.telegram_bot import TelegramNotifier


class TestOrderManager:
    """Tests for IG order execution."""

    @pytest.fixture
    def mock_ig(self) -> MagicMock:
        ig = MagicMock()
        ig.service.create_open_position.return_value = {"dealReference": "ref123"}
        ig.service.fetch_deal_by_deal_reference.return_value = {
            "dealId": "deal123",
            "dealStatus": "ACCEPTED",
            "reason": "",
        }
        ig.service.close_open_position.return_value = {"dealReference": "ref456"}
        ig.service.update_open_position.return_value = {"dealReference": "ref789"}
        return ig

    def test_open_position(self, mock_ig: MagicMock) -> None:
        mgr = OrderManager(mock_ig)
        result = mgr.open_position(
            epic="CS.D.EURUSD.CFD.IP",
            direction=Direction.LONG,
            size=1.0,
            stop_loss=1.077,
            take_profit=1.086,
        )
        assert result.deal_id == "deal123"
        assert result.status == "ACCEPTED"

    def test_close_position(self, mock_ig: MagicMock) -> None:
        mgr = OrderManager(mock_ig)
        result = mgr.close_position(
            deal_id="deal123",
            direction=Direction.LONG,
            size=1.0,
            epic="CS.D.EURUSD.CFD.IP",
        )
        assert isinstance(result, OrderResult)

    def test_update_stop_loss(self, mock_ig: MagicMock) -> None:
        mgr = OrderManager(mock_ig)
        result = mgr.update_stop_loss("deal123", 1.079)
        assert isinstance(result, OrderResult)


class TestPortfolio:
    """Tests for portfolio tracking."""

    def test_add_and_close_position(self) -> None:
        portfolio = Portfolio(initial_capital=10000)
        pos = LivePosition(
            deal_id="d1",
            epic="CS.D.EURUSD.CFD.IP",
            instrument="EUR/USD",
            direction=Direction.LONG,
            size=1.0,
            entry_price=1.0800,
        )
        portfolio.add_position(pos)
        assert portfolio.position_count == 1

        pnl = portfolio.close_position("d1", exit_price=1.0850)
        assert pnl == pytest.approx(0.005)
        assert portfolio.position_count == 0
        assert portfolio.capital == pytest.approx(10000.005)

    def test_close_unknown_position(self) -> None:
        portfolio = Portfolio()
        pnl = portfolio.close_position("unknown", 1.08)
        assert pnl == 0.0

    def test_update_prices(self) -> None:
        portfolio = Portfolio()
        pos = LivePosition(
            deal_id="d1",
            epic="EPIC1",
            instrument="EUR/USD",
            direction=Direction.SHORT,
            size=1.0,
            entry_price=1.0800,
        )
        portfolio.add_position(pos)
        portfolio.update_prices({"EPIC1": 1.0750})
        assert pos.unrealized_pnl == pytest.approx(0.005)

    def test_get_summary(self) -> None:
        portfolio = Portfolio(initial_capital=10000)
        summary = portfolio.get_summary()
        assert summary["capital"] == 10000
        assert summary["open_positions"] == 0

    def test_equity_includes_unrealized(self) -> None:
        portfolio = Portfolio(initial_capital=10000)
        pos = LivePosition(
            deal_id="d1",
            epic="E1",
            instrument="X",
            direction=Direction.LONG,
            size=100.0,
            entry_price=1.08,
        )
        portfolio.add_position(pos)
        portfolio.update_prices({"E1": 1.09})
        assert portfolio.equity > portfolio.capital


class TestTelegramNotifier:
    """Tests for Telegram notifications."""

    async def test_notify_trade_opened(self) -> None:
        notifier = TelegramNotifier("token", "chat_id")
        notifier._bot = AsyncMock()
        await notifier.notify_trade_opened({
            "instrument": "EUR/USD",
            "direction": "LONG",
            "entry_price": 1.08,
        })
        notifier._bot.send_message.assert_called_once()

    async def test_notify_trade_closed(self) -> None:
        notifier = TelegramNotifier("token", "chat_id")
        notifier._bot = AsyncMock()
        await notifier.notify_trade_closed({"pnl": 50.0, "instrument": "EUR/USD"})
        notifier._bot.send_message.assert_called_once()

    async def test_send_status(self) -> None:
        notifier = TelegramNotifier("token", "chat_id")
        notifier._bot = AsyncMock()
        await notifier.send_status({"capital": 10000, "equity": 10050, "open_positions": 1})
        notifier._bot.send_message.assert_called_once()


class TestHealthChecker:
    """Tests for health monitoring."""

    def test_healthy_by_default(self) -> None:
        hc = HealthChecker()
        assert hc.is_healthy()

    def test_unhealthy_component(self) -> None:
        hc = HealthChecker()
        hc.register_component("database")
        assert not hc.is_healthy()
        hc.update_health("database", healthy=True)
        assert hc.is_healthy()

    def test_get_status(self) -> None:
        hc = HealthChecker()
        hc.register_component("broker")
        hc.update_health("broker", healthy=True, details="Connected")
        status = hc.get_status()
        assert status["components"]["broker"]["healthy"]

    def test_record_candle(self) -> None:
        hc = HealthChecker()
        now = datetime.now(tz=UTC)
        hc.record_candle_processed(now)
        status = hc.get_status()
        assert status["last_candle_time"] is not None


class TestMetricsExporter:
    """Tests for Prometheus metrics."""

    def test_record_trade(self) -> None:
        # Should not raise
        record_trade("EUR/USD", "LONG", 50.0)
        record_trade("EUR/USD", "SHORT", -20.0)

    def test_update_portfolio_metrics(self) -> None:
        update_portfolio_metrics(10000, 10050, 2, 50.0)
