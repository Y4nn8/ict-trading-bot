"""Tests for news module: adapters, interpreter, event manager, store."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.common.models import ImpactLevel, NewsEvent
from src.news.base import NewsSource
from src.news.calendar.finnhub import FinnhubCalendarSource, _map_impact
from src.news.event_manager import EventManager
from src.news.interpreter import NewsAction, NewsInterpreter
from src.news.store import NewsStore


class TestNewsSourceABC:
    """Tests for NewsSource interface."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            NewsSource()  # type: ignore[abstract]


class TestFinnhubAdapter:
    """Tests for Finnhub calendar adapter."""

    def test_map_impact(self) -> None:
        assert _map_impact("1") == ImpactLevel.LOW
        assert _map_impact("2") == ImpactLevel.MEDIUM
        assert _map_impact("3") == ImpactLevel.HIGH
        assert _map_impact(None) is None
        assert _map_impact("unknown") is None

    async def test_connect_creates_session(self) -> None:
        source = FinnhubCalendarSource(api_key="test")
        await source.connect()
        assert source._session is not None
        await source.disconnect()
        assert source._session is None


class TestNewsInterpreter:
    """Tests for LLM news interpreter."""

    def test_parse_response(self) -> None:
        interpreter = NewsInterpreter(client=MagicMock())
        text = """ACTION: pause
SENTIMENT: bearish
IMPACT_SCORE: 0.8
AFFECTED: EUR/USD, GBP/USD
REASONING: NFP much higher than expected"""
        result = interpreter._parse_response(text)
        assert result["action"] == NewsAction.PAUSE
        assert result["sentiment"] == "bearish"
        assert result["impact_score"] == 0.8
        assert "EUR/USD" in result["affected_instruments"]
        assert result["reasoning"] == "NFP much higher than expected"

    def test_parse_invalid_action(self) -> None:
        interpreter = NewsInterpreter(client=MagicMock())
        text = "ACTION: invalid_action\nSENTIMENT: neutral"
        result = interpreter._parse_response(text)
        assert result["action"] == NewsAction.NONE

    def test_parse_empty_response(self) -> None:
        interpreter = NewsInterpreter(client=MagicMock())
        result = interpreter._parse_response("")
        assert result["action"] == NewsAction.NONE
        assert result["impact_score"] == 0.0

    async def test_interpret_failure_returns_default(self) -> None:
        mock_client = AsyncMock()
        mock_client.messages.create.side_effect = Exception("API error")
        interpreter = NewsInterpreter(client=mock_client)
        event = NewsEvent(
            time=datetime(2024, 1, 15, tzinfo=UTC),
            source="test",
            event_type="test",
            title="Test Event",
        )
        result = await interpreter.interpret(event, ["EUR/USD"])
        assert result["action"] == NewsAction.NONE


class TestEventManager:
    """Tests for event action manager."""

    def test_pause_action(self) -> None:
        mgr = EventManager(pre_event_pause_minutes=30, post_event_resume_minutes=15)
        event_time = datetime(2024, 1, 15, 13, 30, tzinfo=UTC)
        mgr.apply_action(NewsAction.PAUSE, event_time, {"reasoning": "NFP"})

        # Before pause end: paused
        assert mgr.is_paused(event_time + timedelta(minutes=5))
        # After pause end: not paused
        assert not mgr.is_paused(event_time + timedelta(minutes=20))

    def test_tighten_stops_action(self) -> None:
        mgr = EventManager(post_event_resume_minutes=15)
        event_time = datetime(2024, 1, 15, 13, 30, tzinfo=UTC)
        mgr.apply_action(NewsAction.TIGHTEN_STOPS, event_time, {})

        assert mgr.should_tighten_stops(event_time + timedelta(minutes=5))
        assert not mgr.should_tighten_stops(event_time + timedelta(minutes=20))

    def test_trigger_entry_action(self) -> None:
        mgr = EventManager()
        mgr.apply_action(
            NewsAction.TRIGGER_ENTRY,
            datetime(2024, 1, 15, tzinfo=UTC),
            {"sentiment": "bullish", "affected_instruments": ["EUR/USD"]},
        )
        triggers = mgr.pop_triggers()
        assert len(triggers) == 1
        assert triggers[0]["sentiment"] == "bullish"
        # Pop clears the list
        assert mgr.pop_triggers() == []

    def test_none_action(self) -> None:
        mgr = EventManager()
        mgr.apply_action(NewsAction.NONE, datetime(2024, 1, 15, tzinfo=UTC), {})
        assert not mgr.is_paused(datetime(2024, 1, 15, tzinfo=UTC))

    def test_not_paused_by_default(self) -> None:
        mgr = EventManager()
        assert not mgr.is_paused(datetime(2024, 1, 15, tzinfo=UTC))
        assert not mgr.should_tighten_stops(datetime(2024, 1, 15, tzinfo=UTC))


class TestNewsStore:
    """Tests for news database storage (mocked DB)."""

    @pytest.fixture
    def mock_db(self) -> AsyncMock:
        db = AsyncMock()
        db.execute = AsyncMock(return_value="INSERT")
        db.fetch = AsyncMock(return_value=[])
        return db

    @pytest.fixture
    def store(self, mock_db: AsyncMock) -> NewsStore:
        return NewsStore(mock_db)

    async def test_save_event(self, store: NewsStore) -> None:
        event = NewsEvent(
            time=datetime(2024, 1, 15, 13, 30, tzinfo=UTC),
            source="finnhub",
            event_type="nfp",
            title="Non-Farm Payrolls",
            currency="USD",
            impact_level=ImpactLevel.HIGH,
        )
        await store.save_event(event)
        store._db.execute.assert_called_once()

    async def test_save_events_returns_count(self, store: NewsStore) -> None:
        events = [
            NewsEvent(
                time=datetime(2024, 1, 15, tzinfo=UTC),
                source="test",
                event_type="test",
            ),
            NewsEvent(
                time=datetime(2024, 1, 16, tzinfo=UTC),
                source="test",
                event_type="test",
            ),
        ]
        count = await store.save_events(events)
        assert count == 2

    async def test_get_events_empty(self, store: NewsStore) -> None:
        result = await store.get_events(
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 31, tzinfo=UTC),
        )
        assert result == []
