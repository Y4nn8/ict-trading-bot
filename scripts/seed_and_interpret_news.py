"""Fetch real historical news, interpret with Claude Haiku, store in DB.

Sources:
- Finnhub: economic calendar (rate-limited, requires API key)
- GDELT: breaking news (free, no API key)

Interpretation is done once per event and cached in DB.

Usage:
    uv run python -m scripts.seed_and_interpret_news --weeks 9
"""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import UTC, datetime, timedelta

from dotenv import load_dotenv

from src.common.config import load_config
from src.common.db import Database
from src.common.logging import get_logger, setup_logging
from src.common.models import ImpactLevel, NewsEvent
from src.news.store import NewsStore

load_dotenv()

logger = get_logger(__name__)


async def fetch_finnhub_news(
    start: datetime, end: datetime
) -> list[NewsEvent]:
    """Fetch general market news from Finnhub (free tier).

    The economic calendar endpoint requires a paid plan,
    but general/forex news is free.
    """
    api_key = os.environ.get("FINNHUB_API_KEY", "")
    if not api_key:
        await logger.awarning("no_finnhub_key_skipping")
        return []

    import aiohttp

    events: list[NewsEvent] = []
    async with aiohttp.ClientSession() as session:
        for category in ["general", "forex"]:
            url = f"https://finnhub.io/api/v1/news?category={category}&token={api_key}"
            async with session.get(url) as resp:
                if resp.status != 200:
                    continue
                data = await resp.json()

            for item in data:
                headline = str(item.get("headline", ""))
                if not headline:
                    continue

                from datetime import UTC

                ts = item.get("datetime", 0)
                try:
                    event_time = datetime.fromtimestamp(int(ts), tz=UTC)
                except (ValueError, TypeError, OSError):
                    continue

                if event_time < start or event_time > end:
                    continue

                events.append(NewsEvent(
                    time=event_time,
                    source="finnhub_news",
                    event_type="market_news",
                    title=headline,
                    content=str(item.get("summary", "")),
                    # impact_level set after LLM interpretation
                ))

    await logger.ainfo("finnhub_news_fetched", count=len(events))
    return events


async def fetch_gdelt_events(
    start: datetime, end: datetime
) -> list[NewsEvent]:
    """Fetch breaking news from GDELT."""
    from src.news.realtime.gdelt import GDELTNewsSource

    source = GDELTNewsSource(max_articles_per_query=30)
    await source.connect()
    try:
        events = await source.get_events_chunked(start, end, chunk_days=7)
        return events
    finally:
        await source.disconnect()


async def interpret_events(
    events: list[NewsEvent],
    instruments: list[str],
) -> list[NewsEvent]:
    """Run Claude Haiku interpretation on High/Medium impact events."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        await logger.awarning("no_anthropic_key_skipping_interpretation")
        return events

    from anthropic import AsyncAnthropic

    from src.news.interpreter import NewsInterpreter

    client = AsyncAnthropic(api_key=api_key)
    interpreter = NewsInterpreter(client)

    interpreted = []
    for event in events:
        # Skip events already interpreted (e.g. GDELT with tone)
        if event.llm_analysis and "action" in event.llm_analysis:
            interpreted.append(event)
            continue

        # Skip LOW impact events that already have a level set
        if event.impact_level == ImpactLevel.LOW:
            interpreted.append(event)
            continue

        analysis = await interpreter.interpret(event, instruments)

        # Derive impact_level from LLM impact_score
        score = float(analysis.get("impact_score", 0))
        if score >= 0.7:
            derived_impact = ImpactLevel.HIGH
        elif score >= 0.4:
            derived_impact = ImpactLevel.MEDIUM
        else:
            derived_impact = ImpactLevel.LOW

        # Derive instruments list from LLM instrument_sentiments
        inst_sentiments = analysis.get("instrument_sentiments", {})
        affected = [k for k, v in inst_sentiments.items() if v != "none"]

        interpreted.append(
            event.model_copy(update={
                "llm_analysis": analysis,
                "impact_level": derived_impact,
                "instruments": affected if affected else event.instruments,
            })
        )
        await logger.ainfo(
            "event_interpreted",
            title=event.title,
            action=analysis.get("action", "none"),
            impact=derived_impact,
            sentiments=analysis.get("instrument_sentiments", {}),
        )

    return interpreted


async def seed_news(weeks: int) -> None:
    """Fetch from Finnhub + GDELT, interpret, store."""
    config = load_config()
    setup_logging(config.logging.level, json_format=False)

    db = Database(config.database)
    await db.connect()
    store = NewsStore(db)

    instruments = [ic.name for ic in config.instruments]
    now = datetime.now(tz=UTC)
    start = now - timedelta(weeks=weeks)

    try:
        # 1. Fetch from both sources
        await logger.ainfo("fetching_finnhub_news", weeks=weeks)
        finnhub_events = await fetch_finnhub_news(start, now)

        await logger.ainfo("fetching_gdelt", weeks=weeks)
        gdelt_events = await fetch_gdelt_events(start, now)

        all_events = finnhub_events + gdelt_events
        await logger.ainfo(
            "events_fetched",
            finnhub=len(finnhub_events),
            gdelt=len(gdelt_events),
            total=len(all_events),
        )

        if not all_events:
            await logger.awarning("no_events_found")
            return

        # 2. Interpret with LLM
        await logger.ainfo("interpreting_events")
        interpreted = await interpret_events(all_events, instruments)

        # 3. Store
        count = await store.save_events(interpreted)
        await logger.ainfo("news_seed_complete", stored=count)

    finally:
        await db.disconnect()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch and interpret real news for backtesting"
    )
    parser.add_argument(
        "--weeks", type=int, default=9,
        help="Number of weeks of history (default: 9)",
    )
    args = parser.parse_args()
    asyncio.run(seed_news(args.weeks))


if __name__ == "__main__":
    main()
