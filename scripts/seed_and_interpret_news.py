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
    """Fetch historical market news from Finnhub via company-news endpoint.

    Uses ETF proxies to get macro/forex-relevant news:
    SPY→SPX500, GLD→Gold, EWJ→Japan, DIA→Dow, FXE→EUR.

    Fetches in weekly chunks with rate limiting (55 req/min).
    """
    api_key = os.environ.get("FINNHUB_API_KEY", "")
    if not api_key:
        await logger.awarning("no_finnhub_key_skipping")
        return []

    import asyncio
    import time
    from collections import deque

    import aiohttp

    # ETF proxies for our instruments
    etf_tickers = {
        "SPY": "USD",    # S&P 500 → USD/equities
        "GLD": "USD",    # Gold ETF → XAUUSD
        "EWJ": "JPY",    # Japan ETF → NIKKEI/JPY
        "DIA": "USD",    # Dow Jones ETF → DOW30
        "EWG": "EUR",    # Germany ETF → DAX40/EUR
    }

    events: list[NewsEvent] = []
    request_times: deque[float] = deque()

    async with aiohttp.ClientSession() as session:
        for ticker, currency in etf_tickers.items():
            chunk_start = start
            while chunk_start < end:
                chunk_end = min(chunk_start + timedelta(days=7), end)

                # Rate limit: 55 req/min
                now = time.monotonic()
                while request_times and now - request_times[0] > 60:
                    request_times.popleft()
                if len(request_times) >= 55:
                    wait = 60 - (now - request_times[0]) + 0.5
                    if wait > 0:
                        await asyncio.sleep(wait)
                request_times.append(time.monotonic())

                url = (
                    f"https://finnhub.io/api/v1/company-news"
                    f"?symbol={ticker}"
                    f"&from={chunk_start.strftime('%Y-%m-%d')}"
                    f"&to={chunk_end.strftime('%Y-%m-%d')}"
                    f"&token={api_key}"
                )
                try:
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            chunk_start = chunk_end
                            continue
                        data = await resp.json()
                except Exception:
                    chunk_start = chunk_end
                    continue

                for item in data:
                    headline = str(item.get("headline", ""))
                    if not headline:
                        continue

                    ts = item.get("datetime", 0)
                    try:
                        event_time = datetime.fromtimestamp(
                            int(ts), tz=UTC
                        )
                    except (ValueError, TypeError, OSError):
                        continue

                    events.append(NewsEvent(
                        time=event_time,
                        source="finnhub_company",
                        event_type="market_news",
                        title=headline,
                        content=str(item.get("summary", ""))[:500],
                        currency=currency,
                    ))

                await logger.ainfo(
                    "finnhub_chunk",
                    ticker=ticker,
                    start=chunk_start.strftime("%Y-%m-%d"),
                    news=len(data) if isinstance(data, list) else 0,
                )
                chunk_start = chunk_end

        # Also fetch recent general news (last 3 days)
        for category in ["general", "forex"]:
            url = f"https://finnhub.io/api/v1/news?category={category}&token={api_key}"
            try:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for item in data:
                            headline = str(item.get("headline", ""))
                            ts = item.get("datetime", 0)
                            if not headline:
                                continue
                            try:
                                event_time = datetime.fromtimestamp(
                                    int(ts), tz=UTC
                                )
                            except (ValueError, TypeError, OSError):
                                continue
                            if event_time < start:
                                continue
                            events.append(NewsEvent(
                                time=event_time,
                                source="finnhub_news",
                                event_type="market_news",
                                title=headline,
                                content=str(item.get("summary", ""))[:500],
                            ))
            except Exception:
                pass

    # Deduplicate by title (same news from multiple tickers)
    seen_titles: set[str] = set()
    unique_events: list[NewsEvent] = []
    for event in events:
        title_key = event.title[:60].lower()
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_events.append(event)

    await logger.ainfo(
        "finnhub_news_fetched",
        raw=len(events),
        deduplicated=len(unique_events),
    )
    return unique_events


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


_MACRO_KEYWORDS = {
    "fed", "ecb", "boj", "boe", "rate", "inflation", "cpi", "gdp",
    "employment", "nfp", "payroll", "unemployment", "tariff", "trade war",
    "sanction", "recession", "crash", "crisis", "war", "oil", "gold",
    "treasury", "bond", "yield", "dollar", "euro", "yen", "pound",
    "stock market", "wall street", "s&p", "nasdaq", "dow", "dax",
    "nikkei", "forex", "currency", "central bank", "stimulus",
}


def _is_macro_relevant(title: str) -> bool:
    lower = title.lower()
    return any(kw in lower for kw in _MACRO_KEYWORDS)


async def _interpret_one(
    event: NewsEvent,
    interpreter: object,
    instruments: list[str],
) -> NewsEvent:
    """Interpret a single event with the LLM."""
    from src.news.interpreter import NewsInterpreter

    assert isinstance(interpreter, NewsInterpreter)
    analysis = await interpreter.interpret(event, instruments)

    score = float(analysis.get("impact_score", 0))
    if score >= 0.7:
        derived_impact = ImpactLevel.HIGH
    elif score >= 0.4:
        derived_impact = ImpactLevel.MEDIUM
    else:
        derived_impact = ImpactLevel.LOW

    inst_sentiments = analysis.get("instrument_sentiments", {})
    affected = [k for k, v in inst_sentiments.items() if v != "none"]

    return event.model_copy(update={
        "llm_analysis": analysis,
        "impact_level": derived_impact,
        "instruments": affected if affected else event.instruments,
    })


async def interpret_and_store(
    events: list[NewsEvent],
    instruments: list[str],
    store: NewsStore,
    concurrency: int = 10,
) -> int:
    """Interpret events with LLM (parallel) and store immediately.

    Processes events in batches of `concurrency`, writing each batch
    to DB as soon as it's done. No large in-memory accumulation.

    Args:
        events: News events to process.
        instruments: Traded instruments.
        store: DB store for immediate writes.
        concurrency: Number of parallel LLM calls.

    Returns:
        Total events stored.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        await logger.awarning("no_anthropic_key_skipping_interpretation")
        # Store uninterpreted events
        return await store.save_events(events)

    import asyncio

    from anthropic import AsyncAnthropic

    from src.news.interpreter import NewsInterpreter

    client = AsyncAnthropic(api_key=api_key)
    interpreter = NewsInterpreter(client)

    # Split events into: to_interpret (macro-relevant) and skip (store as-is)
    to_interpret: list[NewsEvent] = []
    to_skip: list[NewsEvent] = []

    for event in events:
        already_done = event.llm_analysis and "action" in event.llm_analysis
        irrelevant = event.title and not _is_macro_relevant(event.title)
        if already_done or irrelevant:
            to_skip.append(event)
        else:
            to_interpret.append(event)

    await logger.ainfo(
        "interpretation_plan",
        to_interpret=len(to_interpret),
        skipped=len(to_skip),
    )

    # Store skipped events immediately
    stored = 0
    if to_skip:
        stored += await store.save_events(to_skip)

    # Process interpretable events in parallel batches
    for batch_start in range(0, len(to_interpret), concurrency):
        batch = to_interpret[batch_start : batch_start + concurrency]

        tasks = [
            _interpret_one(event, interpreter, instruments)
            for event in batch
        ]
        interpreted_batch = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions, store successes
        good: list[NewsEvent] = []
        for result in interpreted_batch:
            if isinstance(result, NewsEvent):
                good.append(result)
            else:
                await logger.awarning("interpret_error", error=str(result))

        if good:
            stored += await store.save_events(good)

        await logger.ainfo(
            "batch_complete",
            batch=batch_start // concurrency + 1,
            stored=len(good),
            total_stored=stored,
            remaining=len(to_interpret) - batch_start - concurrency,
        )

    return stored


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

        # 2. Interpret (parallel, 10 at a time) + store (streaming)
        await logger.ainfo("interpreting_and_storing", total=len(all_events))
        count = await interpret_and_store(
            all_events, instruments, store, concurrency=10
        )
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
