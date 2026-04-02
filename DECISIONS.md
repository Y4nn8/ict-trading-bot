# Decision Journal

## Session 1 — 2026-04-01

### Progress
- Steps completed: [1, 2, 3, 4, 5, 6]
- Current phase: 1 (Foundations) — COMPLETE
- Next step: Phase 2, step 7 (swings.py)
- Active branch: feat/phase1-foundations (PR merged)

### Decisions Made
- **Branch strategy for Phase 1**: single branch `feat/phase1-foundations` for steps 1-6 as they are highly interdependent
- **trading-ig version**: unpinned (latest is 0.0.23, no 0.0.24 exists despite CLAUDE.md suggestion)
- **Pydantic datetime import**: kept at runtime (not in TYPE_CHECKING) because Pydantic needs it for model resolution
- **asyncpg stubs**: added to mypy ignore list (no py.typed marker available)
- **CI test scope**: split into 3 jobs (lint, test, integration) per Copilot review feedback
- **Continuous aggregates**: created in init_db.sql with auto-refresh policies (5min for H1/H4, 15min for D1)
- **pandas as dev dep**: needed for mocking IG API responses in tests (IG returns pandas DataFrames)
- **Batch DB inserts**: switched from row-by-row to `executemany` per review

### Issues Encountered
- Ruff TCH003 vs Pydantic: `noqa: TC003` for datetime import
- structlog `get_logger()`: `type: ignore[no-any-return]`

## Session 2 — 2026-04-01

### Progress
- Steps completed: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
- Current phase: 2 (ICT Detectors) — COMPLETE
- Next step: Phase 3, step 18 (backtest vectorized pre-computation)
- Active branch: feat/phase2-ict-detectors (PR pending)

### Decisions Made
- **Dual implementation pattern**: all detectors have vectorized (Polars/NumPy on full DF) + incremental (candle-by-candle with state) versions
- **Session non-overlap**: changed session ranges to non-overlapping (Asian 0-7, London 7-12, NY 12-21 UTC) to avoid ambiguous classification
- **OTE zone boundary**: `is_in_ote` uses min/max of ote_high/ote_low since 62% and 79% retracement create counter-intuitive naming
- **Swing detection**: Williams 5-bar fractal (left=2, right=2) as default, configurable
- **Order Block detection**: requires displacement (body > 2x ATR) after opposing candle; ATR period=14
- **Liquidity detection**: equal highs/lows within 0.02% tolerance over 50-bar lookback
- **MarketStructureState**: auto-registers unknown instrument/timeframe pairs on first candle

### Issues Encountered
- Polars schema typing: `dict[str, Any]` needed for mixed schema types (Datetime instance + Float64 class)
- Fixture datetime generation: `datetime(h, m)` fails when minute >= 60; switched to `timedelta` addition

### TODO / Open Items
- Phase 4: News module

## Session 3 — 2026-04-02

### Progress
- Steps completed: [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
- Current phase: 3 (Backtest + Strategy) — COMPLETE
- Next step: Phase 4, step 31 (news base)
- Active branch: feat/phase3-backtest-strategy (PR pending)

### Decisions Made
- **Backtest architecture**: two-phase as specified — vectorized pre-computation then M5 event loop
- **Confluence scoring**: 6 weighted factors (FVG 0.15, OB 0.20, MS 0.25, displacement 0.10, killzone 0.15, P/D 0.15)
- **Entry evaluation**: requires both minimum confluence score AND a market structure break for direction
- **Position sizing tiers**: low (<0.4) → 0.5% risk, medium (0.4-0.7) → 1%, high (>0.7) → 2%
- **ATR proxy for SL**: using candle range (high-low) as simple ATR estimate for entry signals, actual ATR in order blocks/displacement
- **Walk-forward**: window generation uses 28th-of-month ceiling to avoid month-end issues
- **Engine coverage**: engine.run() at 26% coverage — needs end-to-end integration test with real fixture data (deferred to step 30)

### Issues Encountered
- ruff auto-fix removed `close` variable as unused but left dangling `float(candle["close"])` expression — manually cleaned up

## Session 4 — 2026-04-02

### Progress
- Steps completed: [31, 32, 33, 34, 35, 36, 37, 38, 39]
- Current phase: 4 (News) — COMPLETE
- Next step: Phase 5, step 41 (live demo)
- Active branch: feat/phase4-news (PR pending)

### Decisions Made
- **NewsSource ABC**: adapter pattern — adding a source = implementing the interface
- **LLM interpreter**: structured prompt with ACTION/SENTIMENT/IMPACT_SCORE/AFFECTED/REASONING format, parsed line-by-line
- **Event actions**: 4 types — none, pause (pre/post event window), tighten_stops, trigger_entry
- **Newsfilter split**: separate realtime (WebSocket) and historical (Query API) classes
- **News store**: events queryable by time range with tolerance for backtest replay alignment

### Issues Encountered
- Anthropic SDK content block types: `response.content[0].text` fails mypy due to union with ThinkingBlock etc. — used `hasattr` check
