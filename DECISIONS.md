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
- Phase 3: Backtest + Strategy (steps 18-30)
