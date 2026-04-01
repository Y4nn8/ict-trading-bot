# Decision Journal

## Session 1 — 2026-04-01

### Progress
- Steps completed: [1, 2, 3, 4, 5, 6]
- Current phase: 1 (Foundations) — COMPLETE
- Next step: Phase 2, step 7 (swings.py)
- Active branch: feat/phase1-foundations (PR pending)

### Decisions Made
- **Branch strategy for Phase 1**: single branch `feat/phase1-foundations` for steps 1-6 as they are highly interdependent
- **trading-ig version**: unpinned (latest is 0.0.23, no 0.0.24 exists despite CLAUDE.md suggestion)
- **Pydantic datetime import**: kept at runtime (not in TYPE_CHECKING) because Pydantic needs it for model resolution
- **asyncpg stubs**: added to mypy ignore list (no py.typed marker available)
- **CI test scope**: unit tests only in CI (`-m "not integration"`), integration tests require live DB
- **Continuous aggregates**: created in init_db.sql with auto-refresh policies (5min for H1/H4, 15min for D1)
- **pandas as dev dep**: needed for mocking IG API responses in tests (IG returns pandas DataFrames)
- **Coverage strategy**: 85% achieved on Phase 1, db.py lower coverage (42%) since async DB methods need integration tests

### Issues Encountered
- Ruff TCH003 vs Pydantic: moving `datetime` to TYPE_CHECKING block breaks Pydantic model resolution. Fixed with `noqa: TC003`
- structlog `get_logger()` returns `BoundLoggerLazyProxy`, not `BoundLogger`. Used `type: ignore[no-any-return]`

### TODO / Open Items
- Configure branch protection rules on master after first PR merge
- Phase 2: ICT detectors (steps 7-17)
