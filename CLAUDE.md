# CLAUDE.md — ICT/SMC Trading Bot

## Project Overview

Automated multi-asset trading bot using ICT (Inner Circle Trader) / SMC (Smart Money Concepts) for market structure analysis, enriched with AI-interpreted news for event trading. Includes a dual self-improvement loop (Optuna daily + LLM weekly).

**Markets**: CFDs via IG Markets — Forex (majors/minors), Indices (NAS100, SPX, DAX), Commodities (Gold, Oil). No crypto. ESMA leverage limits apply (30:1 forex, 20:1 indices, 10:1 commodities).

## Tech Stack

- **Language**: Python 3.12+
- **Dependency management**: uv (pyproject.toml + uv.lock)
- **Database**: PostgreSQL 16 + TimescaleDB (Docker)
- **Broker API**: IG Markets via `trading_ig` library
- **LLM**: Claude Haiku 4.5 (news interpretation), Claude Sonnet 4.6 (improvement loop) via `anthropic` SDK
- **ML optimization**: Optuna
- **Vectorized compute**: Polars (preferred over Pandas for new code), NumPy
- **Testing**: pytest + coverage (target: >90% on `src/structure/`, >80% global)
- **CI**: GitHub Actions (lint with ruff, type-check with mypy, tests, coverage)
- **Containers**: Docker Compose (PostgreSQL/TimescaleDB + bot + Grafana)
- **Monitoring**: Telegram bot (alerts) + Grafana (dashboards)
- **News sources**: Finnhub REST (economic calendar), newsfilter.io WebSocket (breaking news), FF News API (historical calendar), newsfilter.io Query API (historical news)

## Code Conventions

- Type hints everywhere. Use `from __future__ import annotations`.
- Pydantic v2 for all data models (in `src/common/models.py`). Use `model_validator` over ad-hoc validation.
- Async by default for I/O-bound code (aiohttp, asyncpg, websockets). Sync OK for pure computation.
- Structured logging with `structlog` (JSON format). No print statements.
- Config via YAML files in `config/`, loaded and validated with Pydantic in `src/common/config.py`.
- All configurable parameters must have defaults in `config/default.yml` and be overridable per instrument in `config/instruments/<n>.yml`.
- Docstrings: Google style, on all public functions/classes.
- No global state. Dependency injection via constructor parameters.
- Error handling: custom exceptions in `src/common/exceptions.py`. Never catch bare `Exception`.
- Imports: absolute imports only (`from src.structure.fvg import ...`), never relative.

## Git Workflow

### Branching Strategy

- **`master`**: stable, always functional, protected. Never commit or merge directly into `master` without a PR.
- **Feature branches**: one branch per step or cohesive group of steps. Naming: `feat/<module>-<description>`. All branches fork from `master` and merge back into `master` via Pull Request.
- No `develop` branch. Quality in `master` is guaranteed by PRs + CI + reviews.

### Tools

Use the `gh` CLI (GitHub CLI) for all remote git operations: push, PR creation, merge. The repo must be created on GitHub at the start of Phase 1.

### Per-Step Workflow

For each implementation step:

1. **Create a branch** from `master`:
   ```bash
   git checkout master && git pull
   git checkout -b feat/structure-fvg-detector
   ```

2. **Implement** code + tests. Atomic and frequent commits (conventional commits).

3. **Verify locally** that everything passes:
   ```bash
   uv run pytest                    # all tests, not just new ones
   uv run ruff check src/ tests/    # lint
   uv run mypy src/                 # type-check
   ```

4. **Internal review** — run `/review` in Claude Code. Fix any issues raised before proceeding.

5. **Push and create the PR**:
   ```bash
   git push -u origin feat/structure-fvg-detector
   gh pr create --base master --title "feat(structure): add FVG detector" --body "..."
   ```

6. **Request a Copilot review** on the PR:
   ```bash
   gh pr edit <PR_NUMBER> --add-reviewer @copilot
   ```

7. **Wait for CI to pass** (GitHub Actions: tests + lint + coverage).

8. **Address feedback** use /fix-pr-comments command to loop with copilot comments

9. **Merge the PR**:
   ```bash
   gh pr merge <PR_NUMBER> --squash --delete-branch
   ```

10. **Move to the next step** from an up-to-date `master`.

### Commit Convention

Format: `<type>(<scope>): <description>`

Types:
- `feat`: new feature
- `test`: add or modify tests
- `fix`: bug fix
- `refactor`: refactoring with no behavior change
- `chore`: config, CI, dependencies, docker
- `docs`: documentation

Scope = module name (`common`, `structure`, `backtest`, `news`, `execution`, `improvement`, `monitoring`).

Examples:
- `feat(structure): add FVG vectorized detector`
- `test(structure): add annotated fixtures for BOS/CHoCH detection`
- `chore(docker): add TimescaleDB continuous aggregates init script`
- `fix(backtest): prevent look-ahead bias in D1 candle aggregation`

### Branch Granularity

- **One branch per module** when the module is independent (e.g. `feat/structure-fvg-detector`, `feat/structure-order-blocks`)
- **One branch per phase** when steps are highly interdependent (e.g. `feat/phase1-foundations` for steps 1-6)
- When in doubt, prefer smaller branches.

## Decision Journal (`DECISIONS.md`)

Maintain a `DECISIONS.md` file at the project root. This file is **mandatory** and must be updated every work session. It serves as cross-session memory: every new Claude Code session must read it first to know the project state and past decisions.

### Format

```markdown
# Decision Journal

## Session N — YYYY-MM-DD

### Progress
- Steps completed: [list by number]
- Current phase: X
- Next step: Y
- Active branch: feat/...

### Decisions Made
- **[Topic]**: [Decision + rationale in 1-2 lines]

### Issues Encountered
- [Problem description + solution applied]

### TODO / Open Items
- [Items to address in future sessions]
```

### Rules

- **When to write**: at the end of each session, or after any significant decision. Commit DECISIONS.md with the message `docs: update decision journal`.
- **What to document**:
  - Any architectural choice not covered by CLAUDE.md (e.g. "chose asyncpg over psycopg3 because...")
  - Any deviation from the original plan (e.g. "merged swings.py and market_structure.py because...")
  - Any technical blocker and the solution chosen
  - Dependencies added that were not planned and why
  - Default parameter values chosen for configs and their rationale
  - Significant backtest results
- **What NOT to document**: obvious implementation details, cosmetic changes.
- **Length**: concise. 5-15 lines per session. This is a journal, not a report.

### Session Startup Procedure

Every new work session must begin with:
1. Read `CLAUDE.md` (plan and conventions)
2. Read `DECISIONS.md` (current state and decision history)
3. Check the active git branch and latest commits
4. Resume from the next uncompleted step

## Project Structure

```
ict-trading-bot/
├── CLAUDE.md
├── pyproject.toml
├── docker-compose.yml
├── Dockerfile
├── config/
│   ├── default.yml                # All configurable params with defaults
│   ├── improvement.yml            # Improvement loop params
│   └── instruments/               # Per-instrument overrides
├── src/
│   ├── common/                    # Models, config, DB, logging, exceptions
│   ├── market_data/               # IG client, ingestion, TimescaleDB storage, TF aggregation
│   ├── structure/                 # ICT/SMC detectors (core)
│   ├── strategy/                  # Confluence scoring, entry/exit, filters
│   ├── news/                      # News ingestion (adapter pattern), LLM interpretation
│   ├── execution/                 # Position sizing, order management, risk/circuit breakers
│   ├── backtest/                  # Hybrid engine (vectorized pre-compute + event loop)
│   ├── improvement/               # Optuna optimizer + LLM analyzer + patch/version management
│   └── monitoring/                # Telegram bot, Grafana metrics export, health checks
├── tests/
│   ├── conftest.py
│   ├── fixtures/                  # Manually annotated OHLCV data for detector tests
│   ├── unit/
│   ├── integration/
│   └── backtest_validation/
├── scripts/                       # seed_historical_data.py, seed_news_history.py, etc.
├── grafana/dashboards/
└── docs/
```

## Architecture Decisions

### Detectors: Dual Implementation

Every ICT detector (swings, FVG, OB, liquidity, displacement) MUST have two implementations:
1. **Vectorized** (`detect_*_vectorized(df: pl.DataFrame) -> pl.DataFrame`): operates on full history, used for backtest pre-computation. Polars-native.
2. **Incremental** (`detect_*_incremental(candle, state) -> list`): operates candle-by-candle, used in live. Wraps the vectorized version on a sliding window internally.

Both must produce identical results on the same data. Test this with a parametrized test that runs both and compares outputs.

### Multi-Timeframe

- Base timeframe stored in DB: M5 (ingested from IG)
- Higher timeframes computed via TimescaleDB continuous aggregates: H1, H4, D1
- `MarketStructureState` (in `src/structure/state.py`) maintains independent state per TF per instrument
- At each M5 candle, check if an H1/H4/D1 candle just closed → update the corresponding TF state
- **Critical**: never look ahead. In backtest, a D1 candle is only available after its close time.

### Backtest Engine (Hybrid)

```
Phase 1: Pre-compute (vectorized, fast)
  swings, FVGs, OBs, liquidity pools, session masks → indexed by (time, price_range)

Phase 2: Event loop (stateful, candle-by-candle on M5)
  for each M5 candle:
    update MarketStructureState (BOS/CHoCH) per TF
    lookup pre-computed detectors by time + price range
    replay historical news events if timestamp matches
    evaluate confluence score → entry/exit decision
    update simulated portfolio (spread, slippage, swap)
```

Simulation must include: variable spread by instrument/session, random slippage (0-2 pips scaled by volatility), overnight swap costs, order rejection simulation.

### Walk-Forward Validation

All strategy evaluation uses walk-forward, never simple backtest:
- Default: train=4 months, test=1 month, step=1 month
- Params configurable in `config/improvement.yml`
- Metrics aggregated across all windows: mean Sharpe, worst-window MDD, profit factor, win rate, avg R:R

### News Architecture (Adapter Pattern)

```python
class NewsSource(ABC):
    async def connect(self) -> None: ...
    async def subscribe(self, instruments: list[str]) -> None: ...
    async def get_events(self, start: datetime, end: datetime) -> list[NewsEvent]: ...
    async def stream(self) -> AsyncIterator[NewsEvent]: ...
```

Implementations: `FinnhubCalendarSource`, `FFApiHistoricalSource`, `NewsfilterRealtimeSource`, `NewsfilterHistoricalSource`. Adding a new source = implementing this interface, nothing else.

### Improvement: Dual Loop

1. **Optuna (daily)**: optimizes numeric params (confluence weights, sizing thresholds, detector params). Objective = walk-forward Sharpe. Guards: min improvement >2%, MDD degradation <5%, trade count stability.
2. **LLM (weekly)**: Claude Sonnet analyzes weekly trades, proposes structural code changes (new filters, rule modifications). Applied in sandbox branch → walk-forward validation → accept/reject. Max iterations configurable (default: 5). Git tag each iteration.

Anti-overfitting: reject if Sharpe jumps >50% in one iteration. Version everything.

### Risk Management

- **Position sizing**: dynamic, based on confluence score. Score maps to risk % (0.5% → 2% of capital). Thresholds optimizable by Optuna.
- **Circuit breakers**: max daily drawdown (default -3%), max total drawdown (default -10% of initial capital), max simultaneous positions (default 5).

## Implementation Order

Follow this order strictly. Each phase builds on the previous one. Do not skip ahead.

### Phase 1: Foundations
1. `pyproject.toml` with uv — all dependencies declared
2. `docker-compose.yml` — PostgreSQL/TimescaleDB + Grafana
3. `src/common/` — models (Pydantic), config loader, DB connection (asyncpg), structured logging, exceptions
4. `src/market_data/` — IG client wrapper, historical data download, TimescaleDB storage with hypertables, continuous aggregates (M5→H1,H4,D1)
5. `scripts/seed_historical_data.py`
6. Tests for all of the above

**Done when**: `docker compose up` starts everything, historical data is in DB, multi-TF aggregation works.

### Phase 2: ICT Detectors
7. `src/structure/swings.py` — swing highs/lows (Williams fractals)
8. `src/structure/market_structure.py` — BOS, CHoCH detection, trend state
9. `src/structure/fvg.py` — Fair Value Gaps
10. `src/structure/order_blocks.py` — Order Blocks
11. `src/structure/liquidity.py` — liquidity pools (equal H/L, session H/L)
12. `src/structure/premium_discount.py` — Fibonacci premium/discount zones
13. `src/structure/sessions.py` — session/killzone time masks
14. `src/structure/displacement.py` — displacement/impulse detection
15. `src/structure/state.py` — MarketStructureState (multi-TF, multi-instrument)
16. Create annotated test fixtures in `tests/fixtures/` with known BOS/CHoCH/FVG/OB cases
17. Tests: >90% coverage on `src/structure/`. Parametrized tests comparing vectorized vs incremental outputs.

**Done when**: all detectors pass on annotated data, vectorized and incremental produce identical results.

### Phase 3: Backtest + Strategy
18. `src/backtest/vectorized.py` — pre-computation pipeline
19. `src/backtest/engine.py` — M5 event loop consuming pre-computed data
20. `src/backtest/simulator.py` — realistic execution simulation
21. `src/backtest/metrics.py` — Sharpe, Sortino, MDD, PF, win rate, R:R
22. `src/backtest/walk_forward.py` — walk-forward validation engine
23. `src/strategy/confluence.py` — rule-based confluence scoring
24. `src/strategy/entry.py` — entry conditions
25. `src/strategy/exit.py` — exit conditions (SL, TP, trailing, time-based)
26. `src/strategy/filters.py` — spread, session, correlation, max positions
27. `src/execution/position_sizer.py` — dynamic sizing
28. `src/execution/risk_manager.py` — circuit breakers
29. `src/backtest/report.py` — backtest report generation
30. End-to-end backtest on 6+ months of data

**Done when**: walk-forward backtest runs end-to-end with coherent metrics. Strategy doesn't need to be profitable yet.

### Phase 4: News
31. `src/news/base.py` — NewsSource ABC
32. `src/news/calendar/finnhub.py` — Finnhub calendar adapter
33. `src/news/calendar/ff_api.py` — FF News API historical adapter
34. `src/news/realtime/newsfilter.py` — newsfilter.io WebSocket + Query adapter
35. `src/news/interpreter.py` — Claude Haiku LLM interpretation
36. `src/news/event_manager.py` — event actions (pause, tighten_stops, trigger_entry, none)
37. `src/news/store.py` — DB storage with aligned timestamps
38. Integrate news replay into backtest engine
39. `scripts/seed_news_history.py`
40. Backtest comparison: strategy with vs without news filter/trigger

**Done when**: historical news replay works in backtest, news filter prevents entries pre-NFP, event triggers generate entries.

### Phase 5: Live Demo
41. `src/execution/order_manager.py` — IG API order execution (demo account)
42. `src/execution/portfolio.py` — real-time position tracking
43. `src/monitoring/telegram_bot.py` — trade/alert notifications + commands (/status, /pause, /resume, /stats)
44. `src/monitoring/metrics_exporter.py` — Prometheus/Grafana export
45. `src/monitoring/health.py` — health checks
46. Grafana dashboards (trading overview, backtest results)
47. Full Docker Compose with bot service
48. `docs/runbook.md`

**Done when**: bot trades on IG demo, trades visible in IG web UI, Telegram notifies, Grafana displays.

### Phase 6: Improvement Loop
49. `src/improvement/trade_logger.py` — full context capture per trade
50. `src/improvement/optuna_optimizer.py` — daily parameter optimization
51. `src/improvement/llm_analyzer.py` — weekly LLM structural analysis
52. `src/improvement/patch_manager.py` — apply/rollback patches
53. `src/improvement/validator.py` — walk-forward validation post-modification
54. `src/improvement/versioning.py` — automatic git tagging
55. Grafana dashboard for improvement history

**Done when**: Optuna cycle runs and accepts/rejects improvements. LLM cycle proposes, backtests, and accepts/rejects changes.

### Phase 7: Stabilization
56. Run on demo for 2+ weeks, monitor stability
57. Fix issues, tune parameters
58. Optimize performance (backtest speed, execution latency)
59. Complete documentation
60. Security review (secrets management)

## Database Schema

```sql
-- Hypertable for market data
CREATE TABLE candles (
    time TIMESTAMPTZ NOT NULL, instrument TEXT NOT NULL, timeframe TEXT NOT NULL,
    open FLOAT8, high FLOAT8, low FLOAT8, close FLOAT8, volume FLOAT8, spread FLOAT8,
    PRIMARY KEY (time, instrument, timeframe)
);

-- Trades
CREATE TABLE trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    opened_at TIMESTAMPTZ NOT NULL, closed_at TIMESTAMPTZ,
    instrument TEXT NOT NULL, direction TEXT NOT NULL,
    entry_price FLOAT8, exit_price FLOAT8, stop_loss FLOAT8, take_profit FLOAT8,
    size FLOAT8, pnl FLOAT8, pnl_percent FLOAT8, r_multiple FLOAT8,
    confluence_score FLOAT8, setup_type JSONB, context JSONB, news_context JSONB,
    is_backtest BOOLEAN DEFAULT FALSE, backtest_run_id UUID
);

-- News events
CREATE TABLE news_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    time TIMESTAMPTZ NOT NULL, source TEXT NOT NULL, event_type TEXT NOT NULL,
    title TEXT, content TEXT, currency TEXT,
    actual TEXT, forecast TEXT, previous TEXT, impact_level TEXT,
    llm_analysis JSONB, instruments TEXT[]
);

-- Backtest runs
CREATE TABLE backtest_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at TIMESTAMPTZ NOT NULL, completed_at TIMESTAMPTZ,
    config JSONB NOT NULL, walk_forward JSONB, metrics JSONB NOT NULL,
    git_tag TEXT, improvement_type TEXT
);

-- Improvement log
CREATE TABLE improvement_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL, type TEXT NOT NULL,
    proposal JSONB NOT NULL, baseline_metrics JSONB NOT NULL, new_metrics JSONB NOT NULL,
    accepted BOOLEAN NOT NULL, reason TEXT, git_tag_before TEXT, git_tag_after TEXT
);
```

## Testing Strategy

- **Unit tests**: every detector, every scorer, every risk rule. Use annotated fixtures for ICT detectors.
- **Parametrized vectorized vs incremental**: for each detector, run both versions on the same data, assert identical results.
- **Integration tests**: DB read/write, IG API mock, backtest end-to-end on small fixture data.
- **Backtest validation**: walk-forward on real data (not in CI, run manually or in dedicated workflow).
- **Mocking**: mock all external APIs (IG, Finnhub, newsfilter, Anthropic) in tests. Use `pytest-asyncio` for async tests.
- **CI gate**: all tests pass + coverage above threshold + ruff clean + mypy clean.

### GitHub Actions CI (`.github/workflows/ci.yml`)

CI runs on every PR targeting `master`. The PR cannot be merged until CI passes.

```yaml
name: CI
on:
  pull_request:
    branches: [master]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: timescale/timescaledb:latest-pg16
        env:
          POSTGRES_DB: test_trading_bot
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
        ports: ["5432:5432"]
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync
      - run: uv run ruff check src/ tests/
      - run: uv run mypy src/
      - run: uv run pytest --cov=src --cov-report=xml --cov-fail-under=80
        env:
          DATABASE_URL: postgresql://test:test@localhost:5432/test_trading_bot
```

Configure branch protection rules on `master` in GitHub:
- Require PR before merging
- Require status checks to pass (CI job `test`)
- Require review (Copilot review via `gh pr edit --add-reviewer @copilot`)

## Key Libraries

```
# Core
polars, numpy, pydantic>=2.0, structlog, pyyaml, asyncpg, aiohttp

# Broker
trading-ig

# News
websockets, finnhub-python

# LLM
anthropic

# ML/Optimization
optuna, scikit-learn (future: xgboost/lightgbm for supervised scoring)

# Backtest
# No framework — custom engine. Polars for vectorized ops.

# Monitoring
python-telegram-bot, prometheus-client

# Testing
pytest, pytest-asyncio, pytest-cov, pytest-mock, aioresponses

# Dev
ruff, mypy
```
