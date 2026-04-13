-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Hypertable for market data
CREATE TABLE candles (
    time TIMESTAMPTZ NOT NULL,
    instrument TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    open FLOAT8 NOT NULL,
    high FLOAT8 NOT NULL,
    low FLOAT8 NOT NULL,
    close FLOAT8 NOT NULL,
    volume FLOAT8 NOT NULL DEFAULT 0,
    spread FLOAT8,
    PRIMARY KEY (time, instrument, timeframe)
);

SELECT create_hypertable('candles', 'time');

-- Continuous aggregates for H1, H4, D1 from M5 base data
-- H1 aggregate
CREATE MATERIALIZED VIEW candles_h1
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS time,
    instrument,
    first(open, time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, time) AS close,
    sum(volume) AS volume,
    avg(spread) AS spread
FROM candles
WHERE timeframe = 'M5'
GROUP BY time_bucket('1 hour', time), instrument
WITH NO DATA;

-- H4 aggregate
CREATE MATERIALIZED VIEW candles_h4
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('4 hours', time) AS time,
    instrument,
    first(open, time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, time) AS close,
    sum(volume) AS volume,
    avg(spread) AS spread
FROM candles
WHERE timeframe = 'M5'
GROUP BY time_bucket('4 hours', time), instrument
WITH NO DATA;

-- D1 aggregate
CREATE MATERIALIZED VIEW candles_d1
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS time,
    instrument,
    first(open, time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, time) AS close,
    sum(volume) AS volume,
    avg(spread) AS spread
FROM candles
WHERE timeframe = 'M5'
GROUP BY time_bucket('1 day', time), instrument
WITH NO DATA;

-- Refresh policies: refresh continuously
SELECT add_continuous_aggregate_policy('candles_h1',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes');

SELECT add_continuous_aggregate_policy('candles_h4',
    start_offset => INTERVAL '8 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes');

SELECT add_continuous_aggregate_policy('candles_d1',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes');

-- Trades table
CREATE TABLE trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    opened_at TIMESTAMPTZ NOT NULL,
    closed_at TIMESTAMPTZ,
    instrument TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_price FLOAT8,
    exit_price FLOAT8,
    stop_loss FLOAT8,
    take_profit FLOAT8,
    size FLOAT8,
    pnl FLOAT8,
    pnl_percent FLOAT8,
    r_multiple FLOAT8,
    confluence_score FLOAT8,
    setup_type JSONB,
    context JSONB,
    news_context JSONB,
    is_backtest BOOLEAN DEFAULT FALSE,
    backtest_run_id UUID
);

-- News events
CREATE TABLE news_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    time TIMESTAMPTZ NOT NULL,
    source TEXT NOT NULL,
    event_type TEXT NOT NULL,
    title TEXT,
    content TEXT,
    currency TEXT,
    actual TEXT,
    forecast TEXT,
    previous TEXT,
    impact_level TEXT,
    llm_analysis JSONB,
    instruments TEXT[]
);

-- Backtest runs
CREATE TABLE backtest_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    config JSONB NOT NULL,
    walk_forward JSONB,
    metrics JSONB NOT NULL,
    git_tag TEXT,
    improvement_type TEXT
);

-- Improvement log
CREATE TABLE improvement_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL,
    type TEXT NOT NULL,
    proposal JSONB NOT NULL,
    baseline_metrics JSONB NOT NULL,
    new_metrics JSONB NOT NULL,
    accepted BOOLEAN NOT NULL,
    reason TEXT,
    git_tag_before TEXT,
    git_tag_after TEXT
);

-- Tick data (raw bid/ask from Dukascopy or IG)
CREATE TABLE IF NOT EXISTS ticks (
    time TIMESTAMPTZ NOT NULL,
    instrument TEXT NOT NULL,
    bid FLOAT8 NOT NULL,
    ask FLOAT8 NOT NULL
);

SELECT create_hypertable('ticks', 'time', if_not_exists => TRUE);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_ticks_instrument_time ON ticks (instrument, time DESC);
CREATE INDEX idx_candles_instrument_tf ON candles (instrument, timeframe, time DESC);
CREATE INDEX idx_trades_instrument ON trades (instrument, opened_at DESC);
CREATE INDEX idx_trades_backtest ON trades (backtest_run_id) WHERE is_backtest = TRUE;
CREATE INDEX idx_news_events_time ON news_events (time DESC);
CREATE INDEX idx_news_events_currency ON news_events (currency, time DESC);
