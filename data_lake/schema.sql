-- GMMIE data lake schema (DuckDB)
-- Single-file analytical DB. Append-only where possible; PK enforced.
-- All timestamps stored as TIMESTAMP (UTC); dates as DATE.

CREATE TABLE IF NOT EXISTS prices_d (
    date     DATE        NOT NULL,
    symbol   VARCHAR     NOT NULL,
    open     DOUBLE,
    high     DOUBLE,
    low      DOUBLE,
    close    DOUBLE,
    volume   DOUBLE,
    source   VARCHAR,
    PRIMARY KEY (date, symbol)
);
CREATE INDEX IF NOT EXISTS prices_d_sym ON prices_d (symbol);

CREATE TABLE IF NOT EXISTS macro_d (
    date       DATE     NOT NULL,
    series_id  VARCHAR  NOT NULL,
    value      DOUBLE,
    PRIMARY KEY (date, series_id)
);
CREATE INDEX IF NOT EXISTS macro_d_sid ON macro_d (series_id);

CREATE TABLE IF NOT EXISTS cot_disagg (
    report_date            DATE     NOT NULL,
    contract               VARCHAR  NOT NULL,
    open_interest          DOUBLE,
    mm_long                DOUBLE,
    mm_short               DOUBLE,
    mm_spread              DOUBLE,
    swap_long              DOUBLE,
    swap_short             DOUBLE,
    producer_long          DOUBLE,
    producer_short         DOUBLE,
    other_rep_long         DOUBLE,
    other_rep_short        DOUBLE,
    nonrep_long            DOUBLE,
    nonrep_short           DOUBLE,
    PRIMARY KEY (report_date, contract)
);

CREATE TABLE IF NOT EXISTS etf_flows (
    date           DATE     NOT NULL,
    ticker         VARCHAR  NOT NULL,
    holdings_tonnes DOUBLE,
    shares_out     DOUBLE,
    nav            DOUBLE,
    aum_usd        DOUBLE,
    flow_usd       DOUBLE,
    PRIMARY KEY (date, ticker)
);

CREATE TABLE IF NOT EXISTS news (
    id          VARCHAR  PRIMARY KEY,        -- sha256(url || ts)
    ts          TIMESTAMP NOT NULL,
    source      VARCHAR,
    url         VARCHAR,
    title       VARCHAR,
    text        VARCHAR,
    raw_json    VARCHAR
);
CREATE INDEX IF NOT EXISTS news_ts ON news (ts);

-- LLM-extracted structured features. Many-to-one isn't enforced; we keep latest
-- extraction per (news_id, model_version) and let SELECT pick the best.
CREATE TABLE IF NOT EXISTS news_features (
    news_id          VARCHAR NOT NULL,
    model_version    VARCHAR NOT NULL,
    extracted_at     TIMESTAMP NOT NULL,
    event_type       VARCHAR,
    gold_impact      DOUBLE,                 -- [-1, +1]
    confidence       DOUBLE,                 -- [0, 1]
    horizon_minutes  INTEGER,
    novel            BOOLEAN,
    surprise_signed  DOUBLE,                 -- [-1, +1]
    rationale        VARCHAR,
    PRIMARY KEY (news_id, model_version)
);

CREATE TABLE IF NOT EXISTS fomc_statements (
    date         DATE PRIMARY KEY,
    statement    VARCHAR,
    diff_prior   VARCHAR,                    -- diff vs prior statement
    embedding    BLOB
);

CREATE TABLE IF NOT EXISTS predictions (
    id            VARCHAR PRIMARY KEY,       -- uuid4
    ts            TIMESTAMP NOT NULL,
    model_version VARCHAR NOT NULL,
    horizon_d     INTEGER NOT NULL,
    ref_price     DOUBLE,
    q10           DOUBLE,
    q25           DOUBLE,
    q50           DOUBLE,
    q75           DOUBLE,
    q90           DOUBLE,
    regime        VARCHAR,
    confidence    DOUBLE,
    context_json  VARCHAR
);
CREATE INDEX IF NOT EXISTS pred_ts ON predictions (ts);
CREATE INDEX IF NOT EXISTS pred_settle ON predictions (ts, horizon_d);

CREATE TABLE IF NOT EXISTS outcomes (
    prediction_id   VARCHAR NOT NULL,
    settled_at      TIMESTAMP NOT NULL,
    realised_price  DOUBLE,
    realised_logret DOUBLE,
    pinball_loss    DOUBLE,
    PRIMARY KEY (prediction_id)
);

CREATE TABLE IF NOT EXISTS theses (
    date          DATE PRIMARY KEY,
    model         VARCHAR,
    thesis        VARCHAR,
    confidence    DOUBLE,
    predictions   VARCHAR,                   -- JSON
    falsifiable   VARCHAR,                   -- list of falsifiable claims
    brier_score   DOUBLE                     -- filled in retrospectively
);

-- Provenance: which backfill run produced what
CREATE TABLE IF NOT EXISTS lake_runs (
    run_id      VARCHAR PRIMARY KEY,
    started_at  TIMESTAMP,
    finished_at TIMESTAMP,
    source      VARCHAR,
    rows_added  INTEGER,
    notes       VARCHAR
);
