-- setup_schema.sql
-- Run as SYS or admin: CREATE USER pythia IDENTIFIED BY pythia; GRANT CONNECT, RESOURCE, UNLIMITED TABLESPACE TO pythia;

-- Semantic search cache
CREATE TABLE pythia_cache (
    id              RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    query           VARCHAR2(4000) NOT NULL,
    query_embedding VECTOR(768, FLOAT64) NOT NULL,
    answer          CLOB           NOT NULL,
    sources         CLOB           NOT NULL,
    model_used      VARCHAR2(100)  NOT NULL,
    search_engine   VARCHAR2(50)   DEFAULT 'searxng',
    created_at      TIMESTAMP      DEFAULT SYSTIMESTAMP,
    hit_count       NUMBER         DEFAULT 0,
    last_hit_at     TIMESTAMP
);

CREATE VECTOR INDEX pythia_cache_vec_idx
    ON pythia_cache (query_embedding)
    ORGANIZATION NEIGHBOR PARTITIONS
    WITH DISTANCE COSINE;

-- Search history
CREATE TABLE pythia_history (
    id               RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    query            VARCHAR2(4000) NOT NULL,
    cache_hit        NUMBER(1)      DEFAULT 0,
    response_time_ms NUMBER         NOT NULL,
    model_used       VARCHAR2(100),
    created_at       TIMESTAMP      DEFAULT SYSTIMESTAMP
);

-- Analytics view
CREATE OR REPLACE VIEW pythia_stats AS
SELECT
    COUNT(*)                                                    AS total_searches,
    SUM(cache_hit)                                              AS cache_hits,
    ROUND(SUM(cache_hit) / NULLIF(COUNT(*), 0) * 100, 1)       AS cache_hit_rate,
    ROUND(AVG(response_time_ms), 0)                             AS avg_response_ms,
    COUNT(DISTINCT TRUNC(created_at))                           AS active_days
FROM pythia_history;
