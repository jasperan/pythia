-- setup_schema.sql
-- Oracle Database 26ai vector schema for Pythia
--
-- Prerequisites:
--   1. Oracle 26ai Free container running (docker compose up -d)
--   2. Connect as ADMIN to create the pythia user
--
-- Step 1: Create user (run as ADMIN)
-- CREATE USER pythia IDENTIFIED BY pythia;
-- GRANT CONNECT, RESOURCE, UNLIMITED TABLESPACE, DB_DEVELOPER_ROLE TO pythia;
-- GRANT CREATE MINING MODEL TO pythia;

-- Optional Step: Load an ONNX embedding model (run as PYTHIA user)
-- The default Pythia runtime does NOT depend on this step.
-- Pythia generates embeddings in Python with sentence-transformers and stores
-- them in Oracle using TO_VECTOR(...).
--
-- If you want to experiment with Oracle-side VECTOR_EMBEDDING(), Oracle 26ai
-- ships with pre-loaded models and also supports custom ONNX loading.
-- Example custom ONNX load:
--
--   BEGIN
--     DBMS_VECTOR.LOAD_ONNX_MODEL(
--       'DM_DUMP',
--       'all_MiniLM_L6_v2.onnx',
--       'ALL_MINILM_L6_V2',
--       JSON('{"function":"embedding","embeddingOutput":"embedding","input":{"input":["DATA"]}}')
--     );
--   END;
--   /
--
-- Example built-in model verification:
-- SELECT VECTOR_EMBEDDING(ALL_MINILM_L6_V2 USING 'test' AS data) FROM DUAL;

-- Step 2: Create tables (run as PYTHIA user)

-- Semantic search cache
-- query_embedding stores the Python-generated embedding via TO_VECTOR(...) at INSERT time
CREATE TABLE pythia_cache (
    id              RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    query           VARCHAR2(4000) NOT NULL,
    query_embedding VECTOR         NOT NULL,
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

-- Research sessions
CREATE TABLE pythia_research (
    id              RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    query           VARCHAR2(4000) NOT NULL,
    query_embedding VECTOR         NOT NULL,
    report          CLOB,
    sub_queries     CLOB,
    rounds_used     NUMBER         DEFAULT 0,
    total_sources   NUMBER         DEFAULT 0,
    model_used      VARCHAR2(100)  NOT NULL,
    elapsed_ms      NUMBER,
    created_at      TIMESTAMP      DEFAULT SYSTIMESTAMP
);

CREATE VECTOR INDEX pythia_research_vec_idx
    ON pythia_research (query_embedding)
    ORGANIZATION NEIGHBOR PARTITIONS
    WITH DISTANCE COSINE;

-- Individual research findings (for cross-session recall)
CREATE TABLE pythia_findings (
    id              RAW(16) DEFAULT SYS_GUID() PRIMARY KEY,
    research_id     RAW(16)        NOT NULL REFERENCES pythia_research(id),
    sub_query       VARCHAR2(4000) NOT NULL,
    finding_embedding VECTOR       NOT NULL,
    summary         CLOB           NOT NULL,
    sources         CLOB,
    round_num       NUMBER         DEFAULT 1,
    created_at      TIMESTAMP      DEFAULT SYSTIMESTAMP
);

CREATE INDEX pythia_findings_research_idx ON pythia_findings (research_id);

CREATE VECTOR INDEX pythia_findings_vec_idx
    ON pythia_findings (finding_embedding)
    ORGANIZATION NEIGHBOR PARTITIONS
    WITH DISTANCE COSINE;

-- Analytics view
CREATE OR REPLACE VIEW pythia_stats AS
SELECT
    COUNT(*)                                                    AS total_searches,
    SUM(cache_hit)                                              AS cache_hits,
    ROUND(SUM(cache_hit) / NULLIF(COUNT(*), 0) * 100, 1)       AS cache_hit_rate,
    ROUND(AVG(response_time_ms), 0)                             AS avg_response_ms,
    COUNT(DISTINCT TRUNC(created_at))                           AS active_days
FROM pythia_history;
