-- setup_schema.sql
-- Oracle Database 26ai with ONNX in-database embeddings
--
-- Prerequisites:
--   1. Oracle 26ai Free container running (docker compose up -d)
--   2. Connect as ADMIN to create user and load ONNX model
--
-- Step 1: Create user (run as ADMIN)
-- CREATE USER pythia IDENTIFIED BY pythia;
-- GRANT CONNECT, RESOURCE, UNLIMITED TABLESPACE, DB_DEVELOPER_ROLE TO pythia;
-- GRANT CREATE MINING MODEL TO pythia;

-- Step 2: Load ONNX embedding model (run as PYTHIA user)
-- The model is loaded once and used for all embedding operations.
-- Oracle 26ai ships with pre-loaded models accessible via VECTOR_EMBEDDING().
-- If using a custom ONNX model (e.g., all-MiniLM-L6-v2):
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
-- For Oracle 26ai ADB-Free, you can use the built-in model directly:
-- SELECT VECTOR_EMBEDDING(ALL_MINILM_L6_V2 USING 'test' AS data) FROM DUAL;

-- Step 3: Create tables (run as PYTHIA user)

-- Semantic search cache
-- query_embedding is generated via VECTOR_EMBEDDING() at INSERT time
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
