-- Migration: Add iterative investigation support to pythia_research
-- Run as PYTHIA user against FREEPDB1
--
-- New columns:
--   slug            — URL-safe identifier for lookup/continuation
--   parent_id       — links continuation/refinement sessions to originals
--   verification_status — pass/pass_with_notes/fail from claim verification
--   verification_summary — human-readable verification summary
--   provenance      — full provenance markdown

ALTER TABLE pythia_research ADD (
    slug                  VARCHAR2(100),
    parent_id             RAW(16) REFERENCES pythia_research(id),
    verification_status   VARCHAR2(50),
    verification_summary  VARCHAR2(4000),
    provenance            CLOB
);

CREATE INDEX pythia_research_slug_idx ON pythia_research (slug);
CREATE INDEX pythia_research_parent_idx ON pythia_research (parent_id);
