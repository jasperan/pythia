"""Oracle AI Vector Search cache — semantic search memory with Python embeddings."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime

import oracledb

from pythia.embeddings import generate_embedding as _generate_embedding

logger = logging.getLogger(__name__)

# Return LOBs as strings/bytes instead of AsyncLOB objects
oracledb.defaults.fetch_lobs = False


@dataclass
class CacheEntry:
    query: str
    answer: str
    sources: list[dict]
    model_used: str
    similarity: float = 0.0
    hit_count: int = 0
    created_at: datetime | None = None


class OracleCache:
    """Oracle AI Vector Search cache with Python-generated embeddings."""

    def __init__(
        self, dsn: str, user: str, password: str, similarity_threshold: float = 0.85,
        embedding_model: str = "ALL_MINILM_L6_V2",
    ):
        self.dsn = dsn
        self.user = user
        self.password = password
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self._pool: oracledb.AsyncConnectionPool | None = None

    async def connect(self) -> None:
        """Create async connection pool. Logs warning and continues if Oracle is unreachable."""
        try:
            self._pool = oracledb.create_pool_async(
                user=self.user,
                password=self.password,
                dsn=self.dsn,
                min=1,
                max=4,
            )
        except Exception as e:
            logger.warning(f"Oracle connection pool creation failed: {e}")
            self._pool = None

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()

    async def generate_embedding(self, text: str) -> str:
        """Generate embedding string, offloaded to a thread."""
        return await asyncio.to_thread(_generate_embedding, text)

    async def lookup(self, query: str) -> tuple[CacheEntry | None, str]:
        """Find semantically similar cached result. Returns (entry, query_embedding).

        The embedding is returned so callers can reuse it for store() on cache miss,
        avoiding a redundant embedding computation.
        """
        if not self._pool:
            return None, ""

        query_embedding = await self.generate_embedding(query)

        # Use VECTOR_DISTANCE for cosine similarity search
        sql = """
            SELECT id, query, answer, sources, model_used, hit_count, created_at,
                   1 - VECTOR_DISTANCE(
                       query_embedding,
                       TO_VECTOR(:1, 384),
                       COSINE
                   ) AS similarity
            FROM pythia_cache
            ORDER BY VECTOR_DISTANCE(
                query_embedding,
                TO_VECTOR(:2, 384),
                COSINE
            )
            FETCH FIRST 1 ROW ONLY
        """
        async with self._pool.acquire() as conn, conn.cursor() as cur:
            await cur.execute(sql, [query_embedding, query_embedding])
            row = await cur.fetchone()
            if not row:
                return None, query_embedding
            similarity = float(row[7])
            if not self._is_cache_hit(similarity):
                return None, query_embedding
            await cur.execute(
                "UPDATE pythia_cache SET hit_count = hit_count + 1, last_hit_at = SYSTIMESTAMP WHERE id = :1",
                [row[0]],
            )
            await conn.commit()
            return CacheEntry(
                query=row[1],
                answer=row[2],
                sources=json.loads(row[3]) if row[3] else [],
                model_used=row[4],
                hit_count=row[5] + 1,
                created_at=row[6],
                similarity=similarity,
            ), query_embedding

    async def store(
        self, query: str, answer: str, sources: list[dict], model_used: str,
        query_embedding: str | None = None,
    ) -> None:
        """Store a search result in the cache. Accepts pre-computed embedding to avoid redundant work."""
        if not self._pool:
            return

        if not query_embedding:
            query_embedding = await self.generate_embedding(query)

        sql = """
            INSERT INTO pythia_cache (query, query_embedding, answer, sources, model_used)
            VALUES (:1, TO_VECTOR(:2, 384), :3, :4, :5)
        """
        async with self._pool.acquire() as conn, conn.cursor() as cur:
            sources_json = json.dumps(sources)
            await cur.execute(sql, [query, query_embedding, answer, sources_json, model_used])
            await conn.commit()

    async def record_search(self, query: str, cache_hit: bool, response_time_ms: int, model_used: str) -> None:
        """Record a search in history."""
        if not self._pool:
            return

        sql = """
            INSERT INTO pythia_history (query, cache_hit, response_time_ms, model_used)
            VALUES (:1, :2, :3, :4)
        """
        async with self._pool.acquire() as conn, conn.cursor() as cur:
            await cur.execute(sql, [query, 1 if cache_hit else 0, response_time_ms, model_used])
            await conn.commit()

    async def get_stats(self) -> dict:
        """Get search statistics."""
        if not self._pool:
            return {"total_searches": 0, "cache_hits": 0, "cache_hit_rate": 0, "avg_response_ms": 0}

        async with self._pool.acquire() as conn, conn.cursor() as cur:
            await cur.execute("SELECT * FROM pythia_stats")
            row = await cur.fetchone()
            if not row:
                return {"total_searches": 0, "cache_hits": 0, "cache_hit_rate": 0, "avg_response_ms": 0}
            return {
                "total_searches": row[0] or 0,
                "cache_hits": row[1] or 0,
                "cache_hit_rate": float(row[2] or 0),
                "avg_response_ms": int(row[3] or 0),
                "active_days": row[4] or 0,
            }

    async def get_cache_size(self) -> int:
        """Get number of cached entries."""
        if not self._pool:
            return 0
        async with self._pool.acquire() as conn, conn.cursor() as cur:
            await cur.execute("SELECT COUNT(*) FROM pythia_cache")
            row = await cur.fetchone()
            return row[0] if row else 0

    async def clear_cache(self) -> int:
        """Delete all cached entries. Returns count deleted."""
        if not self._pool:
            return 0
        async with self._pool.acquire() as conn, conn.cursor() as cur:
            await cur.execute("SELECT COUNT(*) FROM pythia_cache")
            row = await cur.fetchone()
            count = row[0] if row else 0
            await cur.execute("DELETE FROM pythia_cache")
            await conn.commit()
            return count

    async def get_history(self, limit: int = 20) -> list[dict]:
        """Get recent search history."""
        if not self._pool:
            return []
        async with self._pool.acquire() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT query, cache_hit, response_time_ms, model_used, created_at "
                "FROM pythia_history ORDER BY created_at DESC FETCH FIRST :1 ROWS ONLY",
                [limit],
            )
            rows = await cur.fetchall()
            return [
                {
                    "query": r[0],
                    "cache_hit": bool(r[1]),
                    "response_time_ms": r[2],
                    "model_used": r[3],
                    "created_at": r[4].isoformat() if r[4] else None,
                }
                for r in rows
            ]

    async def health(self) -> bool:
        """Check Oracle connectivity."""
        try:
            if not self._pool:
                return False
            async with self._pool.acquire() as conn, conn.cursor() as cur:
                await cur.execute("SELECT 1 FROM DUAL")
                return True
        except Exception:
            return False

    async def recall_findings(self, query: str, threshold: float = 0.70, limit: int = 5) -> list[dict]:
        """Recall related findings from past research sessions via vector similarity."""
        if not self._pool:
            return []
        query_embedding = await asyncio.to_thread(_generate_embedding, query)
        sql = """
            SELECT f.sub_query, f.summary, f.sources, r.query AS research_query,
                   1 - VECTOR_DISTANCE(f.finding_embedding, TO_VECTOR(:1, 384), COSINE) AS similarity
            FROM pythia_findings f
            JOIN pythia_research r ON f.research_id = r.id
            ORDER BY VECTOR_DISTANCE(f.finding_embedding, TO_VECTOR(:2, 384), COSINE)
            FETCH FIRST :3 ROWS ONLY
        """
        async with self._pool.acquire() as conn, conn.cursor() as cur:
            await cur.execute(sql, [query_embedding, query_embedding, limit])
            rows = await cur.fetchall()
            results = []
            for row in rows:
                sim = float(row[4])
                if sim < threshold:
                    continue
                results.append({
                    "sub_query": row[0],
                    "summary": row[1],
                    "sources": json.loads(row[2]) if row[2] else [],
                    "research_query": row[3],
                    "similarity": sim,
                })
            return results

    async def store_research(
        self, query: str, report: str, sub_queries: list[str],
        rounds_used: int, total_sources: int, model_used: str, elapsed_ms: int,
        slug: str | None = None, parent_id: str | None = None,
        verification_status: str | None = None,
        verification_summary: str | None = None,
        provenance: str | None = None,
        **_kwargs,
    ) -> str:
        """Store a research session. Returns the research ID."""
        if not self._pool:
            return ""
        query_embedding = await asyncio.to_thread(_generate_embedding, query)
        sql = """
            INSERT INTO pythia_research (
                query, query_embedding, report, sub_queries,
                rounds_used, total_sources, model_used, elapsed_ms,
                slug, parent_id, verification_status, verification_summary, provenance
            ) VALUES (
                :1, TO_VECTOR(:2, 384), :3, :4,
                :5, :6, :7, :8,
                :9, :10, :11, :12, :13
            )
            RETURNING id INTO :14
        """
        parent_id_bytes = bytes.fromhex(parent_id) if parent_id else None
        async with self._pool.acquire() as conn, conn.cursor() as cur:
            research_id_var = cur.var(oracledb.DB_TYPE_RAW)
            await cur.execute(sql, [
                query, query_embedding, report, json.dumps(sub_queries),
                rounds_used, total_sources, model_used, elapsed_ms,
                slug, parent_id_bytes, verification_status,
                verification_summary, provenance, research_id_var,
            ])
            await conn.commit()
            return research_id_var.getvalue()[0].hex()

    async def store_findings_batch(
        self, research_id: str, findings: list[dict],
    ) -> None:
        """Store multiple findings in a single transaction. Each dict needs: sub_query, summary, sources, round_num."""
        if not self._pool or not findings:
            return
        sql = """
            INSERT INTO pythia_findings (research_id, sub_query, finding_embedding, summary, sources, round_num)
            VALUES (:1, :2, TO_VECTOR(:3, 384), :4, :5, :6)
        """
        # Generate all embeddings in parallel
        embedding_tasks = [
            self.generate_embedding(f["sub_query"] + " " + f["summary"][:200])
            for f in findings
        ]
        embeddings = await asyncio.gather(*embedding_tasks)
        rows = [
            [
                bytes.fromhex(research_id), f["sub_query"], emb,
                f["summary"], json.dumps(f["sources"]), f["round_num"],
            ]
            for f, emb in zip(findings, embeddings, strict=False)
        ]
        async with self._pool.acquire() as conn, conn.cursor() as cur:
            await cur.executemany(sql, rows)
            await conn.commit()

    async def get_research_by_slug(self, slug: str) -> dict | None:
        """Load a research session by slug. Returns the most recent match."""
        if not self._pool:
            return None
        sql = """
            SELECT id, query, report, sub_queries, rounds_used, total_sources,
                   model_used, elapsed_ms, slug, parent_id,
                   verification_status, verification_summary, provenance, created_at
            FROM pythia_research
            WHERE slug = :1
            ORDER BY created_at DESC
            FETCH FIRST 1 ROW ONLY
        """
        async with self._pool.acquire() as conn, conn.cursor() as cur:
            await cur.execute(sql, [slug])
            row = await cur.fetchone()
            if not row:
                return None
            return {
                "id": row[0].hex() if row[0] else None,
                "query": row[1],
                "report": row[2],
                "sub_queries": json.loads(row[3]) if row[3] else [],
                "rounds_used": row[4],
                "total_sources": row[5],
                "model_used": row[6],
                "elapsed_ms": row[7],
                "slug": row[8],
                "parent_id": row[9].hex() if row[9] else None,
                "verification_status": row[10],
                "verification_summary": row[11],
                "provenance": row[12],
                "created_at": row[13].isoformat() if row[13] else None,
            }

    async def get_findings_for_research(self, research_id: str) -> list[dict]:
        """Load all findings for a research session, ordered by round then creation time."""
        if not self._pool:
            return []
        sql = """
            SELECT sub_query, summary, sources, round_num, created_at
            FROM pythia_findings
            WHERE research_id = :1
            ORDER BY round_num, created_at
        """
        async with self._pool.acquire() as conn, conn.cursor() as cur:
            await cur.execute(sql, [bytes.fromhex(research_id)])
            rows = await cur.fetchall()
            return [
                {
                    "sub_query": row[0],
                    "summary": row[1],
                    "sources": json.loads(row[2]) if row[2] else [],
                    "round_num": row[3],
                    "created_at": row[4].isoformat() if row[4] else None,
                }
                for row in rows
            ]

    def _is_cache_hit(self, similarity: float) -> bool:
        return similarity >= self.similarity_threshold
