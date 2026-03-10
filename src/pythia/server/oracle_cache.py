"""Oracle AI Vector Search cache — semantic search memory with Python embeddings."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

import oracledb

# Return LOBs as strings/bytes instead of AsyncLOB objects
oracledb.defaults.fetch_lobs = False

from pythia.embeddings import generate_embedding as _generate_embedding


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
        """Create async connection pool."""
        self._pool = oracledb.create_pool_async(
            user=self.user,
            password=self.password,
            dsn=self.dsn,
            min=1,
            max=4,
        )

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()

    async def lookup(self, query: str) -> CacheEntry | None:
        """Find semantically similar cached result using Python-generated embeddings."""
        if not self._pool:
            return None

        # Generate embedding in Python
        query_embedding = _generate_embedding(query)
        
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
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, [query_embedding, query_embedding])
                row = await cur.fetchone()
                if not row:
                    return None
                similarity = float(row[7])
                if not self._is_cache_hit(similarity):
                    return None
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
                )

    async def store(
        self, query: str, answer: str, sources: list[dict], model_used: str
    ) -> None:
        """Store a search result in the cache with Python-generated embedding."""
        if not self._pool:
            return

        # Generate embedding in Python
        query_embedding = _generate_embedding(query)
        
        sql = """
            INSERT INTO pythia_cache (query, query_embedding, answer, sources, model_used)
            VALUES (:1, TO_VECTOR(:2, 384), :3, :4, :5)
        """
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
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
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, [query, 1 if cache_hit else 0, response_time_ms, model_used])
                await conn.commit()

    async def get_stats(self) -> dict:
        """Get search statistics."""
        if not self._pool:
            return {"total_searches": 0, "cache_hits": 0, "cache_hit_rate": 0, "avg_response_ms": 0}

        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
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
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT COUNT(*) FROM pythia_cache")
                row = await cur.fetchone()
                return row[0] if row else 0

    async def clear_cache(self) -> int:
        """Delete all cached entries. Returns count deleted."""
        if not self._pool:
            return 0
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
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
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
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
            async with self._pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1 FROM DUAL")
                    return True
        except Exception:
            return False

    async def recall_findings(self, query: str, threshold: float = 0.70, limit: int = 5) -> list[dict]:
        """Recall related findings from past research sessions via vector similarity."""
        if not self._pool:
            return []
        query_embedding = _generate_embedding(query)
        sql = """
            SELECT f.sub_query, f.summary, f.sources, r.query AS research_query,
                   1 - VECTOR_DISTANCE(f.finding_embedding, TO_VECTOR(:1, 384), COSINE) AS similarity
            FROM pythia_findings f
            JOIN pythia_research r ON f.research_id = r.id
            ORDER BY VECTOR_DISTANCE(f.finding_embedding, TO_VECTOR(:2, 384), COSINE)
            FETCH FIRST :3 ROWS ONLY
        """
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
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
    ) -> str:
        """Store a research session. Returns the research ID."""
        if not self._pool:
            return ""
        query_embedding = _generate_embedding(query)
        sql = """
            INSERT INTO pythia_research (query, query_embedding, report, sub_queries, rounds_used, total_sources, model_used, elapsed_ms)
            VALUES (:1, TO_VECTOR(:2, 384), :3, :4, :5, :6, :7, :8)
            RETURNING id INTO :9
        """
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                research_id_var = cur.var(oracledb.DB_TYPE_RAW)
                await cur.execute(sql, [
                    query, query_embedding, report, json.dumps(sub_queries),
                    rounds_used, total_sources, model_used, elapsed_ms, research_id_var,
                ])
                await conn.commit()
                return research_id_var.getvalue()[0].hex()

    async def store_finding(
        self, research_id: str, sub_query: str, summary: str,
        sources: list[dict], round_num: int,
    ) -> None:
        """Store an individual research finding with embedding for future recall."""
        if not self._pool:
            return
        finding_embedding = _generate_embedding(sub_query + " " + summary[:200])
        sql = """
            INSERT INTO pythia_findings (research_id, sub_query, finding_embedding, summary, sources, round_num)
            VALUES (:1, :2, TO_VECTOR(:3, 384), :4, :5, :6)
        """
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, [
                    bytes.fromhex(research_id), sub_query, finding_embedding,
                    summary, json.dumps(sources), round_num,
                ])
                await conn.commit()

    def _is_cache_hit(self, similarity: float) -> bool:
        return similarity >= self.similarity_threshold
