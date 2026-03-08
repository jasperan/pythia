"""Oracle AI Vector Search cache — semantic search memory."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

import oracledb


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
    """Oracle AI Vector Search cache for search results."""

    def __init__(self, dsn: str, user: str, password: str, similarity_threshold: float = 0.85):
        self.dsn = dsn
        self.user = user
        self.password = password
        self.similarity_threshold = similarity_threshold
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

    async def lookup(self, query_embedding: list[float]) -> CacheEntry | None:
        """Find semantically similar cached result."""
        if not self._pool:
            return None

        sql = """
            SELECT query, answer, sources, model_used, hit_count, created_at,
                   1 - VECTOR_DISTANCE(query_embedding, :1, COSINE) AS similarity
            FROM pythia_cache
            ORDER BY VECTOR_DISTANCE(query_embedding, :2, COSINE)
            FETCH FIRST 1 ROW ONLY
        """
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                vec_str = json.dumps(query_embedding)
                await cur.execute(sql, [vec_str, vec_str])
                row = await cur.fetchone()
                if not row:
                    return None
                similarity = float(row[6])
                if not self._is_cache_hit(similarity):
                    return None
                await cur.execute(
                    "UPDATE pythia_cache SET hit_count = hit_count + 1, last_hit_at = SYSTIMESTAMP WHERE query = :1",
                    [row[0]],
                )
                await conn.commit()
                return CacheEntry(
                    query=row[0],
                    answer=row[1],
                    sources=json.loads(row[2]) if row[2] else [],
                    model_used=row[3],
                    hit_count=row[4] + 1,
                    created_at=row[5],
                    similarity=similarity,
                )

    async def store(
        self, query: str, query_embedding: list[float], answer: str, sources: list[dict], model_used: str
    ) -> None:
        """Store a search result in the cache."""
        if not self._pool:
            return

        sql = """
            INSERT INTO pythia_cache (query, query_embedding, answer, sources, model_used)
            VALUES (:1, :2, :3, :4, :5)
        """
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                vec_str = json.dumps(query_embedding)
                sources_json = json.dumps(sources)
                await cur.execute(sql, [query, vec_str, answer, sources_json, model_used])
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

    def _is_cache_hit(self, similarity: float) -> bool:
        return similarity >= self.similarity_threshold
