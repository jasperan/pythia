"""Deep Research agent — autonomous multi-step research with iterative search and synthesis."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum

from pythia.config import ResearchConfig
from pythia.scraper import scrape_urls
from pythia.server.ollama import OllamaClient
from pythia.server.oracle_cache import OracleCache
from pythia.server.searxng import SearxngClient, SearchResult

logger = logging.getLogger(__name__)


class ResearchEventType(str, Enum):
    STATUS = "status"
    PLAN = "plan"
    ROUND_START = "round_start"
    SUB_QUERY = "sub_query"
    FINDING = "finding"
    GAP_ANALYSIS = "gap_analysis"
    RECALL = "recall"
    TOKEN = "token"
    DONE = "done"


@dataclass
class ResearchEvent:
    event_type: ResearchEventType
    data: dict


@dataclass
class Finding:
    sub_query: str
    summary: str
    sources: list[dict] = field(default_factory=list)
    round_num: int = 1


# --- Prompts ---

_DECOMPOSE_SYSTEM = """You are a research planning agent. Given a research question, decompose it into specific, searchable sub-questions that together would fully answer the original question.

Return a JSON object with a single key "sub_queries" containing a list of strings.
Each sub-query should be a focused, web-searchable question.
Return between 2 and {max_sub_queries} sub-queries.
Order them from most fundamental to most specific."""

_DECOMPOSE_USER = """Research question: {query}

Return JSON: {{"sub_queries": ["question 1", "question 2", ...]}}"""

_SUMMARIZE_SYSTEM = """You are a research analyst. Given search results for a specific sub-question, extract and summarize the key findings.

Be concise but thorough. Include specific facts, numbers, and claims.
Cite sources using [N] notation matching the source indices provided.
If the search results don't adequately answer the sub-question, say so."""

_SUMMARIZE_USER = """Sub-question: {sub_query}

Search Results:
{context}

Summarize the key findings that answer this sub-question. Use [N] citations."""

_GAP_ANALYSIS_SYSTEM = """You are a research analyst reviewing findings so far for a research question. Identify what information is still missing or insufficient.

Return a JSON object with:
- "sufficient": true if the findings adequately answer the original question, false otherwise
- "gaps": a list of specific follow-up questions to search for (empty if sufficient)
- "reasoning": a brief explanation of what's missing

Return at most {max_follow_ups} follow-up questions."""

_GAP_ANALYSIS_USER = """Original research question: {query}

Findings so far:
{findings_text}

Analyze gaps and return JSON: {{"sufficient": true/false, "gaps": [...], "reasoning": "..."}}"""

_REPORT_SYSTEM = """You are Pythia, an AI research engine. Synthesize a comprehensive research report from the collected findings.

Rules:
1. Structure the report with clear markdown sections (## headers)
2. Cite sources using [N] notation inline
3. Start with a brief executive summary
4. Organize findings logically, not by search order
5. End with key takeaways or conclusions
6. Be thorough but avoid repetition
7. If findings from past research sessions are included, integrate them naturally"""

_REPORT_USER = """Research question: {query}

{recall_section}

Collected findings from {num_rounds} round(s) of research:

{findings_text}

All sources:
{sources_text}

Write a comprehensive, well-structured research report."""


class ResearchAgent:
    """Autonomous deep research agent that iteratively searches, analyzes, and synthesizes."""

    # Approximate max chars for synthesis prompt to stay within model context limits.
    # qwen3.5:9b context is 16K tokens (~4 chars/token). Reserve ~4K tokens for
    # system prompt + generation, leaving ~12K tokens (~48K chars) for user content.
    _MAX_SYNTHESIS_CHARS = 40000

    def __init__(
        self,
        ollama: OllamaClient,
        cache: OracleCache,
        searxng: SearxngClient,
        config: ResearchConfig,
    ):
        self.ollama = ollama
        self.cache = cache
        self.searxng = searxng
        self.config = config

    async def research(
        self, query: str, model_override: str | None = None,
    ) -> AsyncIterator[ResearchEvent]:
        """Run autonomous deep research on a query. Yields events as they occur."""
        start = time.monotonic()
        # Use model as a local variable — never mutate self.ollama.model
        model = model_override or self.ollama.model

        all_findings: list[Finding] = []
        all_sources: list[dict] = []
        source_counter = 0

        # Phase 1: Recall related past research
        yield ResearchEvent(ResearchEventType.STATUS, {"message": "Searching knowledge base for related research..."})
        recalled = []
        try:
            recalled = await self.cache.recall_findings(
                query, threshold=self.config.recall_threshold, limit=5,
            )
        except Exception as e:
            logger.debug(f"Recall failed (table may not exist yet): {e}")

        if recalled:
            yield ResearchEvent(ResearchEventType.RECALL, {
                "count": len(recalled),
                "findings": [
                    {"sub_query": r["sub_query"], "similarity": round(r["similarity"], 2), "from_query": r["research_query"]}
                    for r in recalled
                ],
            })

        # Phase 2: Decompose query into sub-questions
        yield ResearchEvent(ResearchEventType.STATUS, {"message": "Planning research strategy..."})
        sub_queries = await self._decompose_query(query, model)
        yield ResearchEvent(ResearchEventType.PLAN, {"sub_queries": sub_queries})

        # Phase 3: Iterative search rounds
        round_num = 1
        for round_num in range(1, self.config.max_rounds + 1):
            yield ResearchEvent(ResearchEventType.ROUND_START, {
                "round": round_num,
                "max_rounds": self.config.max_rounds,
                "num_queries": len(sub_queries),
            })

            # Search all sub-queries in parallel
            round_findings, round_sources, source_counter = await self._search_round(
                sub_queries, round_num, source_counter, model,
            )

            for finding in round_findings:
                yield ResearchEvent(ResearchEventType.FINDING, {
                    "sub_query": finding.sub_query,
                    "summary_preview": finding.summary[:200],
                    "num_sources": len(finding.sources),
                    "round": round_num,
                })

            all_findings.extend(round_findings)
            all_sources.extend(round_sources)

            # Don't analyze gaps on last possible round
            if round_num >= self.config.max_rounds:
                break

            # Phase 4: Analyze gaps
            yield ResearchEvent(ResearchEventType.STATUS, {"message": f"Analyzing research completeness (round {round_num})..."})
            gap_result = await self._analyze_gaps(query, all_findings, model)
            yield ResearchEvent(ResearchEventType.GAP_ANALYSIS, gap_result)

            if gap_result.get("sufficient", True):
                break

            sub_queries = gap_result.get("gaps", [])
            if not sub_queries:
                break

        # Phase 5: Synthesize report
        yield ResearchEvent(ResearchEventType.STATUS, {"message": "Synthesizing research report..."})
        report_parts = []
        async for token in self._synthesize_report(query, all_findings, all_sources, recalled, model):
            report_parts.append(token)
            yield ResearchEvent(ResearchEventType.TOKEN, {"content": token})

        report_text = "".join(report_parts)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Store research session and findings
        try:
            all_sub_queries = list({f.sub_query for f in all_findings})
            research_id = await self.cache.store_research(
                query=query, report=report_text, sub_queries=all_sub_queries,
                rounds_used=round_num, total_sources=len(all_sources),
                model_used=model, elapsed_ms=elapsed_ms,
            )
            if research_id:
                for finding in all_findings:
                    await self.cache.store_finding(
                        research_id=research_id,
                        sub_query=finding.sub_query,
                        summary=finding.summary,
                        sources=finding.sources,
                        round_num=finding.round_num,
                    )
        except Exception as e:
            logger.warning(f"Failed to store research results: {e}")

        # Record in search history
        await self.cache.record_search(
            query=f"[research] {query}", cache_hit=False,
            response_time_ms=elapsed_ms, model_used=model,
        )

        yield ResearchEvent(ResearchEventType.DONE, {
            "rounds_used": round_num,
            "total_findings": len(all_findings),
            "total_sources": len(all_sources),
            "recalled_findings": len(recalled),
            "elapsed_ms": elapsed_ms,
        })

    async def _decompose_query(self, query: str, model: str) -> list[str]:
        """Use LLM to decompose a research question into sub-queries."""
        system = _DECOMPOSE_SYSTEM.format(max_sub_queries=self.config.max_sub_queries)
        user = _DECOMPOSE_USER.format(query=query)

        try:
            response = await self.ollama.generate(system, user, json_mode=True, model=model)
            data = json.loads(response)
            sub_queries = data.get("sub_queries", [])
            if isinstance(sub_queries, list) and sub_queries:
                return sub_queries[:self.config.max_sub_queries]
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse decomposition response: {e}")

        # Fallback: use the original query plus a couple of variations
        return [query, f"What is {query}", f"{query} latest developments"]

    async def _search_round(
        self, sub_queries: list[str], round_num: int, source_counter: int, model: str,
    ) -> tuple[list[Finding], list[dict], int]:
        """Search all sub-queries in parallel, then number sources sequentially."""
        sem = asyncio.Semaphore(3)

        async def _search_one(sq: str) -> tuple[str, list[SearchResult] | None]:
            """Search a single sub-query. Returns (sub_query, results_or_None)."""
            async with sem:
                try:
                    results = await self.searxng.search(sq)
                    return sq, results if results else None
                except Exception as e:
                    logger.warning(f"Search failed for '{sq}': {e}")
                    return sq, None

        # Phase 1: Search all sub-queries in parallel (IO-bound, bounded by semaphore)
        tasks = [_search_one(sq) for sq in sub_queries]
        search_results = await asyncio.gather(*tasks)

        # Phase 2: Assign source numbers sequentially — no race condition
        indexed_items: list[tuple[str, list[dict], list[SearchResult]]] = []
        pre_built: list[Finding] = []
        for sq, results in search_results:
            if results is None:
                pre_built.append(Finding(sub_query=sq, summary="No results found.", sources=[], round_num=round_num))
                continue
            round_sources = []
            for r in results:
                source_counter += 1
                round_sources.append({
                    "index": source_counter, "title": r.title,
                    "url": r.url, "snippet": r.snippet,
                })
            indexed_items.append((sq, round_sources, results))

        # Phase 3: Scrape + summarize in parallel (bounded by semaphore)
        async def _scrape_and_summarize(
            sq: str, round_sources: list[dict], results: list[SearchResult],
        ) -> tuple[Finding, list[dict]]:
            async with sem:
                scraped_content: dict[str, str] = {}
                if self.config.deep_scrape:
                    urls_snippets = [(r.url, r.snippet) for r in results[:3]]
                    scraped = await scrape_urls(urls_snippets)
                    scraped_content = {s.url: s.content for s in scraped}

                context_parts = []
                for s in round_sources:
                    content = scraped_content.get(s["url"], s["snippet"])
                    context_parts.append(f"[{s['index']}] {s['title']}\nURL: {s['url']}\n{content}")
                context = "\n\n".join(context_parts)

                system = _SUMMARIZE_SYSTEM
                user = _SUMMARIZE_USER.format(sub_query=sq, context=context)
                try:
                    summary = await self.ollama.generate(system, user, model=model)
                except Exception as e:
                    logger.warning(f"Summarization failed for '{sq}': {e}")
                    summary = f"Search returned {len(results)} results but summarization failed."

                finding = Finding(sub_query=sq, summary=summary, sources=round_sources, round_num=round_num)
                return finding, round_sources

        findings: list[Finding] = list(pre_built)
        all_sources: list[dict] = []

        if indexed_items:
            summarize_tasks = [_scrape_and_summarize(sq, src, res) for sq, src, res in indexed_items]
            summarize_results = await asyncio.gather(*summarize_tasks)
            for finding, sources in summarize_results:
                findings.append(finding)
                all_sources.extend(sources)

        return findings, all_sources, source_counter

    async def _analyze_gaps(self, query: str, findings: list[Finding], model: str) -> dict:
        """Use LLM to identify gaps in current research findings."""
        findings_text = "\n\n".join(
            f"### {f.sub_query}\n{f.summary}" for f in findings
        )
        system = _GAP_ANALYSIS_SYSTEM.format(max_follow_ups=self.config.max_sub_queries)
        user = _GAP_ANALYSIS_USER.format(query=query, findings_text=findings_text)

        try:
            response = await self.ollama.generate(system, user, json_mode=True, model=model)
            data = json.loads(response)
            return {
                "sufficient": data.get("sufficient", True),
                "gaps": data.get("gaps", [])[:self.config.max_sub_queries],
                "reasoning": data.get("reasoning", ""),
            }
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse gap analysis: {e}")
            return {"sufficient": True, "gaps": [], "reasoning": "Gap analysis failed, proceeding with synthesis."}

    async def _synthesize_report(
        self, query: str, findings: list[Finding],
        all_sources: list[dict], recalled: list[dict], model: str,
    ) -> AsyncIterator[str]:
        """Stream the final research report synthesis."""
        findings_text = "\n\n".join(
            f"### {f.sub_query} (Round {f.round_num})\n{f.summary}" for f in findings
        )

        sources_text = "\n".join(
            f"[{s['index']}] {s['title']} — {s['url']}" for s in all_sources
        )

        recall_section = ""
        if recalled:
            recall_parts = ["Related findings from past research sessions:"]
            for r in recalled:
                recall_parts.append(
                    f"- From \"{r['research_query']}\" (similarity {r['similarity']:.2f}): {r['summary'][:300]}"
                )
            recall_section = "\n".join(recall_parts)

        num_rounds = max((f.round_num for f in findings), default=1)

        user = _REPORT_USER.format(
            query=query,
            recall_section=recall_section,
            num_rounds=num_rounds,
            findings_text=findings_text,
            sources_text=sources_text,
        )

        # Truncate if prompt exceeds model context limits
        if len(user) > self._MAX_SYNTHESIS_CHARS:
            logger.warning(f"Synthesis prompt too long ({len(user)} chars), truncating to {self._MAX_SYNTHESIS_CHARS}")
            user = user[:self._MAX_SYNTHESIS_CHARS] + "\n\n[Content truncated for context length. Synthesize from available findings.]"

        async for token in self.ollama.generate_stream(_REPORT_SYSTEM, user, model=model):
            yield token
