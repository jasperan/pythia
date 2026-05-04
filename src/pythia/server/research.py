"""Deep Research agent — autonomous multi-step research with iterative search, synthesis, verification, and provenance."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from pythia.config import ResearchConfig
from pythia.provenance import ProvenanceRecord
from pythia.scraper import scrape_urls
from pythia.server.llm_client import LLMClient
from pythia.server.oracle_cache import OracleCache
from pythia.server.searxng import SearxngClient, SearchResult
from pythia.skills import SkillLoader
from pythia.verification import verify_report
from pythia.workspace import WorkspaceChangelog, generate_slug

logger = logging.getLogger(__name__)


class ResearchEventType(StrEnum):
    STATUS = "status"
    PLAN = "plan"
    ROUND_START = "round_start"
    FINDING = "finding"
    GAP_ANALYSIS = "gap_analysis"
    RECALL = "recall"
    TOKEN = "token"
    VERIFY = "verify"
    COMPLETENESS_CHECK = "completeness_check"
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
    error: str | None = None


# --- Prompts ---

_DECOMPOSE_SYSTEM = """You are a research planning agent. Given a research question, decompose it into specific, searchable sub-questions that together would fully answer the original question.

Return a JSON object with a single key "sub_queries" containing a list of strings.
Each sub-query should be a focused, web-searchable question.
Return between 2 and {max_sub_queries} sub-queries.
Order them from most fundamental to most specific."""

_DECOMPOSE_USER = """Research question: {query}

Return JSON: {{"sub_queries": ["question 1", "question 2", ...]}}"""

_SUMMARIZE_SYSTEM = """You are a research analyst. Given search results for a specific sub-question, extract and summarize the key findings.

Use only the supplied search-result text. Do not infer facts from the title,
URL, or prior knowledge if the provided text does not support them.
Be concise but thorough. Include specific facts, numbers, and claims only when
they appear in the supplied text.
Cite sources using [N] notation matching the source indices provided.
When sources conflict, prioritize official documentation, primary sources, and
standards over blogs, forums, social media, or generated summaries.
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
7. If findings from past research sessions are included, integrate them naturally
8. Use only claims supported by the collected findings and source list
9. Do not introduce facts from prior knowledge, assumptions, or source titles alone
10. Prefer official documentation and primary sources when evidence conflicts
11. If the evidence is weak, conflicting, or missing, say so instead of filling the gap"""

_REPORT_USER = """Research question: {query}

{recall_section}

Collected findings from {num_rounds} round(s) of research:

{findings_text}

All sources:
{sources_text}

Write a comprehensive, well-structured research report."""

_REPAIR_SYSTEM = """You are a research report repair agent. Rewrite the report so every factual claim is grounded in the supplied findings and source excerpts.

Rules:
1. Remove unsupported claims instead of trying to preserve them
2. Keep only citations that correspond to supplied source indices
3. Prefer official documentation and primary sources when evidence conflicts
4. Explicitly state when evidence is missing or inconclusive
5. Preserve useful structure, but correctness is more important than length
6. Every factual sentence or bullet must include at least one numeric citation like [1]
7. Do not use placeholder citations such as [N], [N1], [source], or bare URLs
8. If you cannot cite a claim with a supplied source index, remove the claim"""

_REPAIR_USER = """Original research question: {query}

Verification result:
Status: {status}
Summary: {summary}
Issues:
{issues}

Current report:
{report}

Collected findings:
{findings_text}

Sources:
{sources_text}

Rewrite the report so unsupported claims are removed or corrected. Return only the repaired markdown report."""

_COMPLETENESS_SYSTEM = """You are a research completeness verifier. Given a research question and the synthesized report, determine if the report adequately answers the original question.

Return a JSON object:
{{
  "status": "COMPLETE" | "INCOMPLETE" | "STUCK",
  "reasoning": "Brief explanation of your assessment",
  "follow_up_queries": ["query 1", "query 2"]
}}

Rules:
- COMPLETE: The report thoroughly answers the question with supporting evidence.
- INCOMPLETE: The report has significant gaps. Provide follow_up_queries to fill them (max {max_follow_ups}).
- STUCK: The question cannot be answered with web search (too niche, future event, requires private data). Provide empty follow_up_queries.
- Only mark INCOMPLETE for substantive gaps, not minor details.
- follow_up_queries should be specific, web-searchable questions."""

_COMPLETENESS_USER = """Original research question: {query}

Synthesized report:
{report}

Assess completeness and return JSON."""


_CONTINUE_DECOMPOSE_SYSTEM = """You are a research planning agent continuing a prior investigation. Given the original research question, prior findings, and an optional focus area, generate follow-up sub-questions that fill remaining gaps or explore new angles.

Return a JSON object with a single key "sub_queries" containing a list of strings.
Each sub-query should target information NOT already covered by the prior findings.
Return between 2 and {max_sub_queries} sub-queries.
If a focus is provided, prioritize questions related to it."""

_CONTINUE_DECOMPOSE_USER = """Original research question: {query}

{focus_section}

Prior findings:
{prior_findings_text}

Generate follow-up sub-queries targeting gaps or new angles. Return JSON: {{"sub_queries": ["question 1", ...]}}"""

_REFINE_DECOMPOSE_SYSTEM = """You are a research planning agent refining a prior investigation. The user wants to go deeper on a specific aspect. Given the original question, prior report, and the user's directive, generate focused sub-questions that address the directive.

Return a JSON object with a single key "sub_queries" containing a list of strings.
Each sub-query should be tightly focused on the user's directive.
Return between 2 and {max_sub_queries} sub-queries."""

_REFINE_DECOMPOSE_USER = """Original research question: {query}

User's directive: {directive}

Prior report summary (first 2000 chars):
{report_preview}

Generate sub-queries focused on the directive. Return JSON: {{"sub_queries": ["question 1", ...]}}"""

_REFINE_REPORT_SYSTEM = """You are Pythia, an AI research engine. You are refining a prior research report with new focused findings.

Rules:
1. Start with the context from the prior report
2. Integrate the new findings into the appropriate sections
3. Expand the sections related to the user's directive with the new detail
4. Keep the overall report structure coherent
5. Use [N] citation notation for new sources
6. Clearly mark what is new vs. what was in the prior report where helpful"""

_REFINE_REPORT_USER = """Original research question: {query}

User's refinement directive: {directive}

Prior report:
{prior_report}

New focused findings:
{new_findings_text}

New sources:
{new_sources_text}

Write an updated, comprehensive report integrating the new findings."""


class ResearchAgent:
    """Autonomous deep research agent with verification and provenance tracking."""

    _MAX_SYNTHESIS_CHARS = 40000
    _MAX_SUMMARY_CONTEXT_CHARS = 12000
    _SUMMARY_RETRY_CONTEXT_CHARS = 6000

    def __init__(
        self,
        ollama: LLMClient,
        cache: OracleCache,
        searxng: SearxngClient,
        config: ResearchConfig,
        skills_dir: Path | str | None = None,
        workspace_dir: Path | str | None = None,
    ):
        self.ollama = ollama
        self.cache = cache
        self.searxng = searxng
        self.config = config
        self.skills = SkillLoader(skills_dir)
        self.changelog = WorkspaceChangelog(workspace_dir)

    async def research(
        self, query: str, model_override: str | None = None,
        skill_override: str | None = None,
    ) -> AsyncIterator[ResearchEvent]:
        """Run autonomous deep research on a query. Yields events as they occur."""
        start = time.monotonic()
        model = model_override or self.ollama.model

        skill = self.skills.get(skill_override) if skill_override else self.skills.match(query)
        effective_max_rounds = skill.max_rounds if skill and skill.max_rounds else self.config.max_rounds
        effective_max_sub = skill.max_sub_queries if skill and skill.max_sub_queries else self.config.max_sub_queries
        effective_scrape = skill.requires_scrape if skill and skill.requires_scrape is not None else self.config.deep_scrape

        if skill:
            logger.info(f"Using skill: {skill.name}")

        slug = generate_slug(query)
        all_findings: list[Finding] = []
        all_sources: list[dict] = []
        source_counter = 0
        seen_source_urls: set[str] = set()

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

        yield ResearchEvent(ResearchEventType.STATUS, {"message": "Planning research strategy..."})
        sub_queries = await self._decompose_query(query, model, max_sub=effective_max_sub)
        yield ResearchEvent(ResearchEventType.PLAN, {"sub_queries": sub_queries, "slug": slug})

        self.changelog.append_entry(slug, "Research started", f"Query: {query}", "in_progress", "Execute research rounds")

        round_num = 1
        for round_num in range(1, effective_max_rounds + 1):
            yield ResearchEvent(ResearchEventType.ROUND_START, {
                "round": round_num,
                "max_rounds": effective_max_rounds,
                "num_queries": len(sub_queries),
            })

            round_findings, round_sources, source_counter = await self._search_round(
                sub_queries, round_num, source_counter, model,
                scrape=effective_scrape, seen_source_urls=seen_source_urls,
            )

            for finding in round_findings:
                yield ResearchEvent(ResearchEventType.FINDING, {
                    "sub_query": finding.sub_query,
                    "summary_preview": finding.summary[:200],
                    "num_sources": len(finding.sources),
                    "round": round_num,
                    "summary_failed": finding.error is not None,
                    "error": finding.error,
                })

            all_findings.extend(round_findings)
            all_sources.extend(round_sources)

            if round_num >= effective_max_rounds:
                break

            yield ResearchEvent(ResearchEventType.STATUS, {"message": f"Analyzing research completeness (round {round_num})..."})
            gap_result = await self._analyze_gaps(
                query, all_findings, model, max_follow_ups=effective_max_sub,
            )
            yield ResearchEvent(ResearchEventType.GAP_ANALYSIS, gap_result)

            if gap_result.get("sufficient", True):
                break

            sub_queries = gap_result.get("gaps", [])
            if not sub_queries:
                break

            self.changelog.append_entry(
                slug, f"Round {round_num} complete — {len(round_findings)} findings",
                f"Gaps: {', '.join(gap_result.get('gaps', [])[:3])}" if not gap_result.get('sufficient') else "No significant gaps",
                "in_progress", f"Continue with round {round_num + 1}",
            )

        # --- Completeness verification loop ---
        max_cc = self.config.max_completeness_checks
        report_text = ""

        for cc_attempt in range(max_cc + 1):
            yield ResearchEvent(ResearchEventType.STATUS, {"message": "Synthesizing research report..."})
            report_parts = []
            async for token in self._synthesize_report(query, all_findings, all_sources, recalled, model):
                report_parts.append(token)
                yield ResearchEvent(ResearchEventType.TOKEN, {"content": token})

            report_text = self._normalize_citation_markers("".join(report_parts))

            if cc_attempt >= max_cc:
                break

            yield ResearchEvent(ResearchEventType.STATUS, {"message": f"Verifying report completeness (check {cc_attempt + 1}/{max_cc})..."})
            cc_result = await self._verify_completeness(
                query, report_text, model, max_follow_ups=effective_max_sub,
            )
            yield ResearchEvent(ResearchEventType.COMPLETENESS_CHECK, {
                "attempt": cc_attempt + 1,
                "max_attempts": max_cc,
                **cc_result,
            })

            if cc_result["status"] in ("COMPLETE", "STUCK"):
                break

            follow_ups = cc_result.get("follow_up_queries", [])
            if not follow_ups:
                break

            yield ResearchEvent(ResearchEventType.STATUS, {"message": f"Report incomplete, running {len(follow_ups)} follow-up queries..."})
            round_num += 1
            yield ResearchEvent(ResearchEventType.ROUND_START, {
                "round": round_num,
                "max_rounds": effective_max_rounds + max_cc,
                "num_queries": len(follow_ups),
            })

            extra_findings, extra_sources, source_counter = await self._search_round(
                follow_ups, round_num, source_counter, model,
                scrape=effective_scrape, seen_source_urls=seen_source_urls,
            )

            for finding in extra_findings:
                yield ResearchEvent(ResearchEventType.FINDING, {
                    "sub_query": finding.sub_query,
                    "summary_preview": finding.summary[:200],
                    "num_sources": len(finding.sources),
                    "round": round_num,
                    "summary_failed": finding.error is not None,
                    "error": finding.error,
                })

            all_findings.extend(extra_findings)
            all_sources.extend(extra_sources)

        yield ResearchEvent(ResearchEventType.STATUS, {"message": "Verifying research output..."})
        report_text, verification, verify_events = await self._verify_and_repair_report(
            query, report_text, all_findings, all_sources, model,
        )
        for verify_event in verify_events:
            yield ResearchEvent(ResearchEventType.VERIFY, verify_event)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        provenance = ProvenanceRecord(
            topic=query, slug=slug, rounds=round_num,
            sources_consulted=len(all_sources),
            sources_accepted=len(all_sources),
            sources_rejected=0,
            verification_status=verification.status,
            verification_summary=verification.summary,
            model_used=model, elapsed_ms=elapsed_ms,
        )
        corpus_path = self._write_corpus(
            slug=slug,
            query=query,
            report=report_text,
            findings=all_findings,
            sources=all_sources,
            provenance=provenance,
            kind="research",
        )
        provenance.research_files = [corpus_path]

        try:
            all_sub_queries = list({f.sub_query for f in all_findings})
            research_id = await self.cache.store_research(
                query=query, report=report_text, sub_queries=all_sub_queries,
                rounds_used=round_num, total_sources=len(all_sources),
                model_used=model, elapsed_ms=elapsed_ms, slug=slug,
                provenance=provenance.to_markdown(),
                sources_consulted=provenance.sources_consulted,
                sources_accepted=provenance.sources_accepted,
                sources_rejected=provenance.sources_rejected,
                verification_status=provenance.verification_status,
                verification_summary=provenance.verification_summary,
            )
            if research_id:
                await self.cache.store_findings_batch(
                    research_id=research_id,
                    findings=[
                        {
                            "sub_query": f.sub_query,
                            "summary": f.summary,
                            "sources": f.sources,
                            "round_num": f.round_num,
                        }
                        for f in all_findings
                    ],
                )
        except Exception as e:
            logger.warning(f"Failed to store research results: {e}")

        self.changelog.append_entry(
            slug, "Research complete",
            f"Report: {len(report_text)} chars, {len(all_findings)} findings, {len(all_sources)} sources. Verification: {verification.status}",
            "completed", "Review report and provenance",
        )

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
            "slug": slug,
            "verification_status": verification.status,
            "provenance": provenance.to_dict(),
            "corpus_path": corpus_path,
            "failed_findings": sum(1 for f in all_findings if f.error),
        })

    async def refine_research(
        self, slug: str, directive: str,
        model_override: str | None = None,
    ) -> AsyncIterator[ResearchEvent]:
        """Refine a prior investigation with a user-directed focus area."""
        start = time.monotonic()
        model = model_override or self.ollama.model

        yield ResearchEvent(ResearchEventType.STATUS, {"message": f"Loading investigation '{slug}' for refinement..."})

        prior = await self.cache.get_research_by_slug(slug)
        if not prior:
            yield ResearchEvent(ResearchEventType.DONE, {
                "error": f"No investigation found with slug '{slug}'",
                "rounds_used": 0, "total_findings": 0, "total_sources": 0,
                "recalled_findings": 0, "elapsed_ms": 0, "slug": slug,
                "verification_status": "fail", "provenance": {},
            })
            return

        prior_findings_raw = await self.cache.get_findings_for_research(prior["id"])

        yield ResearchEvent(ResearchEventType.STATUS, {
            "message": f"Loaded prior report. Generating focused queries for: {directive[:100]}",
        })

        original_query = prior["query"]
        effective_max_rounds = self.config.max_rounds
        effective_max_sub = self.config.max_sub_queries
        effective_scrape = self.config.deep_scrape

        # Decompose with directive focus
        report_preview = (prior.get("report") or "")[:2000]
        system = _REFINE_DECOMPOSE_SYSTEM.format(max_sub_queries=effective_max_sub)
        user = _REFINE_DECOMPOSE_USER.format(
            query=original_query, directive=directive, report_preview=report_preview,
        )

        try:
            response = await self.ollama.generate(system, user, json_mode=True, model=model)
            data = json.loads(response)
            sub_queries = data.get("sub_queries", [])[:effective_max_sub]
            if not sub_queries:
                sub_queries = [directive, f"{original_query} {directive}"]
        except (json.JSONDecodeError, KeyError):
            sub_queries = [directive, f"{original_query} {directive}"]

        yield ResearchEvent(ResearchEventType.PLAN, {"sub_queries": sub_queries, "slug": slug})

        new_findings: list[Finding] = []
        all_sources: list[dict] = []
        source_counter = 0
        seen_source_urls = {
            source.get("url")
            for finding in prior_findings_raw
            for source in finding.get("sources", [])
            if source.get("url")
        }

        round_num = 0
        for round_num in range(1, effective_max_rounds + 1):
            yield ResearchEvent(ResearchEventType.ROUND_START, {
                "round": round_num,
                "max_rounds": effective_max_rounds,
                "num_queries": len(sub_queries),
            })

            round_findings, round_sources, source_counter = await self._search_round(
                sub_queries, round_num, source_counter, model,
                scrape=effective_scrape, seen_source_urls=seen_source_urls,
            )

            for finding in round_findings:
                yield ResearchEvent(ResearchEventType.FINDING, {
                    "sub_query": finding.sub_query,
                    "summary_preview": finding.summary[:200],
                    "num_sources": len(finding.sources),
                    "round": round_num,
                    "summary_failed": finding.error is not None,
                    "error": finding.error,
                })

            new_findings.extend(round_findings)
            all_sources.extend(round_sources)

            if round_num >= effective_max_rounds:
                break

            # Gap analysis against directive, not just original query
            all_so_far = [
                Finding(sub_query=f["sub_query"], summary=f["summary"], sources=f["sources"], round_num=f["round_num"])
                for f in prior_findings_raw
            ] + new_findings
            gap_result = await self._analyze_gaps(
                f"{original_query} — focusing on: {directive}",
                all_so_far,
                model,
                max_follow_ups=effective_max_sub,
            )
            yield ResearchEvent(ResearchEventType.GAP_ANALYSIS, gap_result)

            if gap_result.get("sufficient", True):
                break
            sub_queries = gap_result.get("gaps", [])
            if not sub_queries:
                break

        if not round_num:
            round_num = 1

        # Synthesize merged report using refinement-specific prompt
        new_findings_text = self._format_findings_for_prompt(new_findings)
        new_sources_text = self._format_sources_for_prompt(all_sources)
        prior_report = prior.get("report") or ""

        refine_user = _REFINE_REPORT_USER.format(
            query=original_query, directive=directive,
            prior_report=prior_report, new_findings_text=new_findings_text,
            new_sources_text=new_sources_text,
        )
        if len(refine_user) > self._MAX_SYNTHESIS_CHARS:
            refine_user = refine_user[:self._MAX_SYNTHESIS_CHARS] + "\n\n[Content truncated.]"

        yield ResearchEvent(ResearchEventType.STATUS, {"message": "Synthesizing refined report..."})
        report_parts = []
        async for token in self.ollama.generate_stream(_REFINE_REPORT_SYSTEM, refine_user, model=model):
            report_parts.append(token)
            yield ResearchEvent(ResearchEventType.TOKEN, {"content": token})

        report_text = self._normalize_citation_markers("".join(report_parts))

        yield ResearchEvent(ResearchEventType.STATUS, {"message": "Verifying refined report..."})
        report_text, verification, verify_events = await self._verify_and_repair_report(
            original_query, report_text, new_findings, all_sources, model,
        )
        for verify_event in verify_events:
            yield ResearchEvent(ResearchEventType.VERIFY, verify_event)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        provenance = ProvenanceRecord(
            topic=original_query, slug=slug, rounds=round_num,
            sources_consulted=len(all_sources),
            sources_accepted=len(all_sources),
            sources_rejected=0,
            verification_status=verification.status,
            verification_summary=verification.summary,
            model_used=model, elapsed_ms=elapsed_ms,
        )
        corpus_path = self._write_corpus(
            slug=slug,
            query=original_query,
            report=report_text,
            findings=new_findings,
            sources=all_sources,
            provenance=provenance,
            kind="refinement",
            directive=directive,
        )
        provenance.research_files = [corpus_path]

        try:
            all_sub_queries = list({f.sub_query for f in new_findings})
            research_id = await self.cache.store_research(
                query=original_query, report=report_text, sub_queries=all_sub_queries,
                rounds_used=round_num, total_sources=len(all_sources),
                model_used=model, elapsed_ms=elapsed_ms, slug=slug,
                parent_id=prior["id"],
                verification_status=verification.status,
                verification_summary=verification.summary,
                provenance=provenance.to_markdown(),
            )
            if research_id and new_findings:
                await self.cache.store_findings_batch(
                    research_id=research_id,
                    findings=[
                        {
                            "sub_query": f.sub_query, "summary": f.summary,
                            "sources": f.sources, "round_num": f.round_num,
                        }
                        for f in new_findings
                    ],
                )
        except Exception as e:
            logger.warning(f"Failed to store refinement results: {e}")

        yield ResearchEvent(ResearchEventType.DONE, {
            "rounds_used": round_num,
            "total_findings": len(new_findings),
            "total_sources": len(all_sources),
            "refined_from": slug,
            "directive": directive,
            "recalled_findings": 0,
            "elapsed_ms": elapsed_ms,
            "slug": slug,
            "verification_status": verification.status,
            "provenance": provenance.to_dict(),
            "corpus_path": corpus_path,
            "failed_findings": sum(1 for f in new_findings if f.error),
        })

    async def _decompose_query(self, query: str, model: str, max_sub: int | None = None) -> list[str]:
        max_sub = max_sub or self.config.max_sub_queries
        system = _DECOMPOSE_SYSTEM.format(max_sub_queries=max_sub)
        user = _DECOMPOSE_USER.format(query=query)

        try:
            response = await self.ollama.generate(system, user, json_mode=True, model=model)
            data = json.loads(response)
            sub_queries = data.get("sub_queries", [])
            if isinstance(sub_queries, list) and sub_queries:
                return sub_queries[:max_sub]
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse decomposition response: {e}")

        # Fallback: use the original query plus a couple of variations
        return [query, f"What is {query}", f"{query} latest developments"][:max_sub]

    async def _search_round(
        self, sub_queries: list[str], round_num: int, source_counter: int, model: str,
        scrape: bool = True,
        seen_source_urls: set[str] | None = None,
    ) -> tuple[list[Finding], list[dict], int]:
        sem = asyncio.Semaphore(3)
        seen_source_urls = seen_source_urls if seen_source_urls is not None else set()

        async def _search_one(sq: str) -> tuple[str, list[SearchResult] | None]:
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
            new_results = []
            for r in results:
                if r.url in seen_source_urls:
                    continue
                seen_source_urls.add(r.url)
                new_results.append(r)
                source_counter += 1
                round_sources.append({
                    "index": source_counter, "title": r.title,
                    "url": r.url, "snippet": r.snippet,
                })
            if not round_sources:
                pre_built.append(Finding(
                    sub_query=sq,
                    summary="No new sources found; all search results were already used.",
                    sources=[],
                    round_num=round_num,
                ))
                continue
            indexed_items.append((sq, round_sources, new_results))

        # Phase 3: Scrape + summarize in parallel (bounded by semaphore)
        async def _scrape_and_summarize(
            sq: str, round_sources: list[dict], results: list[SearchResult],
        ) -> tuple[Finding, list[dict]]:
            async with sem:
                scraped_content: dict[str, str] = {}
                if scrape:
                    urls_snippets = [(r.url, r.snippet) for r in results[:3]]
                    scraped = await scrape_urls(urls_snippets)
                    scraped_content = {s.url: s.content for s in scraped}

                context_parts = []
                for s in round_sources:
                    content = scraped_content.get(s["url"], s["snippet"])
                    context_parts.append(f"[{s['index']}] {s['title']}\nURL: {s['url']}\n{content}")
                context = "\n\n".join(context_parts)

                summary, error = await self._summarize_with_retries(sq, context, len(results), model)

                finding = Finding(
                    sub_query=sq,
                    summary=summary,
                    sources=round_sources,
                    round_num=round_num,
                    error=error,
                )
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

    async def _summarize_with_retries(
        self,
        sub_query: str,
        context: str,
        result_count: int,
        model: str,
    ) -> tuple[str, str | None]:
        last_error: Exception | None = None
        context_limits = [self._MAX_SUMMARY_CONTEXT_CHARS, self._SUMMARY_RETRY_CONTEXT_CHARS]

        for attempt, limit in enumerate(context_limits, 1):
            clipped_context = context[:limit]
            if len(context) > limit:
                clipped_context += "\n\n[Search context truncated for summarization.]"
            user = _SUMMARIZE_USER.format(sub_query=sub_query, context=clipped_context)
            try:
                summary = await self.ollama.generate(_SUMMARIZE_SYSTEM, user, model=model)
                return self._normalize_citation_markers(summary), None
            except Exception as e:
                last_error = e
                logger.warning(
                    "Summarization attempt %s/%s failed for %r: %s",
                    attempt,
                    len(context_limits),
                    sub_query,
                    e,
                )

        error = type(last_error).__name__ if last_error else "UnknownError"
        return f"Search returned {result_count} results but summarization failed after retry.", error

    async def _analyze_gaps(
        self,
        query: str,
        findings: list[Finding],
        model: str,
        max_follow_ups: int | None = None,
    ) -> dict:
        """Use LLM to identify gaps in current research findings."""
        max_follow_ups = max_follow_ups or self.config.max_sub_queries
        findings_text = "\n\n".join(
            f"### {f.sub_query}\n{f.summary}" for f in findings
        )
        system = _GAP_ANALYSIS_SYSTEM.format(max_follow_ups=max_follow_ups)
        user = _GAP_ANALYSIS_USER.format(query=query, findings_text=findings_text)

        try:
            response = await self.ollama.generate(system, user, json_mode=True, model=model)
            data = json.loads(response)
            return {
                "sufficient": data.get("sufficient", True),
                "gaps": data.get("gaps", [])[:max_follow_ups],
                "reasoning": data.get("reasoning", ""),
            }
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse gap analysis: {e}")
            return {"sufficient": True, "gaps": [], "reasoning": "Gap analysis failed, proceeding with synthesis."}

    async def _verify_completeness(
        self,
        query: str,
        report: str,
        model: str,
        max_follow_ups: int | None = None,
    ) -> dict:
        """Post-synthesis completeness verification. Returns status, reasoning, follow_up_queries."""
        max_follow_ups = max_follow_ups or self.config.max_sub_queries
        system = _COMPLETENESS_SYSTEM.format(max_follow_ups=max_follow_ups)
        user = _COMPLETENESS_USER.format(query=query, report=report)

        try:
            response = await self.ollama.generate(system, user, json_mode=True, model=model)
            data = json.loads(response)
            return {
                "status": data.get("status", "COMPLETE"),
                "reasoning": data.get("reasoning", ""),
                "follow_up_queries": data.get("follow_up_queries", [])[:max_follow_ups],
            }
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Completeness verification failed: {e}")
            return {"status": "COMPLETE", "reasoning": "Verification failed, proceeding.", "follow_up_queries": []}

    async def _verify_and_repair_report(
        self,
        query: str,
        report_text: str,
        findings: list[Finding],
        all_sources: list[dict],
        model: str,
    ) -> tuple[str, object, list[dict]]:
        verification = await verify_report(self.ollama, query, report_text, all_sources, model)
        events = [self._verification_event_data(verification, attempt=1, repaired=False)]

        if verification.status != "fail":
            return report_text, verification, events

        repaired = await self._repair_report(
            query=query,
            report_text=report_text,
            findings=findings,
            all_sources=all_sources,
            verification=verification,
            model=model,
        )
        if not repaired or repaired.strip() == report_text.strip():
            return report_text, verification, events

        report_text = self._normalize_citation_markers(repaired.strip())
        verification = await verify_report(self.ollama, query, report_text, all_sources, model)
        events.append(self._verification_event_data(verification, attempt=2, repaired=True))
        if verification.status == "fail":
            report_text = self._build_evidence_ledger_report(query, findings)
            verification = await verify_report(self.ollama, query, report_text, all_sources, model)
            events.append(
                self._verification_event_data(
                    verification, attempt=3, repaired=True, fallback=True,
                )
            )
        return report_text, verification, events

    async def _repair_report(
        self,
        *,
        query: str,
        report_text: str,
        findings: list[Finding],
        all_sources: list[dict],
        verification,
        model: str,
    ) -> str:
        issues = "\n".join(
            f"- {issue.get('severity', 'unknown')} {issue.get('type', 'issue')}: "
            f"{issue.get('claim', '')} — {issue.get('explanation', '')}"
            for issue in verification.issues
        ) or "(No structured issues returned.)"

        user = _REPAIR_USER.format(
            query=query,
            status=verification.status,
            summary=verification.summary,
            issues=issues,
            report=report_text[:12000],
            findings_text=self._format_findings_for_prompt(findings),
            sources_text=self._format_sources_for_prompt(all_sources),
        )
        if len(user) > self._MAX_SYNTHESIS_CHARS:
            user = user[:self._MAX_SYNTHESIS_CHARS] + "\n\n[Content truncated for context length. Repair from available evidence.]"

        try:
            repaired = await self.ollama.generate(_REPAIR_SYSTEM, user, model=model)
            return self._normalize_citation_markers(repaired)
        except Exception as e:
            logger.warning("Report repair failed: %s", e)
            return report_text

    def _verification_event_data(
        self, verification, *, attempt: int, repaired: bool, fallback: bool = False,
    ) -> dict:
        return {
            "status": verification.status,
            "claims_checked": verification.claims_checked,
            "issues_found": len(verification.issues),
            "summary": verification.summary,
            "attempt": attempt,
            "repaired": repaired,
            "fallback": fallback,
        }

    async def _synthesize_report(
        self, query: str, findings: list[Finding],
        all_sources: list[dict], recalled: list[dict], model: str,
    ) -> AsyncIterator[str]:
        """Stream the final research report synthesis."""
        findings_text = self._format_findings_for_prompt(findings)
        sources_text = self._format_sources_for_prompt(all_sources)

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

    def _format_findings_for_prompt(self, findings: list[Finding]) -> str:
        return "\n\n".join(
            f"### {f.sub_query} (Round {f.round_num})\n{f.summary}" for f in findings
        )

    def _format_sources_for_prompt(self, sources: list[dict]) -> str:
        parts = []
        for source in sources:
            snippet = source.get("snippet", "")
            excerpt = f"\nExcerpt: {snippet[:700]}" if snippet else ""
            parts.append(
                f"[{source.get('index', '?')}] {source.get('title', 'Untitled')} — "
                f"{source.get('url', 'no URL')}{excerpt}"
            )
        return "\n\n".join(parts)

    def _normalize_citation_markers(self, text: str) -> str:
        return re.sub(r"\[N(\d+)\]", r"[\1]", text)

    def _build_evidence_ledger_report(self, query: str, findings: list[Finding]) -> str:
        lines = [
            "# Evidence Ledger Report",
            "",
            f"Research question: {query}",
            "",
            "The narrative synthesis failed verification, so this fallback report lists only collected source excerpts and search summaries with explicit citations.",
            "",
            "## Source-Backed Findings",
            "",
        ]
        for finding in findings:
            lines.extend([
                f"### Round {finding.round_num}: {finding.sub_query}",
                "",
                f"Search returned {len(finding.sources)} source(s) for this sub-question.",
            ])
            if finding.error:
                lines.append(f"Summary generation failed with {finding.error}.")
            if finding.sources:
                lines.extend(["", "Source excerpts:"])
                for source in finding.sources:
                    index = source.get("index", "?")
                    title = source.get("title", "Untitled")
                    snippet = source.get("snippet", "").strip() or "No source excerpt was available."
                    lines.append(f"- [{index}] {title}: {snippet}")
            lines.append("")
        return "\n".join(lines).strip()

    def _write_corpus(
        self,
        *,
        slug: str,
        query: str,
        report: str,
        findings: list[Finding],
        sources: list[dict],
        provenance: ProvenanceRecord,
        kind: str,
        directive: str | None = None,
    ) -> str:
        corpus_dir = self.changelog.workspace_dir / ".pythia" / "research" / slug
        corpus_dir.mkdir(parents=True, exist_ok=True)
        filename = "corpus.md" if kind == "research" else f"{kind}-corpus.md"
        path = corpus_dir / filename

        lines = [
            f"# Research Corpus: {query}",
            "",
            f"- Slug: `{slug}`",
            f"- Type: `{kind}`",
            f"- Rounds: {provenance.rounds}",
            f"- Sources: {len(sources)}",
            f"- Verification: {provenance.verification_status}",
        ]
        if directive:
            lines.append(f"- Directive: {directive}")
        lines.extend(["", "## Final Report", "", report.strip(), "", "## Findings", ""])

        for idx, finding in enumerate(findings, 1):
            lines.extend([
                f"### Finding {idx}: {finding.sub_query}",
                "",
                f"- Round: {finding.round_num}",
                f"- Source count: {len(finding.sources)}",
                f"- Status: {'failed' if finding.error else 'ok'}",
                "",
                finding.summary.strip(),
                "",
            ])

        lines.extend(["## Sources", ""])
        for source in sources:
            index = source.get("index", "?")
            title = source.get("title", "Untitled")
            url = source.get("url", "")
            snippet = source.get("snippet", "")
            lines.extend([
                f"### [{index}] {title}",
                "",
                f"- URL: {url}",
            ])
            if snippet:
                lines.extend(["", snippet.strip()])
            lines.append("")

        lines.extend(["## Provenance", "", provenance.to_markdown().strip(), ""])
        path.write_text("\n".join(lines))
        return str(path)

    async def continue_research(
        self, slug: str, focus: str | None = None,
        model_override: str | None = None,
    ) -> AsyncIterator[ResearchEvent]:
        """Continue a prior investigation by slug, running additional rounds to fill gaps."""
        start = time.monotonic()
        model = model_override or self.ollama.model

        yield ResearchEvent(ResearchEventType.STATUS, {"message": f"Loading prior investigation '{slug}'..."})

        prior = await self.cache.get_research_by_slug(slug)
        if not prior:
            yield ResearchEvent(ResearchEventType.DONE, {
                "error": f"No investigation found with slug '{slug}'",
                "rounds_used": 0, "total_findings": 0, "total_sources": 0,
                "recalled_findings": 0, "elapsed_ms": 0, "slug": slug,
                "verification_status": "fail",
                "provenance": {},
            })
            return

        prior_findings_raw = await self.cache.get_findings_for_research(prior["id"])
        prior_findings = [
            Finding(
                sub_query=f["sub_query"], summary=f["summary"],
                sources=f["sources"], round_num=f["round_num"],
            )
            for f in prior_findings_raw
        ]

        yield ResearchEvent(ResearchEventType.STATUS, {
            "message": f"Loaded {len(prior_findings)} prior findings. Planning continuation...",
        })

        original_query = prior["query"]
        effective_max_rounds = self.config.max_rounds
        effective_max_sub = self.config.max_sub_queries
        effective_scrape = self.config.deep_scrape

        # Decompose with awareness of prior findings
        prior_findings_text = "\n\n".join(
            f"### {f.sub_query}\n{f.summary}" for f in prior_findings
        )
        focus_section = f"Focus area for continuation: {focus}" if focus else ""
        system = _CONTINUE_DECOMPOSE_SYSTEM.format(max_sub_queries=effective_max_sub)
        user = _CONTINUE_DECOMPOSE_USER.format(
            query=original_query, focus_section=focus_section,
            prior_findings_text=prior_findings_text,
        )

        try:
            response = await self.ollama.generate(system, user, json_mode=True, model=model)
            data = json.loads(response)
            sub_queries = data.get("sub_queries", [])[:effective_max_sub]
            if not sub_queries:
                sub_queries = [f"{original_query} latest developments", f"{original_query} in depth"]
        except (json.JSONDecodeError, KeyError):
            sub_queries = [f"{original_query} latest developments", f"{original_query} in depth"]

        yield ResearchEvent(ResearchEventType.PLAN, {"sub_queries": sub_queries, "slug": slug})

        all_findings: list[Finding] = list(prior_findings)
        all_sources: list[dict] = []
        source_counter = prior.get("total_sources", 0)
        new_findings: list[Finding] = []
        seen_source_urls = {
            source.get("url")
            for finding in prior_findings_raw
            for source in finding.get("sources", [])
            if source.get("url")
        }

        round_num = 0
        for round_num in range(1, effective_max_rounds + 1):
            yield ResearchEvent(ResearchEventType.ROUND_START, {
                "round": round_num,
                "max_rounds": effective_max_rounds,
                "num_queries": len(sub_queries),
            })

            round_findings, round_sources, source_counter = await self._search_round(
                sub_queries, round_num, source_counter, model,
                scrape=effective_scrape, seen_source_urls=seen_source_urls,
            )

            for finding in round_findings:
                yield ResearchEvent(ResearchEventType.FINDING, {
                    "sub_query": finding.sub_query,
                    "summary_preview": finding.summary[:200],
                    "num_sources": len(finding.sources),
                    "round": round_num,
                    "summary_failed": finding.error is not None,
                    "error": finding.error,
                })

            all_findings.extend(round_findings)
            new_findings.extend(round_findings)
            all_sources.extend(round_sources)

            if round_num >= effective_max_rounds:
                break

            gap_result = await self._analyze_gaps(
                original_query, all_findings, model, max_follow_ups=effective_max_sub,
            )
            yield ResearchEvent(ResearchEventType.GAP_ANALYSIS, gap_result)

            if gap_result.get("sufficient", True):
                break
            sub_queries = gap_result.get("gaps", [])
            if not sub_queries:
                break

        if not round_num:
            round_num = 1

        yield ResearchEvent(ResearchEventType.STATUS, {"message": "Synthesizing updated report..."})
        report_parts = []
        async for token in self._synthesize_report(original_query, all_findings, all_sources, [], model):
            report_parts.append(token)
            yield ResearchEvent(ResearchEventType.TOKEN, {"content": token})

        report_text = self._normalize_citation_markers("".join(report_parts))

        yield ResearchEvent(ResearchEventType.STATUS, {"message": "Verifying research output..."})
        report_text, verification, verify_events = await self._verify_and_repair_report(
            original_query, report_text, all_findings, all_sources, model,
        )
        for verify_event in verify_events:
            yield ResearchEvent(ResearchEventType.VERIFY, verify_event)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        provenance = ProvenanceRecord(
            topic=original_query, slug=slug, rounds=prior.get("rounds_used", 0) + round_num,
            sources_consulted=len(all_sources) + prior.get("total_sources", 0),
            sources_accepted=len(all_sources) + prior.get("total_sources", 0),
            sources_rejected=0,
            verification_status=verification.status,
            verification_summary=verification.summary,
            model_used=model, elapsed_ms=elapsed_ms,
        )
        corpus_path = self._write_corpus(
            slug=slug,
            query=original_query,
            report=report_text,
            findings=all_findings,
            sources=all_sources,
            provenance=provenance,
            kind="continuation",
            directive=focus,
        )
        provenance.research_files = [corpus_path]

        try:
            all_sub_queries = list({f.sub_query for f in new_findings})
            research_id = await self.cache.store_research(
                query=original_query, report=report_text, sub_queries=all_sub_queries,
                rounds_used=round_num, total_sources=len(all_sources),
                model_used=model, elapsed_ms=elapsed_ms, slug=slug,
                parent_id=prior["id"],
                verification_status=verification.status,
                verification_summary=verification.summary,
                provenance=provenance.to_markdown(),
            )
            if research_id and new_findings:
                await self.cache.store_findings_batch(
                    research_id=research_id,
                    findings=[
                        {
                            "sub_query": f.sub_query, "summary": f.summary,
                            "sources": f.sources, "round_num": f.round_num,
                        }
                        for f in new_findings
                    ],
                )
        except Exception as e:
            logger.warning(f"Failed to store continuation results: {e}")

        yield ResearchEvent(ResearchEventType.DONE, {
            "rounds_used": round_num,
            "total_findings": len(new_findings),
            "total_sources": len(all_sources),
            "prior_findings_loaded": len(prior_findings),
            "continued_from": slug,
            "recalled_findings": 0,
            "elapsed_ms": elapsed_ms,
            "slug": slug,
            "verification_status": verification.status,
            "provenance": provenance.to_dict(),
            "corpus_path": corpus_path,
            "failed_findings": sum(1 for f in new_findings if f.error),
        })
