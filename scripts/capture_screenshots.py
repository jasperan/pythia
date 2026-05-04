"""Capture TUI screenshots for README documentation."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pythia.config import PythiaConfig
from pythia.tui.app import PythiaApp


SCREENSHOTS_DIR = Path(__file__).parent.parent / "docs" / "screenshots"


async def capture_search_screen(app, pilot):
    """Capture the search screen with demo results."""
    from pythia.tui.widgets.session_divider import SessionDivider
    from pythia.tui.widgets.result_card import ResultCard
    from pythia.tui.widgets.source_list import SourceList
    from pythia.tui.widgets.cache_badge import CacheBadge
    from pythia.tui.widgets.status_bar import PythiaStatusBar
    from textual.containers import VerticalScroll

    # Update status bar to show healthy services
    try:
        status = app.screen.query_one(PythiaStatusBar)
        status.update_status(model="qwen3.5:9b", oracle_ok=True, searxng_ok=True, cache_size=42)
    except Exception:
        pass

    results = app.screen.query_one("#results-area", VerticalScroll)

    # First query - cache hit
    div1 = SessionDivider(query="What is RLHF and why does it matter?", timestamp="3:15 PM")
    card1 = ResultCard()
    sources1 = SourceList()
    badge1 = CacheBadge()
    await results.mount(div1)
    await results.mount(card1)
    await results.mount(sources1)
    await results.mount(badge1)
    card1.set_content(
        "**Reinforcement Learning from Human Feedback (RLHF)** is a technique for fine-tuning "
        "language models using human preference data rather than traditional supervised labels.\n\n"
        "The core idea: instead of telling the model what the \"right\" answer is, you show it pairs "
        "of outputs and let humans pick which one is better. A reward model learns these preferences, "
        "then the LLM is optimized via PPO to maximize that reward signal [1][2].\n\n"
        "**Why it matters:** RLHF bridges the gap between \"technically correct\" and \"actually helpful.\" "
        "Models trained with RLHF are better at following instructions, refusing harmful requests, "
        "and producing responses that humans rate as more useful [3]."
    )
    sources1.add_source({"index": 1, "title": "Training language models to follow instructions with human feedback", "url": "https://arxiv.org/abs/2203.02155"})
    sources1.add_source({"index": 2, "title": "Learning to summarize from human feedback", "url": "https://arxiv.org/abs/2009.01325"})
    sources1.add_source({"index": 3, "title": "Anthropic RLHF Research", "url": "https://www.anthropic.com/research"})
    badge1.show_cache_hit(0.92, 18)

    # Second query - web search
    div2 = SessionDivider(query="How does Oracle AI Vector Search compare to pgvector?", timestamp="3:18 PM")
    card2 = ResultCard()
    sources2 = SourceList()
    badge2 = CacheBadge()
    await results.mount(div2)
    await results.mount(card2)
    await results.mount(sources2)
    await results.mount(badge2)
    card2.set_content(
        "Oracle AI Vector Search and pgvector take fundamentally different approaches to the same "
        "problem.\n\n"
        "**Oracle AI Vector Search** runs embeddings *inside the database* via ONNX models loaded "
        "directly into Oracle 23ai/26ai. This means your vectors never leave the DB, and you get "
        "ACID transactions covering both your relational data and vector operations. It supports "
        "IVF, HNSW, and hybrid search natively [1].\n\n"
        "**pgvector** is a PostgreSQL extension that adds vector columns and approximate nearest "
        "neighbor search. It's simpler to set up, has strong community adoption, and integrates "
        "well with the Postgres ecosystem. HNSW indexing was added in 0.5.0 [2].\n\n"
        "The key differentiator: Oracle's in-database ONNX embedding means you can call "
        "`VECTOR_EMBEDDING(model USING 'text')` in SQL. With pgvector, you generate embeddings "
        "externally and store them. For high-security environments where data can't leave the DB, "
        "Oracle wins [3]."
    )
    sources2.add_source({"index": 1, "title": "Oracle AI Vector Search Documentation", "url": "https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/"})
    sources2.add_source({"index": 2, "title": "pgvector: Open-source vector similarity search for Postgres", "url": "https://github.com/pgvector/pgvector"})
    sources2.add_source({"index": 3, "title": "Oracle 23ai Free: In-Database ML and Vector Search", "url": "https://blogs.oracle.com/database/post/oracle-23ai-vector-search"})
    badge2.show_web_search(2340, 8)

    results.scroll_end()
    await pilot.pause(delay=0.5)
    svg = app.export_screenshot()
    (SCREENSHOTS_DIR / "search.svg").write_text(svg)
    print("  Captured: search.svg")


async def capture_research_screen(app, pilot):
    """Capture the research screen with demo tree and report."""
    from pythia.tui.widgets.research_tree import ResearchTree
    from pythia.tui.widgets.research_progress import ResearchProgressBar
    from pythia.tui.widgets.result_card import ResultCard

    app.set_focus(None)
    await pilot.press("2")
    await pilot.pause(delay=0.3)

    tree = app.screen.query_one(ResearchTree)
    progress = app.screen.query_one(ResearchProgressBar)
    result_card = app.screen.query_one(ResultCard)

    # Simulate recalled findings
    tree.set_recall([
        {"sub_query": "RISC-V power efficiency benchmarks", "similarity": 0.82, "from_query": "ARM vs x86 power consumption"},
    ])

    # Round 1 - completed
    tree.add_plan([
        "What are the main RISC-V implementations for edge AI?",
        "How does ARM Cortex-M compare for ML inference?",
        "What are the power consumption tradeoffs?",
    ])
    tree.start_round(1, 3)
    tree.complete_finding("What are the main RISC-V implementations for edge AI?", num_sources=6, preview="SiFive, Kendryte K210...")
    tree.complete_finding("How does ARM Cortex-M compare for ML inference?", num_sources=4, preview="Cortex-M55 with Ethos-U55...")
    tree.complete_finding("What are the power consumption tradeoffs?", num_sources=5, preview="RISC-V custom extensions...")

    # Round 2 - in progress
    tree.add_gaps(["What about RISC-V vector extensions for neural network acceleration?", "Cost comparison: RISC-V vs ARM licensing"], reasoning="Missing details on hardware acceleration and economics")
    tree.start_round(2, 3)
    tree.complete_finding("What about RISC-V vector extensions for neural network acceleration?", num_sources=3, preview="RVV 1.0 specification...")

    progress.update_progress(round_num=2, max_rounds=3, findings=4, sources=18, elapsed_ms=12400)

    # Stream partial report
    result_card.set_content(
        "## Executive Summary\n\n"
        "The RISC-V vs ARM debate for edge AI comes down to a fundamental tradeoff between "
        "**flexibility and ecosystem maturity**. RISC-V offers custom instruction extensions "
        "purpose-built for neural network inference, while ARM provides a battle-tested toolchain "
        "and silicon ecosystem [1][2].\n\n"
        "## Hardware Implementations\n\n"
        "### RISC-V for Edge AI\n"
        "The leading RISC-V implementations for edge AI include SiFive's X280 with vector extensions, "
        "Kendryte's K210 dual-core processor, and ESP32-C3 for ultra-low-power applications. "
        "The key advantage is the ability to add custom instructions for specific neural network "
        "operations without licensing fees [1][3].\n\n"
        "### ARM Cortex-M Series\n"
        "ARM's Cortex-M55 paired with the Ethos-U55 NPU represents the current state of the art "
        "for embedded ML inference. The CMSIS-NN library provides optimized kernels, and the "
        "toolchain maturity means faster time-to-market [2][4]..."
    )

    await pilot.pause(delay=0.5)
    svg = app.export_screenshot()
    (SCREENSHOTS_DIR / "research.svg").write_text(svg)
    print("  Captured: research.svg")


async def capture_history_screen(app, pilot):
    """Capture the history screen with demo entries."""
    from pythia.tui.widgets.history_list import HistoryList, HistoryEntry
    from rich.text import Text
    from textual.widgets import Static

    app.set_focus(None)
    await pilot.press("3")
    await pilot.pause(delay=0.3)

    hl = app.screen.query_one(HistoryList)
    entries = [
        HistoryEntry(query="How does Oracle AI Vector Search compare to pgvector?", cache_hit=False, response_time_ms=2340, model="qwen3.5:9b", is_research=False, timestamp="3:18 PM"),
        HistoryEntry(query="What is RLHF and why does it matter?", cache_hit=True, response_time_ms=18, model="qwen3.5:9b", is_research=False, timestamp="3:15 PM"),
        HistoryEntry(query="RISC-V vs ARM for edge AI inference", cache_hit=False, response_time_ms=14200, model="qwen3.5:9b", is_research=True, timestamp="3:08 PM"),
        HistoryEntry(query="Python asyncio vs threading for IO-bound work", cache_hit=False, response_time_ms=1820, model="qwen3.5:9b", is_research=False, timestamp="2:55 PM"),
        HistoryEntry(query="What is semantic caching and how does it work?", cache_hit=True, response_time_ms=12, model="qwen3.5:9b", is_research=False, timestamp="2:41 PM"),
        HistoryEntry(query="Textual framework best practices for complex TUIs", cache_hit=False, response_time_ms=3100, model="qwen3.5:9b", is_research=False, timestamp="2:30 PM"),
        HistoryEntry(query="State of Rust async ecosystem 2025", cache_hit=False, response_time_ms=18900, model="qwen3.5:9b", is_research=True, timestamp="1:45 PM"),
        HistoryEntry(query="Oracle 26ai Free Docker setup guide", cache_hit=True, response_time_ms=9, model="qwen3.5:9b", is_research=False, timestamp="1:20 PM"),
    ]
    hl.load_entries(entries)

    # Update footer
    footer = Text()
    footer.append(f"  {len(entries)} queries", style="#e0e0e0")
    footer.append(" \u00b7 3 cache hits (38%)", style="#b5bd68")
    footer.append(" \u00b7 avg 5.1s", style="#666666")
    footer.append("\n  \u2191\u2193/jk Navigate  Enter Re-run  r Research  / Filter", style="#808080")
    app.screen.query_one("#history-footer", Static).update(footer)

    await pilot.pause(delay=0.5)
    svg = app.export_screenshot()
    (SCREENSHOTS_DIR / "history.svg").write_text(svg)
    print("  Captured: history.svg")


async def capture_dashboard_screen(app, pilot):
    """Capture the dashboard screen with demo stats."""
    from pythia.tui.widgets.stats_panel import StatsPanel
    from pythia.tui.widgets.sparkline_panel import SparklinePanel

    app.set_focus(None)
    await pilot.press("4")
    await pilot.pause(delay=0.3)

    # Stop dashboard refresh timer
    screen = app.screen
    if hasattr(screen, '_refresh_interval') and screen._refresh_interval:
        screen._refresh_interval.stop()
        screen._refresh_interval = None

    stats = app.screen.query_one(StatsPanel)
    stats.update_stats({
        "total_searches": 142,
        "cache_hits": 61,
        "cache_hit_rate": 42.9,
        "cache_size": 38,
        "avg_response_ms": 1240,
        "active_days": 12,
    })

    sparkline = app.screen.query_one(SparklinePanel)
    sparkline.update_data([
        {"response_time_ms": 18, "cache_hit": True},
        {"response_time_ms": 2340, "cache_hit": False},
        {"response_time_ms": 1820, "cache_hit": False},
        {"response_time_ms": 12, "cache_hit": True},
        {"response_time_ms": 3100, "cache_hit": False},
        {"response_time_ms": 14200, "cache_hit": False},
        {"response_time_ms": 9, "cache_hit": True},
        {"response_time_ms": 890, "cache_hit": False},
        {"response_time_ms": 15, "cache_hit": True},
        {"response_time_ms": 4500, "cache_hit": False},
        {"response_time_ms": 22, "cache_hit": True},
        {"response_time_ms": 1100, "cache_hit": False},
        {"response_time_ms": 2800, "cache_hit": False},
        {"response_time_ms": 11, "cache_hit": True},
        {"response_time_ms": 950, "cache_hit": False},
        {"response_time_ms": 3400, "cache_hit": False},
        {"response_time_ms": 16, "cache_hit": True},
        {"response_time_ms": 2100, "cache_hit": False},
        {"response_time_ms": 13, "cache_hit": True},
        {"response_time_ms": 1600, "cache_hit": False},
    ])

    await pilot.pause(delay=0.5)
    svg = app.export_screenshot()
    (SCREENSHOTS_DIR / "dashboard.svg").write_text(svg)
    print("  Captured: dashboard.svg")


async def main():
    print("Capturing TUI screenshots...")
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    config = PythiaConfig()
    app = PythiaApp(config, auto_start=False)

    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.pause(delay=0.3)

        # Stop health check timer to prevent network errors
        screen = app.screen
        if hasattr(screen, '_health_check_interval') and screen._health_check_interval:
            screen._health_check_interval.stop()
            screen._health_check_interval = None

        await capture_search_screen(app, pilot)
        await capture_research_screen(app, pilot)
        await capture_history_screen(app, pilot)
        await capture_dashboard_screen(app, pilot)

    print(f"\nAll screenshots saved to {SCREENSHOTS_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
