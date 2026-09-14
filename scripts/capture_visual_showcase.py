"""Capture the real terminal UI with isolated illustrative research data."""
import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from pythia.config import PythiaConfig
from pythia.tui.app import PythiaApp
from pythia.tui.widgets.result_card import ResultCard
from pythia.tui.widgets.session_divider import SessionDivider
from pythia.tui.widgets.source_list import SourceList


async def capture():
    output = Path(__file__).resolve().parents[1] / "docs/screenshots/research-desk.svg"
    app = PythiaApp(PythiaConfig(), auto_start=False)
    response = httpx.Response(200, json={"oracle": False, "searxng": False})
    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=response)):
        async with app.run_test(size=(120, 38)) as pilot:
            await pilot.pause()
            area = app.screen.query_one("#results-area")
            await area.mount(SessionDivider(query="How do ideas become evidence?", timestamp="Example"))
            card = ResultCard()
            await area.mount(card)
            card.set_content(
                "## Follow the question, then follow the evidence.\n\n"
                "A useful investigation starts with a precise question. Search for competing "
                "explanations, keep track of your sources, and make the gaps visible.\n\n"
                "### 01  Collect\n"
                "Gather relevant sources and preserve the context around each claim.\n\n"
                "### 02  Connect\n"
                "Compare findings across sources. Separate agreement from independent verification.\n\n"
                "### 03  Revisit\n"
                "Keep the investigation open to new evidence. A clear uncertainty is more useful "
                "than an unsupported conclusion.\n\n"
                "*Illustrative research content for the interface showcase. No live query was run.*"
            )
            sources = SourceList()
            await area.mount(sources)
            sources.add_source({"index": 1, "title": "Pythia · project documentation", "url": "https://github.com/jasperan/pythia"})
            await pilot.pause()
            output.write_text(app.export_screenshot(), encoding="utf-8")
            assert app.screen.query_one("#results-area").size.width > 80
            await pilot.press("ctrl+t")
            await pilot.pause()
            assert app.theme == "light"
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            assert app.screen.query_one("#results-area").size.width > 50
    print(f"Captured {output}; verified theme switching and 80-column layout")


if __name__ == "__main__":
    os.environ.pop("NO_COLOR", None)
    os.environ["COLORTERM"] = "truecolor"
    os.environ["FORCE_COLOR"] = "1"
    asyncio.run(capture())
