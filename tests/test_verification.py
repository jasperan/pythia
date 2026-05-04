"""Tests for research report verification prompt construction."""
from __future__ import annotations

import json

import pytest

from pythia.verification import verify_report


class CapturingLLM:
    model = "test-model"

    def __init__(self):
        self.user_prompt = ""

    async def generate(
        self,
        system: str,
        user: str,
        json_mode: bool = False,
        model: str | None = None,
    ) -> str:
        self.user_prompt = user
        return json.dumps({"claims_checked": 1, "issues": [], "status": "pass", "summary": "ok"})


@pytest.mark.asyncio
async def test_verify_report_includes_source_excerpts_for_cited_sources():
    llm = CapturingLLM()
    report = "Python 3.13 added an experimental JIT compiler [38]."
    sources = [
        {
            "index": 1,
            "title": "Uncited",
            "url": "https://example.com/uncited",
            "snippet": "This should not be the only source context.",
        },
        {
            "index": 38,
            "title": "Python 3.13 whatsnew",
            "url": "https://docs.python.org/3/whatsnew/3.13.html",
            "snippet": "Python 3.13 includes an experimental just-in-time compiler.",
        },
    ]

    result = await verify_report(llm, "Python 3.13", report, sources, "test-model")

    assert result.status == "pass"
    assert "[38] Python 3.13 whatsnew" in llm.user_prompt
    assert "Source excerpt: Python 3.13 includes an experimental just-in-time compiler." in llm.user_prompt


@pytest.mark.asyncio
async def test_verify_report_normalizes_unknown_status_to_fail():
    class BadStatusLLM:
        model = "test-model"

        async def generate(
            self,
            system: str,
            user: str,
            json_mode: bool = False,
            model: str | None = None,
        ) -> str:
            return json.dumps({
                "claims_checked": 3,
                "issues": [],
                "status": "all_claims_unsupported",
                "summary": "Could not support the report.",
            })

    result = await verify_report(
        BadStatusLLM(),
        "Python 3.13",
        "Python 3.13 added a JIT compiler [1].",
        [{"index": 1, "title": "Source", "url": "https://example.com", "snippet": "snippet"}],
        "test-model",
    )

    assert result.status == "fail"
    assert "unsupported status" in result.summary


@pytest.mark.asyncio
async def test_verify_report_locally_verifies_evidence_ledger():
    class ShouldNotBeCalledLLM:
        model = "test-model"

        async def generate(self, *args, **kwargs) -> str:
            raise AssertionError("Evidence ledger verification should not call the LLM")

    result = await verify_report(
        ShouldNotBeCalledLLM(),
        "Python 3.13",
        "# Evidence Ledger Report\n\n- [1] Source title: source excerpt",
        [{"index": 1, "title": "Source", "url": "https://example.com", "snippet": "source excerpt"}],
        "test-model",
    )

    assert result.status == "pass_with_notes"
    assert result.claims_checked == 1
    assert "locally verified" in result.summary
