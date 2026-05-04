#!/usr/bin/env python3
"""Smoke test for the OCI GenAI (Grok 4) backend.

Skips cleanly if the OCI compartment is unset or the proxy is not running.
"""
from __future__ import annotations

import asyncio
import os
import sys

import httpx

from pythia.server.oci_genai import OciGenAIClient


PROXY_URL = os.environ.get("PYTHIA_OCI_PROXY_URL", "http://localhost:9999/v1")


async def main() -> int:
    if not os.environ.get("OCI_COMPARTMENT_ID"):
        print("SKIP: OCI_COMPARTMENT_ID is not set. Configure OCI env and rerun.")
        return 0

    try:
        async with httpx.AsyncClient(timeout=3.0) as http:
            resp = await http.get(f"{PROXY_URL}/health")
            if resp.status_code != 200:
                print(
                    f"SKIP: proxy responded {resp.status_code} at {PROXY_URL}/health. "
                    "Run `cd oci-genai && python proxy.py`."
                )
                return 0
    except Exception as exc:
        print(
            f"SKIP: proxy not running at {PROXY_URL} ({exc}). "
            "Run `cd oci-genai && python proxy.py`."
        )
        return 0

    client = OciGenAIClient(base_url=PROXY_URL, model="xai.grok-4")
    try:
        reply = await client.generate(
            "You are a helpful assistant. Answer with exactly one word.",
            "Say hello.",
        )
    finally:
        await client.close()

    print(f"Grok 4 reply: {reply!r}")
    if not reply or not reply.strip():
        print("FAIL: empty response from Grok 4.")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
