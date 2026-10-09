"""
Unit tests for SuppQATrain's web tools (no API key or network needed).

The env-level test needs train.parquet next to this file, since the module
loads it at import.

Run: pytest tests.py -v
"""

import asyncio
import json
from pathlib import Path

import pytest

HAS_DATA = (Path(__file__).parent / "train.parquet").exists()


# ---------------------------------------------------------------------------
# web_fetch output budget
# ---------------------------------------------------------------------------
# Harnesses cap a tool result's serialized size (text blocks + metadata JSON).
# A fetch of a long page must stay under that cap and still show the page head.

OUTPUT_CAP_BYTES = 32 * 1024
FETCH_URL = "https://example.com/quarterly-report"
LONG_PAGE = "".join(
    f'Item {i}. Net revenue for "segment {i}" rose 4.2% year over year.\n' for i in range(2500)
)


def _serialized_bytes(out) -> int:
    payload = {
        "blocks": [b.model_dump() for b in out.blocks],
        "metadata": out.metadata,
        "reward": out.reward,
        "finished": out.finished,
    }
    return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def _fetch_through(monkeypatch, toolset_cls, page: str):
    """Call ``toolset_cls.web_fetch`` with the backend replaced by one that
    returns ``page`` exactly as the SDK's backends render a fetch."""
    from openreward.tools.web import (
        MAX_FETCH_TEXT_CHARS,
        WebToolResult,
        build_fetch_output,
        render_fetch_text,
    )
    from openreward.toolsets import WebToolset
    from openreward.toolsets._web_common import WebFetchParams

    async def fake_run_web_fetch(*, url, prompt, max_chars=None, **_):
        limit = max_chars or MAX_FETCH_TEXT_CHARS
        text = page if len(page) <= limit else page[:limit] + "\n... (truncated)"
        return WebToolResult.success(
            render_fetch_text(url, text, "example.com", (), limit),
            build_fetch_output(url=url, text=text, prompt=prompt),
        )

    async def no_backend(self):
        return None

    monkeypatch.setattr("openreward.toolsets.web.run_web_fetch", fake_run_web_fetch)
    monkeypatch.setattr(WebToolset, "_get_impl", no_backend)
    params = WebFetchParams(url=FETCH_URL, prompt="segment revenue")
    return asyncio.run(toolset_cls(None).web_fetch(params))


def _assert_long_page_capped(out) -> None:
    text = out.blocks[0].text
    assert _serialized_bytes(out) <= OUTPUT_CAP_BYTES
    assert LONG_PAGE[:5000] in text
    assert "Page truncated" in text


def test_web_fetch_long_page_fits_output_cap(monkeypatch):
    from web_fetch_cap import CappedWebToolset

    _assert_long_page_capped(_fetch_through(monkeypatch, CappedWebToolset, LONG_PAGE))


def test_web_fetch_short_page_unchanged(monkeypatch):
    from web_fetch_cap import CappedWebToolset

    page = LONG_PAGE[:3000]
    out = _fetch_through(monkeypatch, CappedWebToolset, page)
    assert out.blocks[0].text == f"Fetched content from {FETCH_URL}:\n\n{page}"
    assert out.metadata == {"url": FETCH_URL, "prompt": "segment revenue"}


@pytest.mark.skipif(not HAS_DATA, reason="requires train.parquet")
def test_env_web_fetch_fits_output_cap(monkeypatch):
    from openreward.toolsets import WebToolset

    from suppqatrain import SuppQATrain

    toolset_cls = next(t for t in SuppQATrain.toolsets if issubclass(t, WebToolset))
    _assert_long_page_capped(_fetch_through(monkeypatch, toolset_cls, LONG_PAGE))
