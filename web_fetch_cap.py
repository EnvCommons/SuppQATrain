"""Keep ``web_fetch`` results inside a tool-output size budget.

The SDK's fetch result carries the page text twice: once in the text block and
again in ``metadata["content"]``. Harnesses that cap a tool result's serialized
size (text blocks plus metadata JSON, commonly 32 KiB) can only trim text
blocks, so a page whose metadata copy alone exceeds the cap is replaced whole
and the agent sees no content at all. Nothing reads the copy.

``cap_fetch_output`` drops the copy and keeps the head of the page text, up to
``FETCH_TEXT_MAX_BYTES`` UTF-8 bytes, followed by a note saying how much was
cut. That leaves room under 32 KiB for the URL, the prompt echo and JSON
escaping.
"""

from __future__ import annotations

from openreward.environments import TextBlock, ToolOutput, tool
from openreward.toolsets import WebToolset
from openreward.toolsets._web_common import WebFetchParams

FETCH_TEXT_MAX_BYTES = 24_000

# The SDK appends this when it has already cut a page at its own limit.
_SDK_TRUNCATION_MARK = "\n... (truncated)"


def _cap_text(text: str, max_bytes: int) -> str:
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    head = raw[:max_bytes].decode("utf-8", errors="ignore")
    if text.endswith(_SDK_TRUNCATION_MARK):
        total = f"more than {len(text) - len(_SDK_TRUNCATION_MARK):,}"
    else:
        total = f"{len(text):,}"
    return (
        f"{head}\n\n[Page truncated: showing the first {len(head):,} of {total} characters. "
        "web_fetch returns only the start of a long page. To read a later part, fetch a more "
        "specific URL (a single section, page or document), or use web_search to find a source "
        "that states the fact directly.]"
    )


def cap_fetch_output(out: ToolOutput, max_bytes: int = FETCH_TEXT_MAX_BYTES) -> ToolOutput:
    """``out`` without the ``metadata["content"]`` copy and with its text capped."""
    metadata = {k: v for k, v in (out.metadata or {}).items() if k != "content"}
    blocks = [
        TextBlock(text=_cap_text(b.text, max_bytes)) if isinstance(b, TextBlock) else b
        for b in out.blocks
    ]
    return out.model_copy(update={"blocks": blocks, "metadata": metadata or None})


class CappedWebToolset(WebToolset):
    """``WebToolset`` whose ``web_fetch`` result fits the tool-output budget."""

    @tool(concurrent=True)
    async def web_fetch(self, params: WebFetchParams) -> ToolOutput:
        return cap_fetch_output(await WebToolset.web_fetch(self, params))


CappedWebToolset.web_fetch.__doc__ = WebToolset.web_fetch.__doc__
