"""Shared context and helpers for waste rules.

Every dollar figure a rule emits must be derivable from the observed token
counts and the pricing table; when either is missing the rule reports the
token or time quantity alone rather than inventing a number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ...instrument._util import estimate_tokens
from ...pricing import PricingTable
from ...schema import Span


@dataclass
class RuleContext:
    trace_id: str
    spans: list[Span]
    pricing: PricingTable
    llm_spans: list[Span] = field(init=False)

    def __post_init__(self):
        self.llm_spans = [s for s in self.spans if s.kind == "llm"]

    def ok_llm_spans(self) -> list[Span]:
        return [s for s in self.llm_spans if s.status == "ok"]


def provider(span: Span) -> str | None:
    return span.attributes.get("gen_ai.system")


def model(span: Span) -> str | None:
    return span.attributes.get("gen_ai.response.model") or span.attributes.get(
        "gen_ai.request.model"
    )


def cost(span: Span) -> float:
    value = span.attributes.get("cost_usd")
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def prompt_tokens(span: Span) -> int:
    attrs = span.attributes
    return int(
        (attrs.get("gen_ai.usage.input_tokens") or 0)
        + (attrs.get("cached_input_tokens") or 0)
        + (attrs.get("cache_write_tokens") or 0)
    )


def output_tokens(span: Span) -> int:
    return int(span.attributes.get("gen_ai.usage.output_tokens") or 0)


def duration_seconds(span: Span) -> float:
    if span.end_ns is None:
        return 0.0
    return max(0, span.end_ns - span.start_ns) / 1e9


def _block_texts(block: Any, out: list[str], depth: int = 0) -> None:
    """Pull text out of one content block, including tool traffic.

    Tool results nest their payload under ``content`` (a string or more
    blocks) and tool_use blocks carry their arguments in ``input``; both
    must count as message text, otherwise consecutive iterations of a tool
    loop look identical to the similarity rules.
    """
    if depth > 6:
        return
    if isinstance(block, str):
        if block:
            out.append(block)
        return
    if not isinstance(block, dict):
        return
    text = block.get("text")
    if isinstance(text, str) and text:
        out.append(text)
    if block.get("type") == "tool_use" and block.get("input") is not None:
        out.append(json.dumps(block["input"], sort_keys=True, default=str))
    nested = block.get("content")
    if isinstance(nested, str) and nested:
        out.append(nested)
    elif isinstance(nested, list):
        for sub in nested:
            _block_texts(sub, out, depth + 1)


def message_texts(request: Any) -> list[str]:
    """Individual message/content texts from a normalized request payload.

    Handles the three shapes the patchers record: chat messages, Responses
    input (string or item list, plus instructions), and anthropic messages
    with an optional system string. Tool results and tool_use arguments
    count as text.
    """
    if not isinstance(request, dict):
        return []
    texts: list[str] = []
    system = request.get("system") or request.get("instructions")
    if isinstance(system, str) and system:
        texts.append(system)
    elif isinstance(system, list):
        # anthropic system prompts are a block list whenever cache_control
        # is involved; the text inside is still prefix content.
        for block in system:
            _block_texts(block, texts)
    raw_input = request.get("input")
    if isinstance(raw_input, str) and raw_input:
        texts.append(raw_input)
    items = request.get("messages")
    if items is None and isinstance(raw_input, list):
        items = raw_input
    for item in items or []:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str) and content:
            texts.append(content)
        elif isinstance(content, list):
            for block in content:
                _block_texts(block, texts)
        if item.get("type") == "tool_use" and item.get("input") is not None:
            texts.append(json.dumps(item["input"], sort_keys=True, default=str))
    return texts


def message_count(request: Any) -> int:
    if not isinstance(request, dict):
        return 0
    items = request.get("messages")
    if items is None and isinstance(request.get("input"), list):
        items = request["input"]
    return len(items or [])


def request_text(request: Any) -> str:
    return "\n".join(message_texts(request))


def estimate(text: str) -> int:
    return estimate_tokens(text)


def span_when(span: Span) -> str:
    return f"span {span.span_id[:10]} ({span.name})"
