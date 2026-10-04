"""Deterministic synthetic trace. No SDK, API key, or network required."""
from __future__ import annotations

from .schema import Span, Trace, new_trace_id
from .store import Store


def record_demo(store: Store) -> str:
    trace_id = new_trace_id()
    store.insert_trace(Trace(trace_id, "synthetic-demo", start_ns=0, end_ns=4_000_000_000))
    store.insert_span(Span(f"{trace_id}-root", trace_id, "synthetic-agent", kind="agent",
                        start_ns=0, end_ns=4_000_000_000))
    for index, page in enumerate((1, 1, 2)):
        store.insert_span(Span(
            f"{trace_id}-{index}", trace_id, "summarize-page", kind="llm",
            parent_id=f"{trace_id}-root", start_ns=index * 1_000_000_000,
            end_ns=(index + 1) * 1_000_000_000,
            attributes={
                "gen_ai.system": "synthetic", "gen_ai.request.model": "demo-model",
                "gen_ai.usage.input_tokens": 2400, "gen_ai.usage.output_tokens": 160,
                "cached_input_tokens": 0, "cache_write_tokens": 0,
                "cost_usd": 0.012, "request_hash": f"synthetic-page-{page}",
                "request": {"messages": [{"role": "user", "content": f"Summarize synthetic page {page}."}]},
                "response": {"text": f"Synthetic summary {page}."},
            },
        ))
    return trace_id
