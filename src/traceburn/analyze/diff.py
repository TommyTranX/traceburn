"""Diff two recorded traces: structure, metrics, and prompt/response text.

Spans are aligned recursively: children of matched parents pair up by
(name, kind) in order of occurrence, so the second "chat gpt-5" under the
same parent in run A matches the second in run B. Unmatched spans are
reported as added or removed. Matched LLM spans additionally get a unified
text diff of their prompts and responses when they differ.

The result dict is a public JSON interface.
"""

from __future__ import annotations

import difflib
import json
import math
from typing import Any

from ..schema import Span
from ..store import Store


def _tree(spans: list[Span]) -> dict[str | None, list[Span]]:
    ids = {s.span_id for s in spans}
    children: dict[str | None, list[Span]] = {}
    for span in spans:
        parent = span.parent_id if span.parent_id in ids else None
        children.setdefault(parent, []).append(span)
    for kids in children.values():
        kids.sort(key=lambda s: s.start_ns)
    return children


def _metric(span: Span, key: str) -> float:
    value = span.attributes.get(key)
    try:
        parsed = float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _duration_ms(span: Span) -> float:
    if span.end_ns is None:
        return 0.0
    return (span.end_ns - span.start_ns) / 1e6


def _as_text(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, indent=2, sort_keys=True, default=str)


def _text_diff(label: str, a: Any, b: Any, context: int = 2) -> list[str] | None:
    text_a, text_b = _as_text(a), _as_text(b)
    if text_a == text_b:
        return None
    lines = list(
        difflib.unified_diff(
            text_a.splitlines(),
            text_b.splitlines(),
            fromfile=f"a/{label}",
            tofile=f"b/{label}",
            lineterm="",
            n=context,
        )
    )
    return lines or None


def _span_brief(span: Span) -> dict[str, Any]:
    return {
        "span_id": span.span_id,
        "name": span.name,
        "kind": span.kind,
        "status": span.status,
        "duration_ms": round(_duration_ms(span), 3),
        "cost_usd": span.attributes.get("cost_usd"),
    }


def _match_level(
    a_spans: list[Span],
    b_spans: list[Span],
    tree_a: dict,
    tree_b: dict,
    matched: list,
    added: list,
    removed: list,
) -> None:
    consumed_b: set[str] = set()
    counters_a: dict[tuple, int] = {}
    slots_b: dict[tuple, list[Span]] = {}
    for span in b_spans:
        slots_b.setdefault((span.name, span.kind), []).append(span)

    for span_a in a_spans:
        key = (span_a.name, span_a.kind)
        index = counters_a.get(key, 0)
        counters_a[key] = index + 1
        candidates = slots_b.get(key, [])
        if index < len(candidates):
            span_b = candidates[index]
            consumed_b.add(span_b.span_id)
            matched.append((span_a, span_b))
            _match_level(
                tree_a.get(span_a.span_id, []),
                tree_b.get(span_b.span_id, []),
                tree_a,
                tree_b,
                matched,
                added,
                removed,
            )
        else:
            removed.append(span_a)
            for descendant in _descendants(span_a, tree_a):
                removed.append(descendant)

    for span_b in b_spans:
        if span_b.span_id not in consumed_b:
            added.append(span_b)
            for descendant in _descendants(span_b, tree_b):
                added.append(descendant)


def _descendants(span: Span, tree: dict) -> list[Span]:
    out = []
    stack = list(tree.get(span.span_id, []))
    while stack:
        current = stack.pop()
        out.append(current)
        stack.extend(tree.get(current.span_id, []))
    return out


def diff_traces(
    store_a: Store,
    trace_id_a: str,
    store_b: Store | None = None,
    trace_id_b: str | None = None,
) -> dict[str, Any]:
    """Compare two traces. With one store, pass trace_id_b only."""
    store_b = store_b or store_a
    if trace_id_b is None:
        raise ValueError("trace_id_b is required")
    spans_a = store_a.get_spans(trace_id_a, hydrate=True, hydrate_keys=("request", "response"))
    spans_b = store_b.get_spans(trace_id_b, hydrate=True, hydrate_keys=("request", "response"))

    tree_a, tree_b = _tree(spans_a), _tree(spans_b)
    matched_pairs: list[tuple[Span, Span]] = []
    added: list[Span] = []
    removed: list[Span] = []
    roots_a, roots_b = tree_a.get(None, []), tree_b.get(None, [])
    if len(roots_a) == 1 and len(roots_b) == 1:
        # Comparing two runs implies their roots correspond, even when the
        # run was renamed between recordings.
        matched_pairs.append((roots_a[0], roots_b[0]))
        _match_level(
            tree_a.get(roots_a[0].span_id, []),
            tree_b.get(roots_b[0].span_id, []),
            tree_a, tree_b, matched_pairs, added, removed,
        )
    else:
        _match_level(
            roots_a, roots_b, tree_a, tree_b, matched_pairs, added, removed,
        )

    matched = []
    for span_a, span_b in matched_pairs:
        entry: dict[str, Any] = {
            "name": span_a.name,
            "kind": span_a.kind,
            "a": _span_brief(span_a),
            "b": _span_brief(span_b),
            "deltas": {
                "duration_ms": round(_duration_ms(span_b) - _duration_ms(span_a), 3),
                "cost_usd": _metric(span_b, "cost_usd") - _metric(span_a, "cost_usd"),
                "input_tokens": int(
                    _metric(span_b, "gen_ai.usage.input_tokens")
                    + _metric(span_b, "cached_input_tokens")
                    + _metric(span_b, "cache_write_tokens")
                    - _metric(span_a, "gen_ai.usage.input_tokens")
                    - _metric(span_a, "cached_input_tokens")
                    - _metric(span_a, "cache_write_tokens")
                ),
                "output_tokens": int(
                    _metric(span_b, "gen_ai.usage.output_tokens")
                    - _metric(span_a, "gen_ai.usage.output_tokens")
                ),
            },
        }
        if span_a.kind == "llm":
            request_diff = _text_diff(
                "request", span_a.attributes.get("request"), span_b.attributes.get("request")
            )
            response_diff = _text_diff(
                "response", span_a.attributes.get("response"), span_b.attributes.get("response")
            )
            if request_diff:
                entry["request_diff"] = request_diff
            if response_diff:
                entry["response_diff"] = response_diff
        matched.append(entry)

    def totals(spans: list[Span]) -> dict[str, Any]:
        return {
            "spans": len(spans),
            "llm_calls": sum(1 for s in spans if s.kind == "llm"),
            "errors": sum(1 for s in spans if s.status == "error"),
            "duration_ms": round(
                max((s.end_ns or s.start_ns) for s in spans) / 1e6
                - min(s.start_ns for s in spans) / 1e6,
                3,
            )
            if spans
            else 0.0,
            "cost_usd": sum(_metric(s, "cost_usd") for s in spans),
            "input_tokens": int(
                sum(
                    _metric(s, "gen_ai.usage.input_tokens")
                    + _metric(s, "cached_input_tokens")
                    + _metric(s, "cache_write_tokens")
                    for s in spans
                )
            ),
            "output_tokens": int(
                sum(_metric(s, "gen_ai.usage.output_tokens") for s in spans)
            ),
        }

    return {
        "trace_a": trace_id_a,
        "trace_b": trace_id_b,
        "totals_a": totals(spans_a),
        "totals_b": totals(spans_b),
        "matched": matched,
        "added": [_span_brief(s) for s in added],
        "removed": [_span_brief(s) for s in removed],
    }
