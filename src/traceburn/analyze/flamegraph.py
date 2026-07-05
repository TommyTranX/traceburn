"""Fold a trace into a frame tree for flamegraph and waterfall rendering.

The fold output is a public JSON interface: a tree of frames, each with a
total ``value`` and a ``self_value``, where value is either wall-clock
nanoseconds (``weight="latency"``) or estimated dollars (``weight="cost"``).
Latency self-time is the span's duration minus its children's, clamped at
zero because concurrent children can legitimately overlap their parent.
"""

from __future__ import annotations

import math
from typing import Any

from ..schema import Span

WEIGHTS = ("latency", "cost")


def _duration(span: Span) -> int:
    if span.end_ns is None:
        return 0
    return max(0, span.end_ns - span.start_ns)


def _cost(span: Span) -> float:
    value = span.attributes.get("cost_usd")
    try:
        cost = float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
    # The store sanitizes non-finite floats to strings, which float() would
    # happily parse back into nan; keep JSON output JSON-serializable.
    return cost if math.isfinite(cost) else 0.0


def fold(spans: list[Span], weight: str = "latency") -> dict[str, Any]:
    """Fold spans into a frame tree rooted at a synthetic trace frame."""
    if weight not in WEIGHTS:
        raise ValueError(f"weight must be one of {WEIGHTS}, got {weight!r}")

    ids = {s.span_id for s in spans}
    children: dict[str | None, list[Span]] = {}
    for span in spans:
        parent = span.parent_id if span.parent_id in ids else None
        children.setdefault(parent, []).append(span)

    def build(span: Span) -> dict[str, Any]:
        kids = sorted(children.get(span.span_id, []), key=lambda s: s.start_ns)
        child_frames = [build(k) for k in kids]
        if weight == "latency":
            total = _duration(span)
            self_value = max(0, total - sum(_duration(k) for k in kids))
        else:
            own = _cost(span)
            total = own + sum(f["value"] for f in child_frames)
            self_value = own
        return {
            "name": span.name,
            "kind": span.kind,
            "span_id": span.span_id,
            "status": span.status,
            "value": total,
            "self_value": self_value,
            "children": child_frames,
        }

    roots = sorted(children.get(None, []), key=lambda s: s.start_ns)
    root_frames = [build(r) for r in roots]
    total = sum(f["value"] for f in root_frames)
    return {
        "name": "trace",
        "kind": "trace",
        "span_id": None,
        "status": "ok",
        "weight": weight,
        "value": total,
        "self_value": 0,
        "children": root_frames,
    }


def waterfall(spans: list[Span]) -> list[dict[str, Any]]:
    """Flat timeline: one row per span with its depth, ordered by start."""
    ids = {s.span_id for s in spans}
    by_id = {s.span_id: s for s in spans}

    def depth(span: Span) -> int:
        level = 0
        current = span
        while current.parent_id in ids:
            current = by_id[current.parent_id]
            level += 1
            if level > len(spans):
                break
        return level

    rows = []
    for span in sorted(spans, key=lambda s: s.start_ns):
        rows.append(
            {
                "span_id": span.span_id,
                "name": span.name,
                "kind": span.kind,
                "status": span.status,
                "depth": depth(span),
                "start_ns": span.start_ns,
                "end_ns": span.end_ns,
                "duration_ns": None if span.end_ns is None else _duration(span),
                "cost_usd": span.attributes.get("cost_usd"),
            }
        )
    return rows
