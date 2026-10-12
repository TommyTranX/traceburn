"""Shared rules for recorded model-cost completeness."""
from __future__ import annotations

import math
from collections.abc import Iterable

from .schema import Span


def valid_cost(value) -> bool:
    try:
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value >= 0)
    except OverflowError:
        return False


def model_costs(spans: Iterable[Span]) -> dict:
    calls = [s for s in spans if s.kind == "llm"]
    known = [s.attributes.get("cost_usd") for s in calls
             if valid_cost(s.attributes.get("cost_usd"))]
    subtotal = math.fsum(known)
    unknown = len(calls) - len(known)
    return {"known_cost_usd": subtotal, "unknown_cost_calls": unknown,
            "llm_calls": len(calls),
            "total_cost_usd": None if unknown else subtotal}
