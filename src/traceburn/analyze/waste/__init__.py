"""The waste rule runner: run every rule over a trace, rank the findings.

The report dict and the Finding schema (see schema.py) are public JSON
interfaces. Rules are independent modules with one entry point,
``run(ctx) -> list[Finding]``; adding a rule means adding a module here and
listing it in ALL_RULES. A rule that raises is skipped with a log line, it
never takes the report down.
"""

from __future__ import annotations

import logging
from typing import Any

from ...pricing import PricingTable
from ...store import Store
from . import cache, context_bloat, duplicates, loops, model_overkill
from ._common import RuleContext, cost

logger = logging.getLogger("traceburn")

ALL_RULES = (duplicates, cache, context_bloat, loops, model_overkill)

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}


def report(store: Store, trace_id: str, pricing: PricingTable | None = None) -> dict[str, Any]:
    """Run all waste rules over one trace and return the ranked report."""
    spans = store.get_spans(trace_id, hydrate=True, hydrate_keys=("request", "response"))
    ctx = RuleContext(
        trace_id=trace_id,
        spans=spans,
        pricing=pricing or PricingTable.load(),
    )

    findings = []
    for rule in ALL_RULES:
        try:
            findings.extend(rule.run(ctx))
        except Exception:
            logger.warning("waste rule %s failed", rule.__name__, exc_info=True)

    findings.sort(
        key=lambda f: (_SEVERITY_ORDER.get(f.severity, 9), -(f.avoidable_usd or 0.0))
    )

    total_cost = sum(cost(s) for s in ctx.llm_spans)
    # Totals count each span once: when findings from different rules claim
    # overlapping spans, only the highest-ranked claim contributes, and the
    # sum never exceeds the run's cost. Per-finding figures are unchanged.
    avoidable_by_rule: dict[str, float] = {}
    claimed_spans: set[str] = set()
    for finding in findings:
        if not finding.avoidable_usd:
            continue
        if claimed_spans.intersection(finding.span_ids):
            continue
        claimed_spans.update(finding.span_ids)
        avoidable_by_rule[finding.rule_id] = (
            avoidable_by_rule.get(finding.rule_id, 0.0) + finding.avoidable_usd
        )
    total_avoidable = sum(avoidable_by_rule.values())
    if total_cost > 0:
        total_avoidable = min(total_avoidable, total_cost)

    if findings:
        top = findings[0]
        if total_cost > 0 and total_avoidable > 0:
            share = min(100.0, 100.0 * total_avoidable / total_cost)
            headline = (
                f"This run cost ${total_cost:.4f}; about ${total_avoidable:.4f} "
                f"({share:.0f}%) looks avoidable. Top issue: {top.summary}."
            )
        else:
            headline = f"Top issue: {top.summary}."
    else:
        headline = (
            f"No waste found by the current rules. Run cost ${total_cost:.4f}."
        )

    return {
        "trace_id": trace_id,
        "total_cost_usd": total_cost,
        "avoidable_usd": total_avoidable,
        "avoidable_by_rule": avoidable_by_rule,
        "headline": headline,
        "findings": [f.to_dict() for f in findings],
    }
