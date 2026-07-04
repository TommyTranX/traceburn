"""Retry storms, repeated tool calls, and runaway step counts.

Retry storms are identical requests where at least one attempt errored:
the time burned is measured from the spans; failed attempts are usually
not billed, so dollars are only claimed for error spans that carry a cost.
Repeated tool calls need the tool span to carry its arguments under the
``request`` attribute (the documented convention); identical
(name, request) three or more times is flagged. Spans without a
``request`` attribute are never flagged, so constant metadata on tool
spans cannot trip the rule. A runaway step count (more than 50 LLM calls
in one trace) is reported as informational context.
"""

from __future__ import annotations

import json

from ...schema import Finding
from ._common import RuleContext, cost, duration_seconds, span_when

RULE_ID = "loops"
MIN_TOOL_REPEATS = 3
RUNAWAY_LLM_CALLS = 50


def run(ctx: RuleContext) -> list[Finding]:
    findings = []

    # Retry storms: same request hash, at least one error among the attempts.
    by_hash: dict[str, list] = {}
    for span in ctx.llm_spans:
        key = span.attributes.get("request_hash")
        if key:
            by_hash.setdefault(key, []).append(span)
    for group in by_hash.values():
        if len(group) < 2:
            continue
        errors = [s for s in group if s.status == "error"]
        if not errors:
            continue
        group.sort(key=lambda s: s.start_ns)
        wasted_seconds = sum(duration_seconds(s) for s in errors)
        error_cost = sum(cost(s) for s in errors)
        findings.append(
            Finding(
                rule_id=RULE_ID,
                severity="medium" if len(group) >= 3 else "low",
                summary=(
                    f"retry storm: {len(group)} attempts of the same request, "
                    f"{len(errors)} failed"
                ),
                explanation=(
                    f"{span_when(group[0])} was attempted {len(group)} times; "
                    f"{len(errors)} attempt(s) errored, burning "
                    f"{wasted_seconds:.1f}s of wall clock. Add backoff, fix the "
                    f"failing parameter, or cap the retries."
                ),
                trace_id=ctx.trace_id,
                span_ids=[s.span_id for s in group],
                avoidable_seconds=wasted_seconds or None,
                avoidable_usd=error_cost if error_cost > 0 else None,
                confidence="high",
            )
        )

    # Repeated tool calls with identical recorded arguments. Only the
    # "request" attribute counts as arguments; other attributes are
    # metadata and legitimately constant across calls.
    tool_groups: dict[tuple, list] = {}
    for span in ctx.spans:
        if span.kind != "tool":
            continue
        request = span.attributes.get("request")
        if request is None:
            continue
        try:
            args_key = json.dumps(request, sort_keys=True, default=str)
        except (TypeError, ValueError):
            continue
        tool_groups.setdefault((span.name, args_key), []).append(span)
    for (name, _), group in tool_groups.items():
        if len(group) < MIN_TOOL_REPEATS:
            continue
        group.sort(key=lambda s: s.start_ns)
        repeat_seconds = sum(duration_seconds(s) for s in group[1:])
        findings.append(
            Finding(
                rule_id=RULE_ID,
                severity="low",
                summary=f"tool '{name}' called {len(group)} times with identical arguments",
                explanation=(
                    f"{span_when(group[0])} ran {len(group)} times with the same "
                    f"recorded arguments. If the tool is deterministic, cache its "
                    f"result; the repeats spent {repeat_seconds:.1f}s."
                ),
                trace_id=ctx.trace_id,
                span_ids=[s.span_id for s in group],
                avoidable_seconds=repeat_seconds or None,
                confidence="medium",
            )
        )

    llm_count = len(ctx.llm_spans)
    if llm_count > RUNAWAY_LLM_CALLS:
        findings.append(
            Finding(
                rule_id=RULE_ID,
                severity="info",
                summary=f"{llm_count} LLM calls in one trace",
                explanation=(
                    f"This run made {llm_count} model calls. If that is more than "
                    f"the task warrants, look for an agent loop that is not "
                    f"converging; the other findings in this report say where the "
                    f"money went."
                ),
                trace_id=ctx.trace_id,
                span_ids=[],
                confidence="low",
            )
        )
    return findings
