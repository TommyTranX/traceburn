"""Duplicate and near-duplicate LLM calls.

Exact duplicates share a request hash: the same normalized request was sent
more than once. When the recorded responses are identical too, every send
after the first bought the same answer again; confidence is high. When the
request sampled with temperature above zero and the responses differ, the
repetition is presumed intentional (best-of-n) and nothing is flagged.
Retry storms (duplicates involving errors) belong to the loops rule and are
excluded here.

Near-duplicates share a model, the same message count, and almost all of
their prompt text (token-set Jaccard at or above 0.9) without sharing a
hash. Requiring an equal message count keeps a normal agent loop, where
each turn extends the same history, from being flagged: those calls are
the cache rule's business, not duplication. Confidence is medium and only
the repeat cost is claimed.
"""

from __future__ import annotations

import json

from ...schema import Finding
from ._common import (
    RuleContext,
    cost,
    message_count,
    model,
    output_tokens,
    prompt_tokens,
    request_text,
    span_when,
)

RULE_ID = "duplicates"
JACCARD_THRESHOLD = 0.9
MIN_NEAR_DUP_TOKENS = 100


def _token_set(text: str) -> set[str]:
    return set(text.lower().split())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _response_key(span) -> str:
    response = span.attributes.get("response")
    try:
        return json.dumps(response, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(response)


def _sampling_temperature(span) -> float:
    request = span.attributes.get("request")
    if isinstance(request, dict):
        value = request.get("temperature")
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _group_finding(ctx: RuleContext, spans, exact: bool, same_answers: bool = True) -> Finding:
    repeats = spans[1:]
    avoidable_usd = sum(cost(s) for s in repeats)
    avoidable_tokens = sum(prompt_tokens(s) + output_tokens(s) for s in repeats)
    first = spans[0]
    kind = "identical" if exact else "nearly identical"
    if exact and same_answers:
        detail = "Every call after the first bought the same answer again."
        confidence = "high"
    elif exact:
        detail = (
            "The provider returned different answers to the identical request, "
            "so if the repetition was deliberate sampling, ignore this finding."
        )
        confidence = "medium"
    else:
        detail = "The requests differ only marginally."
        confidence = "medium"
    return Finding(
        rule_id=RULE_ID,
        severity="high" if avoidable_usd >= 0.01 and confidence == "high" else "medium",
        summary=f"{len(spans)} {kind} calls to {model(first) or 'unknown model'}",
        explanation=(
            f"{span_when(first)} was sent {len(spans)} times with "
            f"{'the same' if exact else 'nearly the same'} request. {detail} "
            f"Cache the result or hoist the call out of the loop."
        ),
        trace_id=ctx.trace_id,
        span_ids=[s.span_id for s in spans],
        avoidable_tokens=avoidable_tokens or None,
        avoidable_usd=avoidable_usd if avoidable_usd > 0 else None,
        confidence=confidence,
    )


def run(ctx: RuleContext) -> list[Finding]:
    findings = []
    by_hash: dict[str, list] = {}
    for span in ctx.ok_llm_spans():
        key = span.attributes.get("request_hash")
        if key:
            by_hash.setdefault(key, []).append(span)

    claimed = set()
    for group in by_hash.values():
        group.sort(key=lambda s: s.start_ns)
        if len(group) < 2:
            continue
        same_answers = len({_response_key(s) for s in group}) == 1
        if not same_answers and _sampling_temperature(group[0]) > 0:
            # Different answers under explicit sampling: presumed best-of-n
            # on purpose. Precision beats recall; stay silent.
            claimed.update(s.span_id for s in group)
            continue
        findings.append(_group_finding(ctx, group, exact=True, same_answers=same_answers))
        claimed.update(s.span_id for s in group)

    # Near-duplicates among the remainder, grouped greedily by similarity.
    remainder = [
        s
        for s in ctx.ok_llm_spans()
        if s.span_id not in claimed
        and prompt_tokens(s) >= MIN_NEAR_DUP_TOKENS
    ]
    token_sets = {
        s.span_id: _token_set(request_text(s.attributes.get("request"))) for s in remainder
    }
    used = set()
    for i, span_a in enumerate(remainder):
        if span_a.span_id in used:
            continue
        group = [span_a]
        for span_b in remainder[i + 1 :]:
            if span_b.span_id in used:
                continue
            if model(span_a) != model(span_b):
                continue
            # An agent loop extends the same history each turn; those are
            # near-supersets, not near-duplicates. Equal message counts keep
            # this rule to genuine resends.
            if message_count(span_a.attributes.get("request")) != message_count(
                span_b.attributes.get("request")
            ):
                continue
            if _jaccard(token_sets[span_a.span_id], token_sets[span_b.span_id]) >= JACCARD_THRESHOLD:
                group.append(span_b)
        if len(group) >= 2:
            used.update(s.span_id for s in group)
            group.sort(key=lambda s: s.start_ns)
            findings.append(_group_finding(ctx, group, exact=False))
    return findings
