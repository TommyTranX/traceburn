"""Expensive models on trivial steps.

Flags groups of small calls (short prompt, short output, no tool use) that
ran on a model priced at least five times above the provider's cheapest
listed model. The dollar figure is what the same tokens would have cost on
that cheapest model, so it is an upper bound on the saving and the finding
is explicitly a suggestion: quality on the cheaper model must be verified
by the person who owns the step. Confidence is low by design.
"""

from __future__ import annotations

from ...schema import Finding
from ._common import (
    RuleContext,
    cost,
    model,
    output_tokens,
    prompt_tokens,
    provider,
    span_when,
)

RULE_ID = "model_overkill"
RATE_MULTIPLE = 5.0
MAX_PROMPT_TOKENS = 800
MAX_OUTPUT_TOKENS = 150
MIN_GROUP_AVOIDABLE_USD = 0.0005


def run(ctx: RuleContext) -> list[Finding]:
    groups: dict[tuple, list] = {}
    for span in ctx.ok_llm_spans():
        prov, mdl = provider(span), model(span)
        if not prov or not mdl:
            continue
        response = span.attributes.get("response") or {}
        if isinstance(response, dict) and response.get("tool_calls"):
            continue
        if prompt_tokens(span) > MAX_PROMPT_TOKENS:
            continue
        if output_tokens(span) > MAX_OUTPUT_TOKENS:
            continue
        groups.setdefault((prov, mdl, span.name), []).append(span)

    findings = []
    for (prov, mdl, name), spans in groups.items():
        price = ctx.pricing.lookup(prov, mdl)
        cheapest = ctx.pricing.cheapest_for_provider(prov)
        if price is None or cheapest is None:
            continue
        cheap_name, cheap_price = cheapest
        if cheap_name.split("/", 1)[-1] == mdl.lower():
            continue
        if price.input_per_mtok < RATE_MULTIPLE * cheap_price.input_per_mtok:
            continue
        actual = sum(cost(s) for s in spans)
        alt = sum(
            (
                prompt_tokens(s) * cheap_price.input_per_mtok
                + output_tokens(s) * cheap_price.output_per_mtok
            )
            / 1e6
            for s in spans
        )
        avoidable = actual - alt
        if avoidable < MIN_GROUP_AVOIDABLE_USD:
            continue
        cheap_model = cheap_name.split("/", 1)[-1]
        findings.append(
            Finding(
                rule_id=RULE_ID,
                severity="low",
                summary=(
                    f"{len(spans)} small call(s) named '{name}' ran on {mdl}; "
                    f"a cheaper model may do"
                ),
                explanation=(
                    f"{span_when(spans[0])} and its repeats are short prompts with "
                    f"short outputs and no tool use, running on {mdl} at "
                    f"${price.input_per_mtok}/MTok input. The same tokens on "
                    f"{cheap_model} (this provider's cheapest listed model) would "
                    f"cost about ${alt:.4f} instead of ${actual:.4f}. This is a "
                    f"suggestion, not a verdict: try the step on a smaller model "
                    f"and check the quality holds."
                ),
                trace_id=ctx.trace_id,
                span_ids=[s.span_id for s in spans],
                avoidable_tokens=None,
                avoidable_usd=avoidable,
                confidence="low",
            )
        )
    return findings
