"""Oversized and redundant prompt content.

Two checks, deliberately narrow:

1. Duplicate blocks inside one request: the same block of text (200+
   characters) appears more than once in a single prompt, usually from
   history re-appending or a retriever returning the same chunk twice.
   The repeat tokens are quantified at the model's input rate; high
   confidence because the duplication is verbatim.

2. Prompt dominance: a call whose prompt is at least 8000 tokens while the
   output is under 2 percent of that. Informational only; no dollar figure
   is claimed because trimming is a judgment call, not arithmetic.
"""

from __future__ import annotations

from collections import Counter

from ...schema import Finding
from ._common import (
    RuleContext,
    estimate,
    message_texts,
    model,
    output_tokens,
    prompt_tokens,
    provider,
    span_when,
)

RULE_ID = "context_bloat"
MIN_DUP_BLOCK_CHARS = 200
MIN_DUP_TOKENS = 256
DOMINANCE_MIN_PROMPT = 8000
DOMINANCE_MAX_OUTPUT_RATIO = 0.02


def run(ctx: RuleContext) -> list[Finding]:
    findings = []
    dominance_candidates = []

    for span in ctx.ok_llm_spans():
        texts = [t.strip() for t in message_texts(span.attributes.get("request"))]
        counts = Counter(t for t in texts if len(t) >= MIN_DUP_BLOCK_CHARS)
        dup_tokens = sum(
            (n - 1) * estimate(text) for text, n in counts.items() if n > 1
        )
        if dup_tokens >= MIN_DUP_TOKENS:
            price = ctx.pricing.lookup(provider(span), model(span))
            avoidable_usd = None
            if price is not None:
                rate = price.input_per_mtok
                if (span.attributes.get("cached_input_tokens") or 0) > 0 and (
                    price.cached_input_per_mtok is not None
                ):
                    # The prompt was (partly) cache-served; price the repeat
                    # tokens at the cached rate, the conservative bound.
                    rate = price.cached_input_per_mtok
                avoidable_usd = dup_tokens * rate / 1e6
            repeated = [f"{n}x a {estimate(t)}-token block" for t, n in counts.items() if n > 1]
            findings.append(
                Finding(
                    rule_id=RULE_ID,
                    severity="medium",
                    summary=f"duplicate content inside one prompt ({dup_tokens} repeated tokens)",
                    explanation=(
                        f"{span_when(span)} sent the same text more than once in a "
                        f"single request: {', '.join(repeated)}. Deduplicate the "
                        f"context before sending; the repeats add tokens without "
                        f"adding information."
                    ),
                    trace_id=ctx.trace_id,
                    span_ids=[span.span_id],
                    avoidable_tokens=dup_tokens,
                    avoidable_usd=avoidable_usd,
                    confidence="high",
                )
            )

        p_tokens = prompt_tokens(span)
        o_tokens = output_tokens(span)
        if p_tokens >= DOMINANCE_MIN_PROMPT and o_tokens <= p_tokens * DOMINANCE_MAX_OUTPUT_RATIO:
            dominance_candidates.append((p_tokens, o_tokens, span))

    if dominance_candidates:
        dominance_candidates.sort(reverse=True, key=lambda item: item[0])
        p_tokens, o_tokens, span = dominance_candidates[0]
        findings.append(
            Finding(
                rule_id=RULE_ID,
                severity="info",
                summary=(
                    f"{len(dominance_candidates)} call(s) send large prompts for "
                    f"tiny outputs (worst: {p_tokens} in, {o_tokens} out)"
                ),
                explanation=(
                    f"{span_when(span)} sent {p_tokens} prompt tokens to produce "
                    f"{o_tokens} output tokens. If the output does not depend on "
                    f"most of that context, trimming or summarizing it cuts cost "
                    f"and latency. No saving is claimed automatically; how much "
                    f"of the context is needed is a judgment call."
                ),
                trace_id=ctx.trace_id,
                span_ids=[s.span_id for _, _, s in dominance_candidates],
                avoidable_tokens=None,
                avoidable_usd=None,
                confidence="low",
            )
        )
    return findings
