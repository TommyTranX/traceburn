"""Missed prompt-cache opportunities.

When several calls to the same model within a cache-lifetime window share a
long stable prefix and none of them read from or wrote to a prompt cache,
the static prefix was paid for at the full input rate every time. The
minimum cacheable prefix is provider-dependent (OpenAI caches from about
1024 tokens; Anthropic's per-model minimums reach 4096, so 4096 is used
there and for unknown providers, conservatively). The window is five
minutes, the shortest cache lifetime either provider prices by default.
The saving is computed from the pricing table's cached-input rate (and
charged for the first call's cache write where the provider bills one).
When the table has no cache pricing for the model, no dollar figure is
claimed.

Prefix length is estimated from the recorded request text; the span is the
evidence, the estimate is labeled an estimate.

When the provider is anthropic and the shared prefix is verifiably the
whole ``system`` string (not something spanning into the messages too),
the finding carries a renderable fix (see ``analyze/fix.py``): wrap that
field as a cache_control block. Anything less certain stays unfixed rather
than guess at code we cannot see.
"""

from __future__ import annotations

from ...schema import Finding
from ._common import RuleContext, estimate, model, provider, request_text, span_when

RULE_ID = "cache"
MIN_PREFIX_TOKENS = {"openai": 1024, "anthropic": 4096}
DEFAULT_MIN_PREFIX_TOKENS = 4096
WINDOW_SECONDS = 300


def _common_prefix_len(texts: list[str]) -> int:
    if not texts:
        return 0
    first = texts[0]
    limit = min(len(t) for t in texts)
    for i in range(limit):
        c = first[i]
        for text in texts[1:]:
            if text[i] != c:
                return i
    return limit


def _windows(spans, gap_ns: int):
    """Split a start-sorted span list where the gap exceeds the window."""
    group = [spans[0]]
    for span in spans[1:]:
        if span.start_ns - group[-1].start_ns <= gap_ns:
            group.append(span)
        else:
            yield group
            group = [span]
    yield group


def run(ctx: RuleContext) -> list[Finding]:
    findings = []
    by_model: dict[tuple, list] = {}
    for span in ctx.ok_llm_spans():
        attrs = span.attributes
        if (attrs.get("cached_input_tokens") or 0) > 0:
            continue
        if (attrs.get("cache_write_tokens") or 0) > 0:
            continue
        key = (provider(span), model(span))
        if key[1] is None:
            continue
        by_model.setdefault(key, []).append(span)

    for (prov, mdl), spans in by_model.items():
        spans.sort(key=lambda s: s.start_ns)
        if len(spans) < 2:
            continue
        price = ctx.pricing.lookup(prov, mdl)
        min_prefix = MIN_PREFIX_TOKENS.get(
            (prov or "").lower(), DEFAULT_MIN_PREFIX_TOKENS
        )
        for window in _windows(spans, WINDOW_SECONDS * 1_000_000_000):
            if len(window) < 2:
                continue
            texts = [request_text(s.attributes.get("request")) for s in window]
            prefix_chars = _common_prefix_len(texts)
            if prefix_chars == 0:
                continue
            prefix_tokens = estimate(texts[0][:prefix_chars])
            if prefix_tokens < min_prefix:
                continue

            avoidable_usd = None
            if price is not None and price.cached_input_per_mtok is not None:
                repeats = len(window) - 1
                saving = repeats * prefix_tokens * (
                    price.input_per_mtok - price.cached_input_per_mtok
                ) / 1e6
                if price.cache_write_per_mtok is not None:
                    saving -= prefix_tokens * (
                        price.cache_write_per_mtok - price.input_per_mtok
                    ) / 1e6
                if saving <= 0:
                    continue
                avoidable_usd = saving

            first = window[0]

            fix = None
            if (prov or "").lower() == "anthropic":
                system_val = (first.attributes.get("request") or {}).get("system")
                # Two conditions, both necessary: the shared prefix must
                # cover the whole system field (so it is genuinely identical
                # across calls, safe to wrap statically), and the system
                # field must clear the provider's cache minimum on its own
                # (so the suggested fix actually qualifies for caching,
                # rather than wrapping something too short to matter).
                if (
                    isinstance(system_val, str)
                    and prefix_chars >= len(system_val)
                    and estimate(system_val) >= min_prefix
                ):
                    fix = {"kind": "anthropic_cache_control", "span_id": first.span_id}

            findings.append(
                Finding(
                    rule_id=RULE_ID,
                    severity="medium",
                    summary=(
                        f"{len(window)} calls to {mdl} re-sent an uncached "
                        f"~{prefix_tokens}-token prefix"
                    ),
                    explanation=(
                        f"Starting at {span_when(first)}, {len(window)} calls within "
                        f"{WINDOW_SECONDS // 60} minutes shared a stable prefix of roughly "
                        f"{prefix_tokens} tokens (estimated from the recorded prompts) and "
                        f"none used a prompt cache. Mark the static prefix cacheable "
                        f"(or keep it byte-identical and leading, so automatic caching "
                        f"applies) to pay the cached rate on repeats."
                    ),
                    trace_id=ctx.trace_id,
                    span_ids=[s.span_id for s in window],
                    avoidable_tokens=prefix_tokens * (len(window) - 1),
                    avoidable_usd=avoidable_usd,
                    confidence="medium",
                    fix=fix,
                )
            )
    return findings
