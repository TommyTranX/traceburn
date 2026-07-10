"""Render a finding's structured ``fix`` data into a copy-pasteable patch.

Only two kinds exist today, both narrow on purpose: a mechanical fix is only
worth emitting when it is derivable from the recorded request alone, with no
visibility into the caller's actual source code. Every other rule (loops,
duplicates, context bloat) needs a decision only a human can make, so
``render_fix`` returns None for them rather than guess at a patch.

``anthropic_cache_control`` shows the wrapping pattern generically (a
placeholder in place of the real prompt text), not the recorded prompt
itself: the transformation is what is actionable, and a real system prompt
can be long enough, or sensitive enough, that dumping it to a terminal by
default would be the wrong call.
"""

from __future__ import annotations

from ..schema import Finding
from ..store import Store

RENDERABLE_KINDS = ("anthropic_cache_control", "swap_model")


def _render_anthropic_cache_control(fix: dict, store: Store) -> str | None:
    span_id = fix.get("span_id")
    span = store.get_span(span_id, hydrate=True) if span_id else None
    if span is None:
        return None
    system_val = (span.attributes.get("request") or {}).get("system")
    if not isinstance(system_val, str):
        return None
    return (
        "Wrap the system prompt in a cache_control block. Before:\n\n"
        '    system="<your system prompt>"\n\n'
        "After:\n\n"
        "    system=[\n"
        "        {\n"
        '            "type": "text",\n'
        '            "text": "<your system prompt, unchanged>",\n'
        '            "cache_control": {"type": "ephemeral"},\n'
        "        }\n"
        "    ]"
    )


def _render_swap_model(fix: dict, store: Store) -> str | None:
    from_model = fix.get("from_model")
    to_model = fix.get("to_model")
    if not from_model or not to_model:
        return None
    return (
        f'    model="{from_model}"      # before\n'
        f'    model="{to_model}"      # after, try it and check quality'
    )


_RENDERERS = {
    "anthropic_cache_control": _render_anthropic_cache_control,
    "swap_model": _render_swap_model,
}


def render_fix(finding: Finding, store: Store) -> str | None:
    """The literal patch text for a finding's fix, or None if unrenderable."""
    if finding.fix is None:
        return None
    kind = finding.fix.get("kind")
    renderer = _RENDERERS.get(kind)
    if renderer is None:
        return None
    return renderer(finding.fix, store)
