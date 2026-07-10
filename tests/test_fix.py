import pytest

from traceburn.analyze.fix import render_fix
from traceburn.schema import Finding, Span, Trace, new_span_id, new_trace_id


def insert_span(store, trace_id, **kwargs):
    defaults = dict(
        span_id=new_span_id(), trace_id=trace_id, name="chat", kind="llm",
        start_ns=0, end_ns=1_000_000, attributes={},
    )
    defaults.update(kwargs)
    span = Span(**defaults)
    store.insert_span(span)
    return span


def test_anthropic_cache_control_renders_wrapping_pattern(store):
    trace_id = new_trace_id()
    store.insert_trace(Trace(trace_id=trace_id, name="run", start_ns=0))
    span = insert_span(
        store, trace_id,
        attributes={"request": {"system": "you are a helpful assistant with a long policy manual"}},
    )
    finding = Finding(
        rule_id="cache", severity="medium", summary="x", explanation="y",
        trace_id=trace_id, span_ids=[span.span_id],
        fix={"kind": "anthropic_cache_control", "span_id": span.span_id},
    )
    patch = render_fix(finding, store)
    assert patch is not None
    assert "cache_control" in patch
    assert "ephemeral" in patch
    # The real prompt text is never dumped into the patch.
    assert "policy manual" not in patch


def test_anthropic_cache_control_none_when_system_not_a_string(store):
    trace_id = new_trace_id()
    store.insert_trace(Trace(trace_id=trace_id, name="run", start_ns=0))
    span = insert_span(
        store, trace_id,
        attributes={"request": {"system": [{"type": "text", "text": "already structured"}]}},
    )
    finding = Finding(
        rule_id="cache", severity="medium", summary="x", explanation="y",
        trace_id=trace_id, span_ids=[span.span_id],
        fix={"kind": "anthropic_cache_control", "span_id": span.span_id},
    )
    assert render_fix(finding, store) is None


def test_anthropic_cache_control_none_when_span_missing(store):
    finding = Finding(
        rule_id="cache", severity="medium", summary="x", explanation="y",
        trace_id="t", span_ids=[],
        fix={"kind": "anthropic_cache_control", "span_id": "does-not-exist"},
    )
    assert render_fix(finding, store) is None


def test_swap_model_renders_before_after(store):
    finding = Finding(
        rule_id="model_overkill", severity="low", summary="x", explanation="y",
        trace_id="t", fix={"kind": "swap_model", "from_model": "gpt-5", "to_model": "gpt-5-mini"},
    )
    patch = render_fix(finding, store)
    assert patch is not None
    assert "gpt-5" in patch
    assert "gpt-5-mini" in patch


def test_swap_model_none_when_incomplete(store):
    finding = Finding(
        rule_id="model_overkill", severity="low", summary="x", explanation="y",
        trace_id="t", fix={"kind": "swap_model", "from_model": "gpt-5"},
    )
    assert render_fix(finding, store) is None


def test_no_fix_returns_none(store):
    finding = Finding(
        rule_id="duplicates", severity="high", summary="x", explanation="y", trace_id="t",
    )
    assert render_fix(finding, store) is None


def test_unknown_fix_kind_returns_none(store):
    finding = Finding(
        rule_id="mystery", severity="low", summary="x", explanation="y",
        trace_id="t", fix={"kind": "not_a_real_kind"},
    )
    assert render_fix(finding, store) is None
