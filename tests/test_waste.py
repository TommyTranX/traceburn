"""Waste rules: each rule fires on a constructed positive case and stays
silent on a constructed negative case (spec 12.2)."""

import pytest

from traceburn.analyze import waste
from traceburn.analyze.waste import cache, context_bloat, duplicates, loops, model_overkill
from traceburn.analyze.waste._common import RuleContext
from traceburn.schema import Span, Trace, new_span_id, new_trace_id

MS = 1_000_000


def llm_span(
    name="chat gpt-test",
    model="gpt-test",
    provider="openai",
    in_tok=100,
    out_tok=20,
    cached=0,
    write=0,
    cost=0.001,
    status="ok",
    req_hash=None,
    request=None,
    response=None,
    start_ms=0,
    dur_ms=100,
):
    return Span(
        span_id=new_span_id(),
        trace_id="t",
        name=name,
        kind="llm",
        start_ns=start_ms * MS,
        end_ns=(start_ms + dur_ms) * MS,
        status=status,
        error="boom" if status == "error" else None,
        attributes={
            "gen_ai.system": provider,
            "gen_ai.request.model": model,
            "gen_ai.response.model": model,
            "gen_ai.usage.input_tokens": in_tok,
            "gen_ai.usage.output_tokens": out_tok,
            "cached_input_tokens": cached,
            "cache_write_tokens": write,
            "cost_usd": cost,
            "request_hash": req_hash or f"hash-{name}-{start_ms}",
            "request": request or {"messages": [{"role": "user", "content": f"prompt {name} {start_ms}"}]},
            "response": response or {"text": "answer", "tool_calls": []},
        },
    )


def ctx_for(spans, pricing):
    return RuleContext(trace_id="t", spans=spans, pricing=pricing)


# -- duplicates ------------------------------------------------------------


def test_duplicates_fires_on_identical_hashes(pricing):
    spans = [
        llm_span(req_hash="same", cost=0.01, start_ms=0),
        llm_span(req_hash="same", cost=0.01, start_ms=200),
        llm_span(req_hash="same", cost=0.01, start_ms=400),
    ]
    findings = duplicates.run(ctx_for(spans, pricing))
    assert len(findings) == 1
    f = findings[0]
    assert f.rule_id == "duplicates"
    assert f.avoidable_usd == pytest.approx(0.02)
    assert len(f.span_ids) == 3
    assert f.confidence == "high"


def test_duplicates_silent_on_distinct_calls(pricing):
    spans = [llm_span(req_hash=f"h{i}", start_ms=i * 100) for i in range(4)]
    assert duplicates.run(ctx_for(spans, pricing)) == []


def test_near_duplicates_detected_on_in_place_edit(pricing):
    base = ["word%d" % i for i in range(100)]
    edited = list(base)
    edited[3] = "changed"
    text_a = " ".join(base)
    text_b = " ".join(edited)
    spans = [
        llm_span(req_hash="h1", in_tok=200, cost=0.02,
                 request={"messages": [{"role": "user", "content": text_a}]}),
        llm_span(req_hash="h2", in_tok=200, cost=0.02, start_ms=100,
                 request={"messages": [{"role": "user", "content": text_b}]}),
    ]
    findings = duplicates.run(ctx_for(spans, pricing))
    assert len(findings) == 1
    assert findings[0].confidence == "medium"
    assert findings[0].avoidable_usd == pytest.approx(0.02)


def test_near_duplicates_silent_on_shared_template_different_payload(pricing):
    template = "policy manual text " * 200
    spans = [
        llm_span(req_hash=f"h{i}", in_tok=1000, start_ms=i * 100,
                 request={"messages": [{"role": "user", "content": template + tail}]})
        for i, tail in enumerate(["ticket about billing", "ticket about sso loops"])
    ]
    assert duplicates.run(ctx_for(spans, pricing)) == []


def test_near_duplicates_silent_on_different_prompts(pricing):
    spans = [
        llm_span(req_hash="h1", in_tok=200,
                 request={"messages": [{"role": "user", "content": "a completely different question about databases"}]}),
        llm_span(req_hash="h2", in_tok=200, start_ms=100,
                 request={"messages": [{"role": "user", "content": "an unrelated poem in the style of haiku"}]}),
    ]
    assert duplicates.run(ctx_for(spans, pricing)) == []


# -- cache -----------------------------------------------------------------


def big_prefix_request(tail):
    prefix = "system instructions " * 300  # ~6000 chars, ~1500 estimated tokens
    return {
        "messages": [
            {"role": "system", "content": prefix},
            {"role": "user", "content": tail},
        ]
    }


def test_cache_fires_on_uncached_stable_prefix(pricing):
    spans = [
        llm_span(req_hash=f"h{i}", start_ms=i * 1000, in_tok=1600,
                 request=big_prefix_request(f"question {i}"))
        for i in range(3)
    ]
    findings = cache.run(ctx_for(spans, pricing))
    assert len(findings) == 1
    f = findings[0]
    assert f.rule_id == "cache"
    assert f.avoidable_tokens > 2000
    assert f.avoidable_usd is not None and f.avoidable_usd > 0
    assert len(f.span_ids) == 3


def test_cache_silent_when_cache_was_used(pricing):
    spans = [
        llm_span(req_hash=f"h{i}", start_ms=i * 1000, cached=1500,
                 request=big_prefix_request(f"question {i}"))
        for i in range(3)
    ]
    assert cache.run(ctx_for(spans, pricing)) == []


def test_cache_silent_on_short_or_unstable_prefixes(pricing):
    spans = [
        llm_span(req_hash=f"h{i}", start_ms=i * 1000,
                 request={"messages": [{"role": "user", "content": f"totally different {i} " * 5}]})
        for i in range(3)
    ]
    assert cache.run(ctx_for(spans, pricing)) == []


def test_cache_silent_when_calls_are_far_apart(pricing):
    spans = [
        llm_span(req_hash=f"h{i}", start_ms=i * 700_000, in_tok=1600,
                 request=big_prefix_request(f"question {i}"))
        for i in range(3)
    ]
    assert cache.run(ctx_for(spans, pricing)) == []


# -- context_bloat -----------------------------------------------------------


def test_context_bloat_fires_on_intra_request_duplicates(pricing):
    chunk = "retrieved paragraph about the topic " * 40  # ~1440 chars
    request = {
        "messages": [
            {"role": "user", "content": chunk},
            {"role": "user", "content": chunk},
            {"role": "user", "content": "now answer"},
        ]
    }
    findings = context_bloat.run(ctx_for([llm_span(request=request)], pricing))
    dup = [f for f in findings if "duplicate content" in f.summary]
    assert len(dup) == 1
    assert dup[0].avoidable_tokens >= 256
    assert dup[0].confidence == "high"


def test_context_bloat_silent_on_unique_content(pricing):
    request = {
        "messages": [
            {"role": "user", "content": "one paragraph " * 40},
            {"role": "user", "content": "a different paragraph " * 40},
        ]
    }
    findings = context_bloat.run(ctx_for([llm_span(request=request)], pricing))
    assert [f for f in findings if "duplicate content" in f.summary] == []


def test_context_bloat_dominance_is_informational(pricing):
    spans = [llm_span(in_tok=12_000, out_tok=40)]
    findings = context_bloat.run(ctx_for(spans, pricing))
    info = [f for f in findings if f.severity == "info"]
    assert len(info) == 1
    assert info[0].avoidable_usd is None


def test_context_bloat_dominance_silent_on_balanced_ratio(pricing):
    spans = [llm_span(in_tok=12_000, out_tok=2_000)]
    assert [f for f in context_bloat.run(ctx_for(spans, pricing)) if f.severity == "info"] == []


# -- model_overkill ----------------------------------------------------------


def test_model_overkill_fires_on_small_frontier_calls(pricing):
    spans = [
        llm_span(name="format-title", in_tok=200, out_tok=20,
                 cost=(200 * 2.0 + 20 * 8.0) / 1e6, start_ms=i * 100, req_hash=f"h{i}")
        for i in range(3)
    ]
    findings = model_overkill.run(ctx_for(spans, pricing))
    assert len(findings) == 1
    f = findings[0]
    assert f.rule_id == "model_overkill"
    assert f.confidence == "low"
    assert "suggestion" in f.explanation
    assert f.avoidable_usd > 0


def test_model_overkill_silent_on_cheap_model(pricing):
    spans = [
        llm_span(name="format-title", model="gpt-test-mini", in_tok=200, out_tok=20,
                 start_ms=i * 100, req_hash=f"h{i}")
        for i in range(3)
    ]
    assert model_overkill.run(ctx_for(spans, pricing)) == []


def test_model_overkill_silent_on_big_calls(pricing):
    spans = [llm_span(in_tok=5_000, out_tok=900, cost=0.02)]
    assert model_overkill.run(ctx_for(spans, pricing)) == []


def test_model_overkill_silent_on_tool_calls(pricing):
    spans = [
        llm_span(name="route", in_tok=200, out_tok=20, start_ms=i * 100, req_hash=f"h{i}",
                 response={"text": None, "tool_calls": [{"id": "x", "name": "t", "arguments": "{}"}]})
        for i in range(3)
    ]
    assert model_overkill.run(ctx_for(spans, pricing)) == []


# -- loops --------------------------------------------------------------------


def test_loops_fires_on_retry_storm(pricing):
    spans = [
        llm_span(req_hash="same", status="error", cost=0.0, start_ms=0, dur_ms=2000),
        llm_span(req_hash="same", status="error", cost=0.0, start_ms=2500, dur_ms=2000),
        llm_span(req_hash="same", status="ok", cost=0.01, start_ms=5000),
    ]
    findings = loops.run(ctx_for(spans, pricing))
    storm = [f for f in findings if "retry storm" in f.summary]
    assert len(storm) == 1
    assert storm[0].avoidable_seconds == pytest.approx(4.0)
    assert storm[0].severity == "medium"


def test_loops_silent_without_errors(pricing):
    spans = [llm_span(req_hash=f"h{i}", start_ms=i * 100) for i in range(4)]
    assert [f for f in loops.run(ctx_for(spans, pricing)) if "retry" in f.summary] == []


def test_loops_fires_on_repeated_identical_tool_calls(pricing):
    tools = [
        Span(span_id=new_span_id(), trace_id="t", name="search", kind="tool",
             start_ns=i * 100 * MS, end_ns=(i * 100 + 50) * MS,
             attributes={"request": {"query": "same thing"}})
        for i in range(3)
    ]
    findings = loops.run(ctx_for(tools, pricing))
    assert len(findings) == 1
    assert "identical arguments" in findings[0].summary


def test_loops_tool_calls_silent_on_different_args(pricing):
    tools = [
        Span(span_id=new_span_id(), trace_id="t", name="search", kind="tool",
             start_ns=i * 100 * MS, end_ns=(i * 100 + 50) * MS,
             attributes={"request": {"query": f"thing {i}"}})
        for i in range(3)
    ]
    assert loops.run(ctx_for(tools, pricing)) == []


def test_loops_runaway_step_count(pricing):
    spans = [llm_span(req_hash=f"h{i}", start_ms=i * 10) for i in range(51)]
    findings = loops.run(ctx_for(spans, pricing))
    assert any("LLM calls in one trace" in f.summary for f in findings)


# -- runner --------------------------------------------------------------------


def test_report_totals_and_ranking(store, pricing):
    trace_id = new_trace_id()
    store.insert_trace(Trace(trace_id=trace_id, name="run", start_ns=0))
    spans = [
        llm_span(req_hash="dup", cost=0.02, start_ms=0, model="unpriced-model"),
        llm_span(req_hash="dup", cost=0.02, start_ms=100, model="unpriced-model"),
        llm_span(req_hash="other", cost=0.005, start_ms=200, model="unpriced-model"),
    ]
    for s in spans:
        s.trace_id = trace_id
        store.insert_span(s)

    result = waste.report(store, trace_id, pricing=pricing)
    assert result["total_cost_usd"] == pytest.approx(0.045)
    assert result["avoidable_usd"] == pytest.approx(0.02)
    assert result["avoidable_by_rule"] == {"duplicates": pytest.approx(0.02)}
    assert "duplicates" in result["findings"][0]["rule_id"]
    assert "$0.0200" in result["headline"]


def test_report_clean_trace(store, pricing):
    trace_id = new_trace_id()
    store.insert_trace(Trace(trace_id=trace_id, name="run", start_ns=0))
    span = llm_span(cost=0.003, model="unpriced-model")
    span.trace_id = trace_id
    store.insert_span(span)
    result = waste.report(store, trace_id, pricing=pricing)
    assert result["findings"] == []
    assert "No waste found" in result["headline"]


# -- review regression: precision guards --------------------------------------


def test_duplicates_silent_on_intentional_sampling(pricing):
    spans = [
        llm_span(req_hash="same", cost=0.01, start_ms=i * 100,
                 request={"messages": [{"role": "user", "content": "brainstorm"}], "temperature": 0.9},
                 response={"text": f"idea {i}", "tool_calls": []})
        for i in range(3)
    ]
    assert duplicates.run(ctx_for(spans, pricing)) == []


def test_duplicates_medium_confidence_when_answers_differ_without_sampling(pricing):
    spans = [
        llm_span(req_hash="same", cost=0.01, start_ms=i * 100,
                 response={"text": f"answer {i}", "tool_calls": []})
        for i in range(2)
    ]
    findings = duplicates.run(ctx_for(spans, pricing))
    assert len(findings) == 1
    assert findings[0].confidence == "medium"


def test_near_duplicates_silent_on_growing_history(pricing):
    base = ["word%d" % i for i in range(200)]
    msgs = [{"role": "system", "content": " ".join(base)}]
    spans = []
    for turn in range(3):
        msgs = msgs + [{"role": "user", "content": f"turn {turn} says something"}]
        spans.append(
            llm_span(req_hash=f"h{turn}", in_tok=300, start_ms=turn * 100,
                     request={"messages": list(msgs)})
        )
    assert duplicates.run(ctx_for(spans, pricing)) == []


def test_cache_anthropic_needs_bigger_prefix(pricing):
    prefix = "system instructions " * 300  # ~1500 estimated tokens
    spans = [
        llm_span(model="claude-test", provider="anthropic", req_hash=f"h{i}",
                 start_ms=i * 1000, in_tok=1600,
                 request={"messages": [
                     {"role": "system", "content": prefix},
                     {"role": "user", "content": f"q{i}"},
                 ]})
        for i in range(3)
    ]
    # 1500 tokens is cacheable on openai but below anthropic's minimum.
    assert cache.run(ctx_for(spans, pricing)) == []


def test_report_totals_do_not_double_count_across_rules(store, pricing):
    trace_id = new_trace_id()
    store.insert_trace(Trace(trace_id=trace_id, name="run", start_ns=0))
    prefix = "shared preamble " * 400  # big enough for the cache rule
    spans = [
        llm_span(req_hash="dup", cost=0.01, start_ms=i * 1000, in_tok=1700,
                 request={"messages": [
                     {"role": "system", "content": prefix},
                     {"role": "user", "content": "same question"},
                 ]})
        for i in range(3)
    ]
    for s in spans:
        s.trace_id = trace_id
        store.insert_span(s)
    result = waste.report(store, trace_id, pricing=pricing)
    rules_fired = {f["rule_id"] for f in result["findings"]}
    assert "duplicates" in rules_fired and "cache" in rules_fired
    assert result["avoidable_usd"] <= result["total_cost_usd"]
    assert result["avoidable_by_rule"].get("cache") is None


def test_loops_ignores_constant_tool_metadata(pricing):
    tools = [
        Span(span_id=new_span_id(), trace_id="t", name="search", kind="tool",
             start_ns=i * 100 * MS, end_ns=(i * 100 + 50) * MS,
             attributes={"category": "web", "provider": "generic"})
        for i in range(4)
    ]
    assert loops.run(ctx_for(tools, pricing)) == []


def test_loops_tool_rule_uses_request_attribute(pricing):
    tools = [
        Span(span_id=new_span_id(), trace_id="t", name="search", kind="tool",
             start_ns=i * 100 * MS, end_ns=(i * 100 + 50) * MS,
             attributes={"request": {"query": "same"}, "category": "web"})
        for i in range(3)
    ]
    findings = loops.run(ctx_for(tools, pricing))
    assert len(findings) == 1


def test_message_texts_include_tool_traffic():
    from traceburn.analyze.waste._common import message_texts

    request = {
        "messages": [
            {"role": "user", "content": "find the weather"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "NYC"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "72F and sunny"},
            ]},
        ]
    }
    texts = "\n".join(message_texts(request))
    assert "NYC" in texts
    assert "72F and sunny" in texts


def test_message_texts_handle_system_block_list():
    from traceburn.analyze.waste._common import message_texts

    request = {
        "system": [{"type": "text", "text": "policy manual text", "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": "ticket"}],
    }
    texts = message_texts(request)
    assert texts[0] == "policy manual text"


def test_cache_rule_fires_with_system_block_list(pricing):
    prefix = "policy manual section " * 900  # ~4950 estimated tokens
    spans = [
        llm_span(model="claude-test", provider="anthropic", req_hash=f"h{i}",
                 start_ms=i * 1000, in_tok=5100,
                 request={
                     "system": [{"type": "text", "text": prefix}],
                     "messages": [{"role": "user", "content": f"ticket {i}"}],
                 })
        for i in range(3)
    ]
    findings = cache.run(ctx_for(spans, pricing))
    assert len(findings) == 1
