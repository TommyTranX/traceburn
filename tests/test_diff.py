from traceburn.analyze.diff import diff_traces
from traceburn.schema import Span, Trace, new_span_id, new_trace_id

MS = 1_000_000


def build_trace(store, steps):
    """steps: list of (name, kind, attrs) tuples; first is the root."""
    trace_id = new_trace_id()
    store.insert_trace(Trace(trace_id=trace_id, name=steps[0][0], start_ns=0))
    root_id = None
    t = 0
    for name, kind, attrs in steps:
        span_id = new_span_id()
        store.insert_span(
            Span(
                span_id=span_id,
                trace_id=trace_id,
                parent_id=root_id,
                name=name,
                kind=kind,
                start_ns=t * MS,
                end_ns=(t + 10) * MS,
                attributes=attrs,
            )
        )
        if root_id is None:
            root_id = span_id
        t += 10
    return trace_id


def llm_attrs(in_tok, out_tok, cost, response_text="ok"):
    return {
        "gen_ai.usage.input_tokens": in_tok,
        "gen_ai.usage.output_tokens": out_tok,
        "cost_usd": cost,
        "request": {"messages": [{"role": "user", "content": "question"}]},
        "response": {"text": response_text, "tool_calls": []},
    }


def test_diff_matched_added_removed(store):
    a = build_trace(
        store,
        [
            ("agent", "agent", {}),
            ("plan", "llm", llm_attrs(1000, 100, 0.01)),
            ("search", "tool", {}),
            ("write", "llm", llm_attrs(2000, 400, 0.03)),
        ],
    )
    b = build_trace(
        store,
        [
            ("agent", "agent", {}),
            ("plan", "llm", llm_attrs(1400, 120, 0.015, response_text="different")),
            ("write", "llm", llm_attrs(2000, 400, 0.03)),
            ("verify", "tool", {}),
        ],
    )
    result = diff_traces(store, a, trace_id_b=b)

    names_matched = [m["name"] for m in result["matched"]]
    assert "agent" in names_matched and "plan" in names_matched and "write" in names_matched
    assert [s["name"] for s in result["removed"]] == ["search"]
    assert [s["name"] for s in result["added"]] == ["verify"]

    plan = next(m for m in result["matched"] if m["name"] == "plan")
    assert plan["deltas"]["input_tokens"] == 400
    assert plan["deltas"]["output_tokens"] == 20
    assert abs(plan["deltas"]["cost_usd"] - 0.005) < 1e-9
    assert "response_diff" in plan
    assert "request_diff" not in plan

    write = next(m for m in result["matched"] if m["name"] == "write")
    assert write["deltas"]["cost_usd"] == 0
    assert "response_diff" not in write


def test_diff_matches_repeated_names_by_occurrence(store):
    a = build_trace(
        store,
        [
            ("agent", "agent", {}),
            ("chat gpt", "llm", llm_attrs(100, 10, 0.001, "first-a")),
            ("chat gpt", "llm", llm_attrs(100, 10, 0.002, "second-a")),
        ],
    )
    b = build_trace(
        store,
        [
            ("agent", "agent", {}),
            ("chat gpt", "llm", llm_attrs(100, 10, 0.001, "first-a")),
            ("chat gpt", "llm", llm_attrs(100, 10, 0.005, "second-b")),
        ],
    )
    result = diff_traces(store, a, trace_id_b=b)
    chats = [m for m in result["matched"] if m["name"] == "chat gpt"]
    assert len(chats) == 2
    assert chats[0]["deltas"]["cost_usd"] == 0
    assert abs(chats[1]["deltas"]["cost_usd"] - 0.003) < 1e-9
    assert result["added"] == [] and result["removed"] == []


def test_diff_totals(store):
    a = build_trace(store, [("agent", "agent", {}), ("plan", "llm", llm_attrs(500, 50, 0.01))])
    b = build_trace(store, [("agent", "agent", {}), ("plan", "llm", llm_attrs(800, 80, 0.02))])
    result = diff_traces(store, a, trace_id_b=b)
    assert result["totals_a"]["cost_usd"] == 0.01
    assert result["totals_b"]["cost_usd"] == 0.02
    assert result["totals_a"]["input_tokens"] == 500
    assert result["totals_b"]["output_tokens"] == 80


def test_single_roots_match_despite_rename(store):
    a = build_trace(store, [("run-before", "agent", {}), ("plan", "llm", llm_attrs(500, 50, 0.01))])
    b = build_trace(store, [("run-after", "agent", {}), ("plan", "llm", llm_attrs(500, 50, 0.005))])
    result = diff_traces(store, a, trace_id_b=b)
    assert result["added"] == [] and result["removed"] == []
    plan = next(m for m in result["matched"] if m["name"] == "plan")
    assert abs(plan["deltas"]["cost_usd"] + 0.005) < 1e-9
