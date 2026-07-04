from traceburn.analyze.flamegraph import fold, waterfall
from traceburn.schema import Span

MS = 1_000_000


def make_span(span_id, name, start_ms, end_ms, parent=None, kind="custom", cost=None):
    attributes = {"cost_usd": cost} if cost is not None else {}
    return Span(
        span_id=span_id,
        trace_id="t1",
        name=name,
        kind=kind,
        parent_id=parent,
        start_ns=start_ms * MS,
        end_ns=end_ms * MS,
        attributes=attributes,
    )


def sample_spans():
    return [
        make_span("root", "agent", 0, 100, kind="agent"),
        make_span("c1", "plan", 10, 40, parent="root", kind="llm", cost=0.02),
        make_span("c2", "search", 50, 70, parent="root", kind="tool"),
        make_span("g1", "embed", 12, 20, parent="c1", kind="llm", cost=0.001),
    ]


def test_fold_latency_totals_and_self_time():
    tree = fold(sample_spans(), weight="latency")
    assert tree["value"] == 100 * MS
    root = tree["children"][0]
    assert root["name"] == "agent"
    assert root["value"] == 100 * MS
    assert root["self_value"] == (100 - 30 - 20) * MS
    plan = root["children"][0]
    assert plan["value"] == 30 * MS
    assert plan["self_value"] == (30 - 8) * MS


def test_fold_latency_self_time_clamped_for_overlapping_children():
    spans = [
        make_span("root", "agent", 0, 10, kind="agent"),
        make_span("a", "task-a", 0, 10, parent="root"),
        make_span("b", "task-b", 0, 10, parent="root"),
    ]
    tree = fold(spans, weight="latency")
    assert tree["children"][0]["self_value"] == 0


def test_fold_cost_sums_up_the_tree():
    tree = fold(sample_spans(), weight="cost")
    root = tree["children"][0]
    assert abs(root["value"] - 0.021) < 1e-9
    assert root["self_value"] == 0
    plan = root["children"][0]
    assert abs(plan["value"] - 0.021) < 1e-9
    assert abs(plan["self_value"] - 0.02) < 1e-9


def test_fold_rejects_unknown_weight():
    import pytest

    with pytest.raises(ValueError, match="weight"):
        fold(sample_spans(), weight="tokens")


def test_fold_handles_multiple_roots():
    spans = [
        make_span("r1", "first", 0, 10),
        make_span("r2", "second", 20, 40),
    ]
    tree = fold(spans, weight="latency")
    assert [c["name"] for c in tree["children"]] == ["first", "second"]
    assert tree["value"] == 30 * MS


def test_waterfall_depths_and_order():
    rows = waterfall(sample_spans())
    assert [r["name"] for r in rows] == ["agent", "plan", "embed", "search"]
    depths = {r["name"]: r["depth"] for r in rows}
    assert depths == {"agent": 0, "plan": 1, "embed": 2, "search": 1}
