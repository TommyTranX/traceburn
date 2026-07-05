import pytest

starlette = pytest.importorskip("starlette")

from starlette.testclient import TestClient

from traceburn.recorder import Recorder
from traceburn.store import Store
from traceburn.ui.server import create_app


@pytest.fixture
def client(tmp_path):
    db = str(tmp_path / "traces.db")
    recorder = Recorder(store=Store(db))
    with recorder.session("nightly"):
        with recorder.span("run-a", kind="agent"):
            with recorder.span(
                "chat gpt-test",
                kind="llm",
                attributes={
                    "gen_ai.system": "openai",
                    "gen_ai.request.model": "gpt-test",
                    "gen_ai.usage.input_tokens": 1000,
                    "gen_ai.usage.output_tokens": 100,
                    "cost_usd": 0.01,
                    "request": {"messages": [{"role": "user", "content": "hi"}]},
                    "response": {"text": "hello", "tool_calls": []},
                    "request_hash": "dup",
                },
            ):
                pass
            with recorder.span(
                "chat gpt-test",
                kind="llm",
                attributes={"cost_usd": 0.01, "request_hash": "dup",
                            "gen_ai.request.model": "gpt-test", "gen_ai.system": "openai",
                            "response": {"text": "hello", "tool_calls": []}},
            ):
                pass
        with recorder.span("run-b", kind="agent"):
            with recorder.span("step", kind="tool"):
                pass
    recorder.store.close()
    return TestClient(create_app(db))


def _trace_ids(client):
    return [t["trace_id"] for t in client.get("/api/traces").json()["traces"]]


def test_meta(client):
    data = client.get("/api/meta").json()
    assert "version" in data and "db" in data


def test_sessions_and_traces(client):
    sessions = client.get("/api/sessions").json()["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["trace_count"] == 2

    traces = client.get("/api/traces").json()["traces"]
    assert {t["name"] for t in traces} == {"run-a", "run-b"}
    run_a = next(t for t in traces if t["name"] == "run-a")
    assert run_a["stats"]["llm_count"] == 2
    assert run_a["stats"]["cost_usd"] == pytest.approx(0.02)


def test_trace_detail_and_spans(client):
    trace_id = _trace_ids(client)[0]
    detail = client.get(f"/api/traces/{trace_id}").json()
    assert detail["trace_id"] == trace_id
    spans = client.get(f"/api/traces/{trace_id}/spans").json()["spans"]
    assert len(spans) >= 1
    assert client.get(f"/api/traces/{trace_id[:8]}/spans").status_code == 200


def test_flamegraph_and_waterfall(client):
    trace_id = _trace_ids(client)[0]
    for weight in ("latency", "cost"):
        tree = client.get(f"/api/traces/{trace_id}/flamegraph?weight={weight}").json()
        assert tree["weight"] == weight
        assert tree["children"]
    assert client.get(f"/api/traces/{trace_id}/flamegraph?weight=bogus").status_code == 400
    rows = client.get(f"/api/traces/{trace_id}/waterfall").json()["rows"]
    assert rows and rows[0]["depth"] == 0


def test_waste_endpoint(client):
    traces = client.get("/api/traces").json()["traces"]
    run_a = next(t for t in traces if t["name"] == "run-a")
    report = client.get(f"/api/traces/{run_a['trace_id']}/waste").json()
    assert report["findings"]
    assert report["findings"][0]["rule_id"] == "duplicates"


def test_diff_endpoint(client):
    a, b = _trace_ids(client)
    result = client.get(f"/api/diff?a={a}&b={b}").json()
    assert "totals_a" in result and "matched" in result
    assert client.get("/api/diff?a=only").status_code == 400
    assert client.get("/api/diff?a=zzz&b=yyy").status_code == 404


def test_unknown_trace_404(client):
    assert client.get("/api/traces/nope/spans").status_code == 404


def test_index_and_static_served(client):
    home = client.get("/")
    assert home.status_code == 200
    assert "traceburn" in home.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_non_finite_costs_do_not_crash_endpoints(tmp_path):
    from traceburn.schema import Span, Trace, new_span_id, new_trace_id

    db = str(tmp_path / "nan.db")
    store = Store(db)
    trace_id = new_trace_id()
    store.insert_trace(Trace(trace_id=trace_id, name="run", start_ns=0))
    store.insert_span(
        Span(span_id=new_span_id(), trace_id=trace_id, name="chat x", kind="llm",
             start_ns=0, end_ns=1_000_000,
             attributes={"cost_usd": float("nan"), "request_hash": "h"})
    )
    store.close()
    c = TestClient(create_app(db))
    assert c.get(f"/api/traces/{trace_id}/flamegraph?weight=cost").status_code == 200
    assert c.get(f"/api/traces/{trace_id}/waste").status_code == 200
    assert c.get(f"/api/diff?a={trace_id}&b={trace_id}").status_code == 200


def test_bad_limit_falls_back_to_default(client):
    assert client.get("/api/traces?limit=abc").status_code == 200
    assert client.get("/api/sessions?limit=1e3").status_code == 200
    assert client.get("/api/traces?limit=99999999999999999999").status_code == 200
    assert client.get("/api/traces?limit=-5").status_code == 200


def test_foreign_host_rejected(client):
    response = client.get("/api/traces", headers={"host": "evil.example.com"})
    assert response.status_code == 400
