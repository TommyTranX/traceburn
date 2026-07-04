import sqlite3

import pytest

from traceburn.schema import Session, Span, Trace, new_span_id, new_trace_id
from traceburn.store import SCHEMA_VERSION, Store


def make_span(trace_id, **kwargs):
    defaults = dict(
        span_id=new_span_id(),
        trace_id=trace_id,
        name="step",
        kind="custom",
        start_ns=1_000,
        end_ns=2_000,
    )
    defaults.update(kwargs)
    return Span(**defaults)


def test_span_roundtrip(store):
    trace_id = new_trace_id()
    store.insert_trace(Trace(trace_id=trace_id, name="run", start_ns=1_000))
    span = make_span(
        trace_id,
        attributes={
            "gen_ai.system": "openai",
            "request": {"messages": [{"role": "user", "content": "hi"}]},
            "response": {"text": "hello"},
            "request_hash": "abc123",
        },
        status="ok",
    )
    store.insert_span(span)

    loaded = store.get_spans(trace_id)
    assert len(loaded) == 1
    got = loaded[0]
    assert got.span_id == span.span_id
    assert got.name == "step"
    assert got.attributes["request"] == {"messages": [{"role": "user", "content": "hi"}]}
    assert got.attributes["response"] == {"text": "hello"}
    assert got.attributes["gen_ai.system"] == "openai"


def test_blob_dedup(store):
    trace_id = new_trace_id()
    store.insert_trace(Trace(trace_id=trace_id, name="run", start_ns=0))
    payload = {"messages": [{"role": "system", "content": "x" * 500}]}
    for _ in range(3):
        store.insert_span(make_span(trace_id, attributes={"request": payload}))

    conn = sqlite3.connect(store.path)
    (blob_count,) = conn.execute("SELECT COUNT(*) FROM blobs").fetchone()
    (span_count,) = conn.execute("SELECT COUNT(*) FROM spans").fetchone()
    conn.close()
    assert span_count == 3
    assert blob_count == 1


def test_unhydrated_read_keeps_blob_ref(store):
    trace_id = new_trace_id()
    store.insert_trace(Trace(trace_id=trace_id, name="run", start_ns=0))
    store.insert_span(make_span(trace_id, attributes={"request": {"a": 1}}))
    got = store.get_spans(trace_id, hydrate=False)[0]
    assert "$blob" in got.attributes["request"]
    blob = store.get_blob_json(got.attributes["request"]["$blob"])
    assert blob == {"a": 1}


def test_request_hash_indexed_lookup(store):
    trace_id = new_trace_id()
    store.insert_trace(Trace(trace_id=trace_id, name="run", start_ns=0))
    store.insert_span(make_span(trace_id, attributes={"request_hash": "h1"}))
    store.insert_span(make_span(trace_id, attributes={"request_hash": "h1"}))
    store.insert_span(make_span(trace_id, attributes={"request_hash": "h2"}))
    assert len(store.find_spans_by_request_hash("h1")) == 2
    assert len(store.find_spans_by_request_hash("h2")) == 1
    assert store.find_spans_by_request_hash("missing") == []


def test_sessions_and_traces_listing(store):
    store.insert_session(Session(session_id="s1", name="nightly", start_ns=10))
    t1 = Trace(trace_id=new_trace_id(), name="a", session_id="s1", start_ns=10)
    t2 = Trace(trace_id=new_trace_id(), name="b", session_id="s1", start_ns=20)
    store.insert_trace(t1)
    store.insert_trace(t2)
    store.extend_trace_end(t1.trace_id, 99)

    sessions = store.list_sessions()
    assert [s.name for s in sessions] == ["nightly"]
    traces = store.list_traces(session_id="s1")
    assert [t.name for t in traces] == ["b", "a"]
    assert store.get_trace(t1.trace_id).end_ns == 99


def test_extend_trace_end_never_shrinks(store):
    trace_id = new_trace_id()
    store.insert_trace(Trace(trace_id=trace_id, name="run", start_ns=0))
    store.extend_trace_end(trace_id, 500)
    store.extend_trace_end(trace_id, 300)
    assert store.get_trace(trace_id).end_ns == 500


def test_concurrent_reader_sees_writes(store, tmp_path):
    trace_id = new_trace_id()
    store.insert_trace(Trace(trace_id=trace_id, name="run", start_ns=0))
    store.insert_span(make_span(trace_id))

    reader = Store(tmp_path / "traces.db")
    try:
        assert len(reader.get_spans(trace_id)) == 1
        store.insert_span(make_span(trace_id))
        assert len(reader.get_spans(trace_id)) == 2
    finally:
        reader.close()


def test_schema_version_written(store):
    conn = sqlite3.connect(store.path)
    (version,) = conn.execute(
        "SELECT value FROM meta WHERE key = 'schema_version'"
    ).fetchone()
    conn.close()
    assert int(version) == SCHEMA_VERSION


def test_newer_schema_rejected(tmp_path):
    path = tmp_path / "future.db"
    s = Store(path)
    s.close()
    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE meta SET value = ? WHERE key = 'schema_version'",
        (str(SCHEMA_VERSION + 1),),
    )
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="schema version"):
        Store(path)


def test_env_var_default_path(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACEBURN_DB", str(tmp_path / "custom.db"))
    s = Store()
    try:
        assert s.path == str(tmp_path / "custom.db")
        assert (tmp_path / "custom.db").exists()
    finally:
        s.close()


def test_unserializable_attributes_degrade_but_span_survives(store):
    trace_id = new_trace_id()
    store.insert_trace(Trace(trace_id=trace_id, name="run", start_ns=0))
    circular = {}
    circular["self"] = circular
    span = make_span(
        trace_id,
        attributes={
            "tuple_keys": {("a", "b"): 1},
            "an_object": object(),
            "score": float("nan"),
            "rate": float("inf"),
            "raw": b"\xff\xfebinary",
            "loop": circular,
            "int_keys": {1: "one"},
        },
    )
    store.insert_span(span)

    got = store.get_spans(trace_id)[0]
    assert got.attributes["score"] == "NaN"
    assert got.attributes["rate"] == "Infinity"
    assert got.attributes["loop"] == {"self": "<circular>"}
    assert got.attributes["int_keys"] == {"1": "one"}
    assert "�" in got.attributes["raw"]
    assert "('a', 'b')" in got.attributes["tuple_keys"]


def test_stored_attributes_are_valid_json_for_sqlite(store):
    trace_id = new_trace_id()
    store.insert_trace(Trace(trace_id=trace_id, name="run", start_ns=0))
    store.insert_span(make_span(trace_id, attributes={"score": float("nan")}))
    conn = sqlite3.connect(store.path)
    (valid,) = conn.execute("SELECT json_valid(attributes) FROM spans").fetchone()
    conn.close()
    assert valid == 1


def test_find_traces_escapes_like_wildcards(store):
    trace_id = "abc1234" + "0" * 25
    store.insert_trace(Trace(trace_id=trace_id, name="run", start_ns=0))
    assert len(store.find_traces("abc1")) == 1
    assert store.find_traces("ab_1") == []
    assert store.find_traces("%") == []
    assert store.find_traces("") == []


def test_sets_sanitize_deterministically(store):
    from traceburn.store import _sanitize

    assert _sanitize({"stop": {"b", "a", "c"}}) == {"stop": ["a", "b", "c"]}
