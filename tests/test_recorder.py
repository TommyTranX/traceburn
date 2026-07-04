import asyncio
import threading

import pytest


def spans_by_name(store, trace_id):
    return {s.name: s for s in store.get_spans(trace_id)}


def test_nested_spans_share_trace_and_parent(recorder, store):
    with recorder.span("root", kind="agent") as root:
        with recorder.span("child", kind="tool") as child:
            with recorder.span("grandchild") as grandchild:
                pass

    trace_id = root.span.trace_id
    assert child.span.trace_id == trace_id
    assert grandchild.span.trace_id == trace_id

    got = spans_by_name(store, trace_id)
    assert got["root"].parent_id is None
    assert got["child"].parent_id == got["root"].span_id
    assert got["grandchild"].parent_id == got["child"].span_id

    trace = store.get_trace(trace_id)
    assert trace.name == "root"
    assert trace.end_ns >= got["root"].end_ns


def test_sibling_spans_do_not_nest(recorder, store):
    with recorder.span("root") as root:
        with recorder.span("first"):
            pass
        with recorder.span("second"):
            pass
    got = spans_by_name(store, root.span.trace_id)
    assert got["first"].parent_id == got["root"].span_id
    assert got["second"].parent_id == got["root"].span_id


def test_separate_roots_get_separate_traces(recorder, store):
    with recorder.span("one") as a:
        pass
    with recorder.span("two") as b:
        pass
    assert a.span.trace_id != b.span.trace_id
    assert store.get_trace(a.span.trace_id).name == "one"
    assert store.get_trace(b.span.trace_id).name == "two"


def test_session_groups_traces(recorder, store):
    with recorder.session("nightly") as sess:
        with recorder.span("run-a"):
            pass
        with recorder.span("run-b"):
            pass

    traces = store.list_traces(session_id=sess.session_id)
    assert sorted(t.name for t in traces) == ["run-a", "run-b"]
    stored = store.list_sessions()[0]
    assert stored.name == "nightly"
    assert stored.end_ns is not None


def test_exception_records_error_and_reraises(recorder, store):
    with pytest.raises(ValueError, match="boom"):
        with recorder.span("failing") as handle:
            raise ValueError("boom")

    got = store.get_spans(handle.span.trace_id)[0]
    assert got.status == "error"
    assert "ValueError: boom" in got.error
    assert got.end_ns is not None


def test_recorder_survives_store_failure(recorder, store):
    store.close()
    with recorder.span("after-close"):
        pass


def test_decorator_sync_and_name_default(recorder, store):
    @recorder.trace
    def do_work():
        return 42

    assert do_work() == 42
    traces = store.list_traces()
    assert traces[0].name.endswith("do_work")


def test_decorator_async(recorder, store):
    @recorder.trace("async-step", kind="tool")
    async def do_async():
        await asyncio.sleep(0)
        return "ok"

    assert asyncio.run(do_async()) == "ok"
    got = spans_by_name(store, store.list_traces()[0].trace_id)
    assert got["async-step"].kind == "tool"


def test_asyncio_tasks_inherit_parent_correctly(recorder, store):
    async def leaf(n):
        with recorder.span(f"leaf-{n}"):
            await asyncio.sleep(0.001)

    async def main():
        with recorder.span("root") as root:
            await asyncio.gather(leaf(1), leaf(2), leaf(3))
        return root.span.trace_id

    trace_id = asyncio.run(main())
    got = spans_by_name(store, trace_id)
    root_id = got["root"].span_id
    for n in (1, 2, 3):
        assert got[f"leaf-{n}"].parent_id == root_id


def test_threads_are_isolated(recorder, store):
    trace_ids = []

    def worker():
        with recorder.span("thread-root") as h:
            trace_ids.append(h.span.trace_id)

    with recorder.span("main-root") as main:
        t = threading.Thread(target=worker)
        t.start()
        t.join()

    assert trace_ids[0] != main.span.trace_id
    thread_span = store.get_spans(trace_ids[0])[0]
    assert thread_span.parent_id is None


def test_llm_span_gets_cost(recorder, store):
    with recorder.span(
        "call",
        kind="llm",
        attributes={
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-test",
            "gen_ai.usage.input_tokens": 1_000_000,
            "gen_ai.usage.output_tokens": 500_000,
        },
    ) as handle:
        pass

    got = store.get_spans(handle.span.trace_id)[0]
    assert got.attributes["cost_usd"] == pytest.approx(2.0 + 4.0)


def test_existing_cost_not_overwritten(recorder, store):
    with recorder.span(
        "call", kind="llm", attributes={"cost_usd": 0.123}
    ) as handle:
        pass
    got = store.get_spans(handle.span.trace_id)[0]
    assert got.attributes["cost_usd"] == 0.123


def test_unknown_model_gets_no_cost(recorder, store):
    with recorder.span(
        "call",
        kind="llm",
        attributes={
            "gen_ai.system": "someprovider",
            "gen_ai.request.model": "mystery-model",
            "gen_ai.usage.input_tokens": 100,
        },
    ) as handle:
        pass
    got = store.get_spans(handle.span.trace_id)[0]
    assert "cost_usd" not in got.attributes


def test_streaming_style_detach_and_late_end(recorder, store):
    with recorder.span("agent-loop") as root:
        handle = recorder.start_span("stream-call", kind="llm")
        handle.detach()
        with recorder.span("sibling"):
            pass
        handle.set_attribute("finish_reason", "stop")
        handle.end()

    got = spans_by_name(store, root.span.trace_id)
    assert got["sibling"].parent_id == got["agent-loop"].span_id
    assert got["stream-call"].parent_id == got["agent-loop"].span_id
    assert got["stream-call"].attributes["finish_reason"] == "stop"
    assert got["stream-call"].end_ns is not None


def test_end_is_idempotent(recorder, store):
    handle = recorder.start_span("once")
    first = handle.end()
    second = handle.end()
    assert first is second
    assert len(store.get_spans(handle.span.trace_id)) == 1


def test_module_level_api(tmp_path, monkeypatch):
    import traceburn

    monkeypatch.setenv("TRACEBURN_DB", str(tmp_path / "mod.db"))
    rec = traceburn.configure(db_path=str(tmp_path / "mod.db"))

    with traceburn.session("s"):
        with traceburn.span("root"):
            with traceburn.span("inner", kind="tool"):
                pass

    traces = rec.store.list_traces()
    assert len(traces) == 1
    spans = rec.store.get_spans(traces[0].trace_id)
    assert {s.name for s in spans} == {"root", "inner"}


def test_lazy_store_failure_never_raises(tmp_path, monkeypatch):
    from traceburn.recorder import Recorder

    monkeypatch.setenv("TRACEBURN_DB", str(tmp_path / "not-a-dir.db" / "traces.db"))
    (tmp_path / "not-a-dir.db").write_text("a file, not a directory")
    rec = Recorder()

    with rec.session("s"):
        with rec.span("root"):
            with rec.span("child", kind="tool"):
                pass


def test_session_exit_from_other_context_does_not_raise(recorder, store):
    import contextvars

    cm = recorder.session("cross-context")
    ctx = contextvars.copy_context()
    ctx.run(cm.__enter__)
    cm.__exit__(None, None, None)

    stored = store.list_sessions()[0]
    assert stored.name == "cross-context"
    assert stored.end_ns is not None


def test_invalid_kind_raises(recorder):
    with pytest.raises(ValueError, match="kind"):
        recorder.start_span("x", kind="LLM")


def test_generator_function_traced_over_full_iteration(recorder, store):
    @recorder.trace("gen-step", kind="tool")
    def numbers():
        yield 1
        yield 2

    assert list(numbers()) == [1, 2]
    got = spans_by_name(store, store.list_traces()[0].trace_id)
    assert got["gen-step"].kind == "tool"
    assert got["gen-step"].end_ns is not None


def test_async_generator_function_traced(recorder, store):
    @recorder.trace("agen-step", kind="retrieval")
    async def anumbers():
        yield 1
        yield 2

    async def consume():
        return [x async for x in anumbers()]

    assert asyncio.run(consume()) == [1, 2]
    got = spans_by_name(store, store.list_traces()[0].trace_id)
    assert got["agen-step"].status == "ok"


def test_backward_clock_never_yields_negative_duration(recorder, store):
    handle = recorder.start_span("clock-step")
    handle.end(end_ns=handle.span.start_ns - 5_000)
    got = store.get_spans(handle.span.trace_id)[0]
    assert got.end_ns == got.start_ns
