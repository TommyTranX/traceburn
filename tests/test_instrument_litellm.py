"""Patcher tests against the real litellm package.

No network: every call uses litellm's own ``mock_response`` kwarg, which
short-circuits before any provider transport runs. The patcher still sees
the same litellm.completion/acompletion call sites and real response
objects it would see in production.
"""

import asyncio

import pytest

import traceburn


@pytest.fixture
def instrumented(tmp_path):
    rec = traceburn.configure(db_path=str(tmp_path / "traces.db"))
    traceburn.install()
    yield rec
    traceburn.uninstall()


def only_span(rec):
    traces = rec.store.list_traces()
    assert len(traces) == 1
    spans = rec.store.get_spans(traces[0].trace_id)
    assert len(spans) == 1
    return spans[0]


def test_install_reports_litellm_patched(tmp_path):
    traceburn.configure(db_path=str(tmp_path / "t.db"))
    try:
        patched = traceburn.install()
        assert "litellm" in patched
        assert traceburn.install() == []
    finally:
        traceburn.uninstall()


def test_completion_sync_openai_routed(instrumented):
    import litellm

    result = litellm.completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="hello there",
    )
    assert result.choices[0].message.content == "hello there"

    span = only_span(instrumented)
    attrs = span.attributes
    assert span.kind == "llm"
    assert attrs["gen_ai.system"] == "openai"
    assert attrs["gen_ai.request.model"] == "gpt-4o"
    assert attrs["gen_ai.response.model"] == "gpt-4o"
    assert attrs["gen_ai.usage.input_tokens"] > 0
    assert attrs["gen_ai.usage.output_tokens"] > 0
    assert attrs["response"]["text"] == "hello there"
    assert attrs["request"]["messages"] == [{"role": "user", "content": "hi"}]
    assert attrs["request_hash"]
    assert attrs["stream"] is False


def test_completion_sync_anthropic_routed(instrumented):
    import litellm

    result = litellm.completion(
        model="anthropic/claude-sonnet-5-20250929",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="hi from claude",
    )
    assert result.choices[0].message.content == "hi from claude"

    span = only_span(instrumented)
    attrs = span.attributes
    assert attrs["gen_ai.system"] == "anthropic"
    assert attrs["gen_ai.request.model"] == "claude-sonnet-5-20250929"


def test_completion_async(instrumented):
    import litellm

    async def call():
        return await litellm.acompletion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="async hello",
        )

    result = asyncio.run(call())
    assert result.choices[0].message.content == "async hello"
    span = only_span(instrumented)
    assert span.attributes["response"]["text"] == "async hello"


def test_unclassifiable_model_falls_back_to_litellm_provider(instrumented):
    # litellm itself refuses to call a model it can't map to a provider, so
    # this exercises the same fallback path as a real error: the span still
    # records, under the "litellm" provider label, and comes back marked
    # as an error rather than being lost.
    import litellm

    with pytest.raises(Exception):
        litellm.completion(
            model="totally-made-up-model-xyz",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="whatever",
        )
    span = only_span(instrumented)
    assert span.attributes["gen_ai.system"] == "litellm"
    assert span.status == "error"


def test_streaming_captures_text(instrumented):
    import litellm

    stream = litellm.completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="hello streamed",
        stream=True,
    )
    text = "".join(
        chunk.choices[0].delta.content or ""
        for chunk in stream
        if chunk.choices and chunk.choices[0].delta
    )
    assert "hello" in text.lower()

    span = only_span(instrumented)
    attrs = span.attributes
    assert attrs["stream"] is True
    assert attrs["response"]["text"]
    assert span.end_ns is not None


def test_error_records_error_span(instrumented):
    import litellm

    with pytest.raises(Exception):
        litellm.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="litellm.RateLimitError",
        )
    span = only_span(instrumented)
    assert span.status == "error"
    assert span.end_ns is not None


def test_capture_failure_never_breaks_call(instrumented, monkeypatch):
    from traceburn.instrument import litellm as litellm_patcher

    monkeypatch.setattr(litellm_patcher, "_start_llm_span", lambda *a, **k: 1 / 0)
    import litellm

    result = litellm.completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="still works",
    )
    assert result.choices[0].message.content == "still works"


def test_uninstall_restores_originals(tmp_path):
    import litellm

    traceburn.configure(db_path=str(tmp_path / "t.db"))
    traceburn.install()
    assert hasattr(litellm.completion, "__traceburn_original__")
    traceburn.uninstall()
    assert not hasattr(litellm.completion, "__traceburn_original__")


def test_positional_model_and_messages(instrumented):
    import litellm

    result = litellm.completion(
        "gpt-4o", [{"role": "user", "content": "hi"}], mock_response="positional works"
    )
    assert result.choices[0].message.content == "positional works"
    span = only_span(instrumented)
    assert span.attributes["gen_ai.request.model"] == "gpt-4o"
    assert span.attributes["request"]["messages"] == [{"role": "user", "content": "hi"}]
