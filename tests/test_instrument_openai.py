"""Patcher tests against the real openai SDK over a mock HTTP transport.

No network: every request is served by httpx.MockTransport. The SDK still
parses real response models, so these tests exercise the exact objects the
patcher sees in production.
"""

import asyncio
import json

import httpx
import pytest

import traceburn

CHAT_COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-test-2026-01-15",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hello there"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 120,
        "completion_tokens": 8,
        "total_tokens": 128,
        "prompt_tokens_details": {"cached_tokens": 100},
    },
}

CHAT_TOOL_CALL = {
    "id": "chatcmpl-2",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-test",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "NYC"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52},
}

RESPONSES_RESPONSE = {
    "id": "resp_1",
    "object": "response",
    "created_at": 1,
    "model": "gpt-test",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "hi from responses", "annotations": []}],
        }
    ],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "usage": {
        "input_tokens": 50,
        "output_tokens": 5,
        "total_tokens": 55,
        "input_tokens_details": {"cached_tokens": 20},
        "output_tokens_details": {"reasoning_tokens": 0},
    },
}


def sse(*events):
    body = ""
    for event in events:
        body += f"data: {json.dumps(event)}\n\n"
    body += "data: [DONE]\n\n"
    return body


def chat_chunk(content=None, finish=None, usage=None):
    chunk = {
        "id": "chatcmpl-s",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-test",
        "choices": [],
    }
    if content is not None or finish is not None:
        chunk["choices"] = [
            {"index": 0, "delta": {"content": content} if content else {}, "finish_reason": finish}
        ]
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def make_handler(chat_json=None, responses_json=None, chat_sse=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            if chat_sse is not None:
                return httpx.Response(
                    200, text=chat_sse, headers={"content-type": "text/event-stream"}
                )
            return httpx.Response(200, json=chat_json or CHAT_COMPLETION)
        if request.url.path.endswith("/responses"):
            return httpx.Response(200, json=responses_json or RESPONSES_RESPONSE)
        return httpx.Response(404, json={"error": "unexpected path"})

    return handler


def make_client(handler):
    from openai import OpenAI

    return OpenAI(
        api_key="test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def make_async_client(handler):
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


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


def test_install_reports_patched_clients(tmp_path):
    traceburn.configure(db_path=str(tmp_path / "t.db"))
    try:
        patched = traceburn.install()
        assert "openai" in patched
        assert "anthropic" in patched
        assert traceburn.install() == []
    finally:
        traceburn.uninstall()


def test_chat_completion_sync(instrumented):
    client = make_client(make_handler())
    result = client.chat.completions.create(
        model="gpt-test",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.2,
    )
    assert result.choices[0].message.content == "hello there"

    span = only_span(instrumented)
    attrs = span.attributes
    assert span.kind == "llm"
    assert span.name == "chat gpt-test"
    assert attrs["gen_ai.system"] == "openai"
    assert attrs["gen_ai.response.model"] == "gpt-test-2026-01-15"
    assert attrs["gen_ai.usage.input_tokens"] == 20
    assert attrs["cached_input_tokens"] == 100
    assert attrs["gen_ai.usage.output_tokens"] == 8
    assert attrs["finish_reason"] == "stop"
    assert attrs["response"]["text"] == "hello there"
    assert attrs["request"]["messages"] == [{"role": "user", "content": "hi"}]
    assert attrs["request"]["temperature"] == 0.2
    assert attrs["request_hash"]
    assert attrs["stream"] is False


def test_chat_completion_async(instrumented):
    client = make_async_client(make_handler())

    async def call():
        return await client.chat.completions.create(
            model="gpt-test", messages=[{"role": "user", "content": "hi"}]
        )

    result = asyncio.run(call())
    assert result.choices[0].message.content == "hello there"
    span = only_span(instrumented)
    assert span.attributes["gen_ai.usage.output_tokens"] == 8


def test_tool_calls_captured(instrumented):
    client = make_client(make_handler(chat_json=CHAT_TOOL_CALL))
    client.chat.completions.create(
        model="gpt-test",
        messages=[{"role": "user", "content": "weather?"}],
        tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
    )
    span = only_span(instrumented)
    calls = span.attributes["response"]["tool_calls"]
    assert calls == [{"id": "call_1", "name": "get_weather", "arguments": '{"city": "NYC"}'}]
    assert span.attributes["finish_reason"] == "tool_calls"


def test_chat_stream_with_usage(instrumented):
    body = sse(
        chat_chunk(content="Hel"),
        chat_chunk(content="lo"),
        chat_chunk(finish="stop"),
        chat_chunk(usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}),
    )
    client = make_client(make_handler(chat_sse=body))
    stream = client.chat.completions.create(
        model="gpt-test",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    text = "".join(
        chunk.choices[0].delta.content or ""
        for chunk in stream
        if chunk.choices and chunk.choices[0].delta
    )
    assert text == "Hello"

    span = only_span(instrumented)
    attrs = span.attributes
    assert attrs["stream"] is True
    assert attrs["response"]["text"] == "Hello"
    assert attrs["finish_reason"] == "stop"
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["gen_ai.usage.output_tokens"] == 2
    assert "usage_estimated" not in attrs
    assert span.end_ns is not None


def test_chat_stream_without_usage_estimates(instrumented):
    body = sse(chat_chunk(content="Hello world"), chat_chunk(finish="stop"))
    client = make_client(make_handler(chat_sse=body))
    stream = client.chat.completions.create(
        model="gpt-test", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    for _ in stream:
        pass
    span = only_span(instrumented)
    assert span.attributes["usage_estimated"] is True
    assert span.attributes["gen_ai.usage.output_tokens"] >= 1


def test_chat_stream_context_manager_early_exit(instrumented):
    body = sse(chat_chunk(content="Hel"), chat_chunk(content="lo"), chat_chunk(finish="stop"))
    client = make_client(make_handler(chat_sse=body))
    with client.chat.completions.create(
        model="gpt-test", messages=[{"role": "user", "content": "hi"}], stream=True
    ) as stream:
        next(iter(stream))
    span = only_span(instrumented)
    assert span.end_ns is not None


def test_streaming_span_does_not_capture_siblings(instrumented):
    body = sse(chat_chunk(content="x"), chat_chunk(finish="stop"))
    client = make_client(make_handler(chat_sse=body))
    with traceburn.span("agent-loop"):
        stream = client.chat.completions.create(
            model="gpt-test", messages=[{"role": "user", "content": "hi"}], stream=True
        )
        with traceburn.span("tool-step", kind="tool"):
            pass
        for _ in stream:
            pass

    traces = instrumented.store.list_traces()
    assert len(traces) == 1
    spans = {s.name: s for s in instrumented.store.get_spans(traces[0].trace_id)}
    root_id = spans["agent-loop"].span_id
    assert spans["tool-step"].parent_id == root_id
    assert spans["chat gpt-test"].parent_id == root_id


def test_responses_api_sync(instrumented):
    client = make_client(make_handler())
    result = client.responses.create(model="gpt-test", input="say hi")
    assert "responses" in result.output_text or result.output_text

    span = only_span(instrumented)
    attrs = span.attributes
    assert attrs["request"]["endpoint"] == "responses"
    assert attrs["gen_ai.usage.input_tokens"] == 30
    assert attrs["cached_input_tokens"] == 20
    assert attrs["gen_ai.usage.output_tokens"] == 5
    assert attrs["response"]["text"] == "hi from responses"


def test_api_error_records_error_span(instrumented):
    def handler(request):
        return httpx.Response(500, json={"error": {"message": "boom"}})

    client = make_client(handler)
    with pytest.raises(Exception):
        client.chat.completions.create(
            model="gpt-test", messages=[{"role": "user", "content": "hi"}]
        )
    span = only_span(instrumented)
    assert span.status == "error"
    assert span.end_ns is not None


def test_same_request_same_hash(instrumented):
    client = make_client(make_handler())
    for _ in range(2):
        client.chat.completions.create(
            model="gpt-test", messages=[{"role": "user", "content": "hi"}]
        )
    client.chat.completions.create(
        model="gpt-test", messages=[{"role": "user", "content": "different"}]
    )
    traces = instrumented.store.list_traces()
    hashes = [
        instrumented.store.get_spans(t.trace_id)[0].attributes["request_hash"]
        for t in traces
    ]
    assert len(set(hashes)) == 2


def test_capture_failure_never_breaks_call(instrumented, monkeypatch):
    from traceburn.instrument import openai as openai_patcher

    monkeypatch.setattr(
        openai_patcher, "_start_llm_span", lambda *a, **k: 1 / 0
    )
    client = make_client(make_handler())
    result = client.chat.completions.create(
        model="gpt-test", messages=[{"role": "user", "content": "hi"}]
    )
    assert result.choices[0].message.content == "hello there"


def test_uninstall_restores_originals(tmp_path):
    from openai.resources.chat import completions as chat_completions

    traceburn.configure(db_path=str(tmp_path / "t.db"))
    traceburn.install()
    assert hasattr(chat_completions.Completions.create, "__traceburn_original__")
    traceburn.uninstall()
    assert not hasattr(chat_completions.Completions.create, "__traceburn_original__")


def test_async_stream_close_finalizes_span(instrumented):
    body = sse(chat_chunk(content="Hel"), chat_chunk(content="lo"), chat_chunk(finish="stop"))
    client = make_async_client(make_handler(chat_sse=body))

    async def run():
        stream = await client.chat.completions.create(
            model="gpt-test", messages=[{"role": "user", "content": "hi"}], stream=True
        )
        await stream.__anext__()
        await stream.close()

    asyncio.run(run())
    span = only_span(instrumented)
    assert span.end_ns is not None
    assert span.attributes["usage_estimated"] is True


def test_async_stream_aclose_does_not_raise(instrumented):
    body = sse(chat_chunk(content="x"), chat_chunk(finish="stop"))
    client = make_async_client(make_handler(chat_sse=body))

    async def run():
        stream = await client.chat.completions.create(
            model="gpt-test", messages=[{"role": "user", "content": "hi"}], stream=True
        )
        await stream.aclose()

    asyncio.run(run())
    span = only_span(instrumented)
    assert span.end_ns is not None


def test_abandoned_stream_finalizes_at_gc(instrumented):
    import gc

    body = sse(chat_chunk(content="x"), chat_chunk(finish="stop"))
    client = make_client(make_handler(chat_sse=body))
    stream = client.chat.completions.create(
        model="gpt-test", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    next(iter(stream))
    del stream
    gc.collect()
    span = only_span(instrumented)
    assert span.end_ns is not None


def test_raw_response_calls_pass_through_untraced(instrumented):
    client = make_client(make_handler())
    raw = client.chat.completions.with_raw_response.create(
        model="gpt-test", messages=[{"role": "user", "content": "hi"}]
    )
    assert raw.parse().choices[0].message.content == "hello there"
    assert instrumented.store.list_traces() == []


def test_responses_collector_estimates_from_list_input():
    from traceburn.instrument.openai import ResponsesStreamCollector

    collector = ResponsesStreamCollector(
        {"input": [{"role": "user", "content": "hello world, long enough text"}]}
    )
    assert collector._request_text


def test_chat_stream_collector_ignores_secondary_choices():
    from types import SimpleNamespace as NS

    from traceburn.instrument.openai import ChatStreamCollector

    collector = ChatStreamCollector({"messages": []})
    collector.add(
        NS(model="m", id="i", usage=None, choices=[
            NS(index=0, finish_reason=None, delta=NS(content="keep", tool_calls=None)),
            NS(index=1, finish_reason=None, delta=NS(content="drop", tool_calls=None)),
        ])
    )
    assert "".join(collector.text) == "keep"
