"""Patcher tests against the real anthropic SDK over a mock HTTP transport."""

import asyncio
import json

import httpx
import pytest

import traceburn

MESSAGE = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "model": "claude-test-4",
    "content": [{"type": "text", "text": "hello from claude"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 30,
        "output_tokens": 12,
        "cache_creation_input_tokens": 5,
        "cache_read_input_tokens": 25,
    },
}

TOOL_MESSAGE = {
    "id": "msg_02",
    "type": "message",
    "role": "assistant",
    "model": "claude-test-4",
    "content": [
        {"type": "text", "text": "checking"},
        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "NYC"}},
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {"input_tokens": 44, "output_tokens": 20},
}


def stream_body():
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_03",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-test-4",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 30,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 10,
                        "cache_creation_input_tokens": 0,
                    },
                },
            },
        ),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " world"}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 7},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


def make_handler(message_json=None, use_stream=False):
    def handler(request: httpx.Request) -> httpx.Response:
        if use_stream:
            return httpx.Response(
                200, text=stream_body(), headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(200, json=message_json or MESSAGE)

    return handler


def make_client(handler):
    from anthropic import Anthropic

    return Anthropic(
        api_key="test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def make_async_client(handler):
    from anthropic import AsyncAnthropic

    return AsyncAnthropic(
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


def test_messages_create_sync(instrumented):
    client = make_client(make_handler())
    result = client.messages.create(
        model="claude-test-4",
        max_tokens=100,
        system="be brief",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert result.content[0].text == "hello from claude"

    span = only_span(instrumented)
    attrs = span.attributes
    assert span.name == "chat claude-test-4"
    assert attrs["gen_ai.system"] == "anthropic"
    assert attrs["gen_ai.usage.input_tokens"] == 30
    assert attrs["cached_input_tokens"] == 25
    assert attrs["cache_write_tokens"] == 5
    assert attrs["gen_ai.usage.output_tokens"] == 12
    assert attrs["finish_reason"] == "end_turn"
    assert attrs["response"]["text"] == "hello from claude"
    assert attrs["request"]["system"] == "be brief"
    assert attrs["request_hash"]


def test_messages_create_async(instrumented):
    client = make_async_client(make_handler())

    async def call():
        return await client.messages.create(
            model="claude-test-4", max_tokens=100, messages=[{"role": "user", "content": "hi"}]
        )

    result = asyncio.run(call())
    assert result.content[0].text == "hello from claude"
    span = only_span(instrumented)
    assert span.attributes["gen_ai.usage.output_tokens"] == 12


def test_tool_use_captured(instrumented):
    client = make_client(make_handler(message_json=TOOL_MESSAGE))
    client.messages.create(
        model="claude-test-4",
        max_tokens=100,
        messages=[{"role": "user", "content": "weather?"}],
        tools=[{"name": "get_weather", "input_schema": {"type": "object"}}],
    )
    span = only_span(instrumented)
    calls = span.attributes["response"]["tool_calls"]
    assert calls == [{"id": "toolu_1", "name": "get_weather", "arguments": {"city": "NYC"}}]
    assert span.attributes["finish_reason"] == "tool_use"


def test_create_stream_true(instrumented):
    client = make_client(make_handler(use_stream=True))
    stream = client.messages.create(
        model="claude-test-4",
        max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    events = list(stream)
    assert any(getattr(e, "type", "") == "message_stop" for e in events)

    span = only_span(instrumented)
    attrs = span.attributes
    assert attrs["stream"] is True
    assert attrs["response"]["text"] == "Hello world"
    assert attrs["finish_reason"] == "end_turn"
    assert attrs["gen_ai.usage.input_tokens"] == 30
    assert attrs["cached_input_tokens"] == 10
    assert attrs["gen_ai.usage.output_tokens"] == 7
    assert "usage_estimated" not in attrs


def test_stream_helper_manager(instrumented):
    client = make_client(make_handler(use_stream=True))
    collected = []
    with client.messages.stream(
        model="claude-test-4", max_tokens=100, messages=[{"role": "user", "content": "hi"}]
    ) as stream:
        for text in stream.text_stream:
            collected.append(text)

    assert "".join(collected) == "Hello world"
    span = only_span(instrumented)
    attrs = span.attributes
    assert attrs["stream"] is True
    assert attrs["response"]["text"] == "Hello world"
    assert attrs["gen_ai.usage.output_tokens"] == 7
    assert span.end_ns is not None


def test_async_stream_helper_manager(instrumented):
    client = make_async_client(make_handler(use_stream=True))

    async def run():
        collected = []
        async with client.messages.stream(
            model="claude-test-4", max_tokens=100, messages=[{"role": "user", "content": "hi"}]
        ) as stream:
            async for text in stream.text_stream:
                collected.append(text)
        return "".join(collected)

    assert asyncio.run(run()) == "Hello world"
    span = only_span(instrumented)
    assert span.attributes["response"]["text"] == "Hello world"


def test_api_error_records_error_span(instrumented):
    def handler(request):
        return httpx.Response(
            500, json={"type": "error", "error": {"type": "api_error", "message": "boom"}}
        )

    client = make_client(handler)
    with pytest.raises(Exception):
        client.messages.create(
            model="claude-test-4", max_tokens=100, messages=[{"role": "user", "content": "hi"}]
        )
    span = only_span(instrumented)
    assert span.status == "error"


def tool_stream_body():
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_04",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-test-4",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 20, "output_tokens": 1},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_9", "name": "get_weather", "input": {}},
            },
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"city": '}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '"NYC"}'}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}, "usage": {"output_tokens": 15}},
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


def test_streamed_tool_use_captured(instrumented):
    def handler(request):
        return httpx.Response(
            200, text=tool_stream_body(), headers={"content-type": "text/event-stream"}
        )

    client = make_client(handler)
    stream = client.messages.create(
        model="claude-test-4", max_tokens=100,
        messages=[{"role": "user", "content": "weather?"}], stream=True,
    )
    for _ in stream:
        pass
    span = only_span(instrumented)
    calls = span.attributes["response"]["tool_calls"]
    assert calls == [{"id": "toolu_9", "name": "get_weather", "arguments": '{"city": "NYC"}'}]
    assert span.attributes["finish_reason"] == "tool_use"


def test_stream_helper_abandoned_before_any_event(instrumented):
    client = make_client(make_handler(use_stream=True))
    with client.messages.stream(
        model="claude-test-4", max_tokens=100, messages=[{"role": "user", "content": "hi"}]
    ):
        pass
    span = only_span(instrumented)
    assert span.attributes["usage_estimated"] is True
    assert span.attributes["gen_ai.usage.output_tokens"] == 0
    assert span.end_ns is not None
