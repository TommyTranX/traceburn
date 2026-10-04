"""Replay tests, including the zero-token proof.

The proof: record against a mock transport, then replay against a transport
that fails the test if ANY request reaches it. Replay must serve every call
from the store.
"""

import asyncio
import json

import httpx
import pytest

import traceburn
from traceburn.analyze.replay import ReplayMiss, replay


def chat_response(text, model="gpt-test"):
    return {
        "id": f"chatcmpl-{abs(hash(text)) % 10_000}",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 25, "completion_tokens": 7, "total_tokens": 32},
    }


def anthropic_response(text):
    return {
        "id": "msg_replay",
        "type": "message",
        "role": "assistant",
        "model": "claude-test-4",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 20, "output_tokens": 9},
    }


def echo_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    if request.url.path.endswith("/chat/completions"):
        prompt = body["messages"][-1]["content"]
        return httpx.Response(200, json=chat_response(f"answer to: {prompt}"))
    if request.url.path.endswith("/messages"):
        prompt = body["messages"][-1]["content"]
        return httpx.Response(200, json=anthropic_response(f"claude says: {prompt}"))
    return httpx.Response(404)


class PoisonedTransport(httpx.BaseTransport):
    """A transport that records any attempt to use the network."""

    def __init__(self):
        self.requests = 0

    def handle_request(self, request):
        self.requests += 1
        raise AssertionError("network request escaped replay")


@pytest.fixture
def recorded(tmp_path):
    """Record a three-call agent run; return (recorder, trace_id, client factory)."""
    from openai import OpenAI

    rec = traceburn.configure(db_path=str(tmp_path / "traces.db"))
    traceburn.install()
    client = OpenAI(
        api_key="test",
        http_client=httpx.Client(transport=httpx.MockTransport(echo_handler)),
    )
    with traceburn.span("agent", kind="agent") as root:
        for prompt in ("plan the task", "do the task", "plan the task"):
            client.chat.completions.create(
                model="gpt-test", messages=[{"role": "user", "content": prompt}]
            )
    yield rec, root.span.trace_id
    traceburn.uninstall()


def test_replay_serves_all_calls_with_zero_network(recorded):
    from openai import OpenAI

    rec, trace_id = recorded
    poison = PoisonedTransport()
    client = OpenAI(api_key="test", http_client=httpx.Client(transport=poison))

    with replay(trace_id=trace_id, store=rec.store) as replayer:
        assert replayer.recorded_calls == 3
        first = client.chat.completions.create(
            model="gpt-test", messages=[{"role": "user", "content": "plan the task"}]
        )
        second = client.chat.completions.create(
            model="gpt-test", messages=[{"role": "user", "content": "do the task"}]
        )
        third = client.chat.completions.create(
            model="gpt-test", messages=[{"role": "user", "content": "plan the task"}]
        )

    assert poison.requests == 0
    assert first.choices[0].message.content == "answer to: plan the task"
    assert second.choices[0].message.content == "answer to: do the task"
    assert third.choices[0].message.content == "answer to: plan the task"
    assert replayer.hits == 3 and replayer.misses == 0


def test_replay_is_deterministic_across_runs(recorded):
    from openai import OpenAI

    rec, trace_id = recorded
    outputs = []
    for _ in range(2):
        client = OpenAI(api_key="test", http_client=httpx.Client(transport=PoisonedTransport()))
        with replay(trace_id=trace_id, store=rec.store):
            result = client.chat.completions.create(
                model="gpt-test", messages=[{"role": "user", "content": "plan the task"}]
            )
            outputs.append(result.model_dump())
    assert outputs[0] == outputs[1]


def test_replay_miss_raises_with_reason(recorded):
    from openai import OpenAI

    rec, trace_id = recorded
    client = OpenAI(api_key="test", http_client=httpx.Client(transport=PoisonedTransport()))
    with replay(trace_id=trace_id, store=rec.store):
        with pytest.raises(ReplayMiss, match="no recorded response"):
            client.chat.completions.create(
                model="gpt-test", messages=[{"role": "user", "content": "something new"}]
            )


def test_replay_miss_passthrough_hits_network(recorded):
    from openai import OpenAI

    rec, trace_id = recorded
    client = OpenAI(
        api_key="test",
        http_client=httpx.Client(transport=httpx.MockTransport(echo_handler)),
    )
    with replay(trace_id=trace_id, store=rec.store, on_miss="passthrough") as replayer:
        result = client.chat.completions.create(
            model="gpt-test", messages=[{"role": "user", "content": "something new"}]
        )
    assert result.choices[0].message.content == "answer to: something new"
    assert replayer.misses == 1


def test_replay_exhausted_repeats_last_recording(recorded):
    from openai import OpenAI

    rec, trace_id = recorded
    client = OpenAI(api_key="test", http_client=httpx.Client(transport=PoisonedTransport()))
    with replay(trace_id=trace_id, store=rec.store):
        results = [
            client.chat.completions.create(
                model="gpt-test", messages=[{"role": "user", "content": "do the task"}]
            )
            for _ in range(3)
        ]
    assert all(r.choices[0].message.content == "answer to: do the task" for r in results)


def test_streaming_request_is_a_miss(recorded):
    from openai import OpenAI

    rec, trace_id = recorded
    client = OpenAI(api_key="test", http_client=httpx.Client(transport=PoisonedTransport()))
    with replay(trace_id=trace_id, store=rec.store):
        with pytest.raises(ReplayMiss, match="streaming"):
            client.chat.completions.create(
                model="gpt-test",
                messages=[{"role": "user", "content": "plan the task"}],
                stream=True,
            )


def _anthropic_http_client(transport):
    # Keep replay assertions identical across SDK releases with httpx/httpx2.
    from anthropic import _base_client
    sdk_http = getattr(_base_client, "httpx2", None) or _base_client.httpx

    class Adapter(sdk_http.BaseTransport):
        def handle_request(self, request):
            response = transport.handle_request(request)
            return sdk_http.Response(
                response.status_code, headers=dict(response.headers), content=response.read()
            )

    return sdk_http.Client(transport=Adapter())


def test_replay_anthropic(tmp_path):
    from anthropic import Anthropic

    rec = traceburn.configure(db_path=str(tmp_path / "traces.db"))
    traceburn.install()
    try:
        client = Anthropic(
            api_key="test",
            http_client=_anthropic_http_client(httpx.MockTransport(echo_handler)),
        )
        with traceburn.span("agent", kind="agent") as root:
            client.messages.create(
                model="claude-test-4",
                max_tokens=64,
                messages=[{"role": "user", "content": "summarize"}],
            )
    finally:
        traceburn.uninstall()

    poisoned = Anthropic(api_key="test", http_client=_anthropic_http_client(PoisonedTransport()))
    with replay(trace_id=root.span.trace_id, store=rec.store):
        result = poisoned.messages.create(
            model="claude-test-4",
            max_tokens=64,
            messages=[{"role": "user", "content": "summarize"}],
        )
    assert result.content[0].text == "claude says: summarize"


def test_replay_async_client(recorded):
    from openai import AsyncOpenAI

    rec, trace_id = recorded

    class AsyncPoison(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            raise AssertionError("network request escaped replay")

    client = AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(transport=AsyncPoison()))

    async def run():
        with replay(trace_id=trace_id, store=rec.store):
            return await client.chat.completions.create(
                model="gpt-test", messages=[{"role": "user", "content": "do the task"}]
            )

    result = asyncio.run(run())
    assert result.choices[0].message.content == "answer to: do the task"


def test_unpatch_after_replay_restores_normal_calls(recorded):
    from openai import OpenAI
    from openai.resources.chat import completions as chat_mod

    rec, trace_id = recorded
    with replay(trace_id=trace_id, store=rec.store):
        assert getattr(chat_mod.Completions.create, "__traceburn_replay__", False)
    assert not getattr(chat_mod.Completions.create, "__traceburn_replay__", False)
    client = OpenAI(
        api_key="test",
        http_client=httpx.Client(transport=httpx.MockTransport(echo_handler)),
    )
    result = client.chat.completions.create(
        model="gpt-test", messages=[{"role": "user", "content": "live again"}]
    )
    assert result.choices[0].message.content == "answer to: live again"


def test_anthropic_stream_helper_honors_on_miss(tmp_path):
    from anthropic import Anthropic

    rec = traceburn.configure(db_path=str(tmp_path / "traces.db"))
    traceburn.install()
    try:
        client = Anthropic(
            api_key="test",
            http_client=_anthropic_http_client(httpx.MockTransport(echo_handler)),
        )
        with traceburn.span("agent", kind="agent") as root:
            client.messages.create(
                model="claude-test-4", max_tokens=64,
                messages=[{"role": "user", "content": "summarize"}],
            )
    finally:
        traceburn.uninstall()

    poison = PoisonedTransport()
    poisoned = Anthropic(api_key="test", http_client=_anthropic_http_client(poison))
    with replay(trace_id=root.span.trace_id, store=rec.store):
        with pytest.raises(ReplayMiss, match="streaming"):
            with poisoned.messages.stream(
                model="claude-test-4", max_tokens=64,
                messages=[{"role": "user", "content": "summarize"}],
            ):
                pass
    assert poison.requests == 0


def test_nested_replay_contexts_do_not_break_outer(recorded):
    from openai import OpenAI

    rec, trace_id = recorded
    poison = PoisonedTransport()
    client = OpenAI(api_key="test", http_client=httpx.Client(transport=poison))

    with replay(trace_id=trace_id, store=rec.store) as outer:
        with replay(trace_id=trace_id, store=rec.store) as inner:
            client.chat.completions.create(
                model="gpt-test", messages=[{"role": "user", "content": "plan the task"}]
            )
        assert inner.hits == 1 and outer.hits == 0
        result = client.chat.completions.create(
            model="gpt-test", messages=[{"role": "user", "content": "do the task"}]
        )
        assert result.choices[0].message.content == "answer to: do the task"
        assert outer.hits == 1
    assert poison.requests == 0


def test_install_during_replay_leaves_no_stray_interception(recorded):
    from openai import OpenAI
    from openai.resources.chat import completions as chat_mod

    rec, trace_id = recorded
    with replay(trace_id=trace_id, store=rec.store):
        traceburn.install()
    traceburn.uninstall()

    client = OpenAI(
        api_key="test",
        http_client=httpx.Client(transport=httpx.MockTransport(echo_handler)),
    )
    result = client.chat.completions.create(
        model="gpt-test", messages=[{"role": "user", "content": "fresh call"}]
    )
    assert result.choices[0].message.content == "answer to: fresh call"
    # A stranded wrapper may remain, but with no active replayer it must be
    # a transparent passthrough; clean up for other tests.
    from traceburn.analyze import replay as replay_mod

    replay_mod._unwrap_all()
    assert not getattr(chat_mod.Completions.create, "__traceburn_replay__", False)


def test_raw_response_call_is_a_miss_under_replay(recorded):
    from openai import OpenAI

    rec, trace_id = recorded
    client = OpenAI(api_key="test", http_client=httpx.Client(transport=PoisonedTransport()))
    with replay(trace_id=trace_id, store=rec.store):
        with pytest.raises(ReplayMiss, match="raw-response"):
            client.chat.completions.with_raw_response.create(
                model="gpt-test", messages=[{"role": "user", "content": "plan the task"}]
            )


def test_miss_message_mentions_missing_payload(tmp_path, monkeypatch):
    from openai import OpenAI

    rec = traceburn.configure(db_path=str(tmp_path / "traces.db"))
    traceburn.install()
    try:
        from traceburn.instrument import _util as instrument_util

        monkeypatch.setattr(instrument_util, "raw_dump", lambda result: None)
        client = OpenAI(
            api_key="test",
            http_client=httpx.Client(transport=httpx.MockTransport(echo_handler)),
        )
        with traceburn.span("agent", kind="agent") as root:
            client.chat.completions.create(
                model="gpt-test", messages=[{"role": "user", "content": "plan"}]
            )
    finally:
        traceburn.uninstall()

    poisoned = OpenAI(api_key="test", http_client=httpx.Client(transport=PoisonedTransport()))
    with replay(trace_id=root.span.trace_id, store=rec.store) as replayer:
        assert replayer.recorded_calls == 0
        with pytest.raises(ReplayMiss, match="without a stored response payload"):
            poisoned.chat.completions.create(
                model="gpt-test", messages=[{"role": "user", "content": "plan"}]
            )
