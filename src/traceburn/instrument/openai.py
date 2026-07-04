"""Patcher for the openai client: Chat Completions and Responses APIs.

Covers sync and async, streaming and non-streaming, and tool calls, for
clients created before or after ``install()``. The openai SDK is imported
lazily so this module is safe to import when the package is absent.

Known gaps, deliberate for now: the ``parse()`` conveniences and
``with_raw_response`` calls are not traced (raw-response calls are detected
and passed through untraced rather than recorded wrong). The
``chat.completions.stream()`` and ``responses.stream()`` helper managers
route through the patched ``create(stream=True)`` internally, so they are
traced; a helper abandoned before exhaustion finalizes its span at garbage
collection with estimated usage. Streaming calls report exact usage when
the request includes ``stream_options={"include_usage": True}`` (Chat
Completions) or reaches a terminal event (Responses); otherwise token
counts are estimated and the span is marked ``usage_estimated``. When a
request asks for multiple choices (``n > 1``), the recorded response text
and tool calls come from choice 0.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any

from ..recorder import get_recorder
from ._util import (
    TracedAsyncStream,
    TracedSyncStream,
    clean_params,
    estimate_tokens,
    patch_method,
    request_hash,
    unpatch_method,
)

logger = logging.getLogger("traceburn")

CHAT_PARAMS = (
    "model",
    "messages",
    "tools",
    "tool_choice",
    "functions",
    "function_call",
    "temperature",
    "top_p",
    "n",
    "stop",
    "max_tokens",
    "max_completion_tokens",
    "response_format",
    "seed",
    "logit_bias",
    "reasoning_effort",
)

RESPONSES_PARAMS = (
    "model",
    "input",
    "instructions",
    "tools",
    "tool_choice",
    "temperature",
    "top_p",
    "max_output_tokens",
    "reasoning",
    "text",
    "previous_response_id",
    "conversation",
    "prompt",
)

_RESPONSES_TERMINAL_EVENTS = (
    "response.completed",
    "response.incomplete",
    "response.failed",
)


def is_available() -> bool:
    return importlib.util.find_spec("openai") is not None


def _messages_text(messages: Any) -> str:
    parts = []
    try:
        for message in messages or []:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        parts.append(block["text"])
    except Exception:
        pass
    return "\n".join(parts)


def _start_llm_span(endpoint: str, params: tuple[str, ...], kwargs: dict[str, Any]):
    recorder = get_recorder()
    request = clean_params(kwargs, params)
    request["endpoint"] = endpoint
    model = kwargs.get("model") or "unknown"
    attributes = {
        "gen_ai.system": "openai",
        "gen_ai.request.model": model,
        "stream": bool(kwargs.get("stream")),
        "request": request,
        "request_hash": request_hash(request),
    }
    return recorder.start_span(f"chat {model}", kind="llm", attributes=attributes)


def _set_chat_usage(handle, usage: Any) -> None:
    if usage is None:
        return
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    details = getattr(usage, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
    handle.set_attributes(
        {
            "gen_ai.usage.input_tokens": prompt - cached,
            "gen_ai.usage.output_tokens": completion,
            "cached_input_tokens": cached,
        }
    )


def _set_responses_usage(handle, usage: Any) -> None:
    if usage is None:
        return
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    details = getattr(usage, "input_tokens_details", None)
    cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
    handle.set_attributes(
        {
            "gen_ai.usage.input_tokens": input_tokens - cached,
            "gen_ai.usage.output_tokens": output_tokens,
            "cached_input_tokens": cached,
        }
    )


def _capture_chat_response(handle, result: Any) -> None:
    choice = result.choices[0] if getattr(result, "choices", None) else None
    message = getattr(choice, "message", None)
    tool_calls = []
    for call in getattr(message, "tool_calls", None) or []:
        function = getattr(call, "function", None)
        tool_calls.append(
            {
                "id": getattr(call, "id", None),
                "name": getattr(function, "name", None),
                "arguments": getattr(function, "arguments", None),
            }
        )
    handle.set_attributes(
        {
            "gen_ai.response.model": getattr(result, "model", None),
            "gen_ai.response.id": getattr(result, "id", None),
            "finish_reason": getattr(choice, "finish_reason", None),
            "response": {
                "text": getattr(message, "content", None),
                "tool_calls": tool_calls,
            },
        }
    )
    _set_chat_usage(handle, getattr(result, "usage", None))


def _capture_responses_response(handle, result: Any) -> None:
    tool_calls = []
    for item in getattr(result, "output", None) or []:
        if getattr(item, "type", None) == "function_call":
            tool_calls.append(
                {
                    "id": getattr(item, "call_id", None),
                    "name": getattr(item, "name", None),
                    "arguments": getattr(item, "arguments", None),
                }
            )
    handle.set_attributes(
        {
            "gen_ai.response.model": getattr(result, "model", None),
            "gen_ai.response.id": getattr(result, "id", None),
            "finish_reason": getattr(result, "status", None),
            "response": {
                "text": getattr(result, "output_text", None) or None,
                "tool_calls": tool_calls,
            },
        }
    )
    _set_responses_usage(handle, getattr(result, "usage", None))


class ChatStreamCollector:
    """Accumulates chat-completion chunks into final span attributes."""

    def __init__(self, kwargs: dict[str, Any]):
        self._request_text = _messages_text(kwargs.get("messages"))
        self.model = None
        self.response_id = None
        self.finish_reason = None
        self.usage = None
        self.text = []
        self.tool_calls: dict[int, dict[str, Any]] = {}

    def add(self, chunk: Any) -> None:
        self.model = getattr(chunk, "model", None) or self.model
        self.response_id = getattr(chunk, "id", None) or self.response_id
        if getattr(chunk, "usage", None) is not None:
            self.usage = chunk.usage
        for choice in getattr(chunk, "choices", None) or []:
            # Multi-choice (n > 1) responses record choice 0, matching the
            # non-streaming capture.
            if (getattr(choice, "index", 0) or 0) != 0:
                continue
            if getattr(choice, "finish_reason", None):
                self.finish_reason = choice.finish_reason
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            if getattr(delta, "content", None):
                self.text.append(delta.content)
            for call in getattr(delta, "tool_calls", None) or []:
                index = getattr(call, "index", 0) or 0
                slot = self.tool_calls.setdefault(
                    index, {"id": None, "name": None, "arguments": ""}
                )
                slot["id"] = getattr(call, "id", None) or slot["id"]
                function = getattr(call, "function", None)
                if function is not None:
                    slot["name"] = getattr(function, "name", None) or slot["name"]
                    slot["arguments"] += getattr(function, "arguments", None) or ""

    def finalize(self, handle, error: str | None) -> None:
        text = "".join(self.text)
        handle.set_attributes(
            {
                "gen_ai.response.model": self.model,
                "gen_ai.response.id": self.response_id,
                "finish_reason": self.finish_reason,
                "response": {
                    "text": text or None,
                    "tool_calls": list(self.tool_calls.values()),
                },
            }
        )
        if self.usage is not None:
            _set_chat_usage(handle, self.usage)
        else:
            handle.set_attributes(
                {
                    "gen_ai.usage.input_tokens": estimate_tokens(self._request_text),
                    "gen_ai.usage.output_tokens": estimate_tokens(text),
                    "usage_estimated": True,
                }
            )
        handle.end(error=error)


class ResponsesStreamCollector:
    """Accumulates Responses API stream events into final span attributes."""

    def __init__(self, kwargs: dict[str, Any]):
        raw_input = kwargs.get("input")
        if isinstance(raw_input, str):
            self._request_text = raw_input
        else:
            self._request_text = _messages_text(raw_input)
        instructions = kwargs.get("instructions")
        if isinstance(instructions, str):
            self._request_text = f"{instructions}\n{self._request_text}"
        self.completed_response = None
        self.text = []

    def add(self, event: Any) -> None:
        event_type = getattr(event, "type", None)
        if event_type == "response.output_text.delta":
            delta = getattr(event, "delta", None)
            if delta:
                self.text.append(delta)
        elif event_type in _RESPONSES_TERMINAL_EVENTS:
            self.completed_response = getattr(event, "response", None)

    def finalize(self, handle, error: str | None) -> None:
        if self.completed_response is not None:
            _capture_responses_response(handle, self.completed_response)
        else:
            handle.set_attributes(
                {
                    "response": {"text": "".join(self.text) or None, "tool_calls": []},
                    "gen_ai.usage.input_tokens": estimate_tokens(self._request_text),
                    "gen_ai.usage.output_tokens": estimate_tokens("".join(self.text)),
                    "usage_estimated": True,
                }
            )
        handle.end(error=error)


def _is_raw_response_call(kwargs: dict[str, Any]) -> bool:
    # with_raw_response / with_streaming_response calls return response
    # wrappers, not completions; tracing them records nonsense, so pass
    # them through untraced.
    try:
        headers = kwargs.get("extra_headers") or {}
        return bool(headers.get("X-Stainless-Raw-Response"))
    except Exception:
        return False


def _make_wrapper(
    endpoint: str,
    params: tuple[str, ...],
    capture,
    collector_cls,
    is_async: bool,
):
    def factory(original):
        if is_async:

            async def wrapper(self, *args, **kwargs):
                if _is_raw_response_call(kwargs):
                    return await original(self, *args, **kwargs)
                try:
                    handle = _start_llm_span(endpoint, params, kwargs)
                except Exception:
                    logger.debug("span start failed", exc_info=True)
                    return await original(self, *args, **kwargs)
                try:
                    result = await original(self, *args, **kwargs)
                except BaseException as exc:
                    handle.end(error=f"{type(exc).__name__}: {exc}")
                    raise
                if kwargs.get("stream"):
                    handle.detach()
                    return TracedAsyncStream(result, handle, collector_cls(kwargs))
                try:
                    capture(handle, result)
                except Exception:
                    logger.debug("response capture failed", exc_info=True)
                handle.end()
                return result

            return wrapper

        def wrapper(self, *args, **kwargs):
            if _is_raw_response_call(kwargs):
                return original(self, *args, **kwargs)
            try:
                handle = _start_llm_span(endpoint, params, kwargs)
            except Exception:
                logger.debug("span start failed", exc_info=True)
                return original(self, *args, **kwargs)
            try:
                result = original(self, *args, **kwargs)
            except BaseException as exc:
                handle.end(error=f"{type(exc).__name__}: {exc}")
                raise
            if kwargs.get("stream"):
                handle.detach()
                return TracedSyncStream(result, handle, collector_cls(kwargs))
            try:
                capture(handle, result)
            except Exception:
                logger.debug("response capture failed", exc_info=True)
            handle.end()
            return result

        return wrapper

    return factory


def patch() -> bool:
    """Wrap the openai client methods. Idempotent. Returns True if anything
    was newly patched."""
    from openai.resources.chat import completions as chat_completions

    changed = False
    changed |= patch_method(
        chat_completions.Completions,
        "create",
        _make_wrapper(
            "chat.completions", CHAT_PARAMS, _capture_chat_response,
            ChatStreamCollector, is_async=False,
        ),
    )
    changed |= patch_method(
        chat_completions.AsyncCompletions,
        "create",
        _make_wrapper(
            "chat.completions", CHAT_PARAMS, _capture_chat_response,
            ChatStreamCollector, is_async=True,
        ),
    )
    try:
        from openai.resources import responses as responses_mod

        changed |= patch_method(
            responses_mod.Responses,
            "create",
            _make_wrapper(
                "responses", RESPONSES_PARAMS, _capture_responses_response,
                ResponsesStreamCollector, is_async=False,
            ),
        )
        changed |= patch_method(
            responses_mod.AsyncResponses,
            "create",
            _make_wrapper(
                "responses", RESPONSES_PARAMS, _capture_responses_response,
                ResponsesStreamCollector, is_async=True,
            ),
        )
    except Exception:
        logger.debug("openai Responses API not patched", exc_info=True)
    return changed


def unpatch() -> None:
    from openai.resources.chat import completions as chat_completions

    unpatch_method(chat_completions.Completions, "create")
    unpatch_method(chat_completions.AsyncCompletions, "create")
    try:
        from openai.resources import responses as responses_mod

        unpatch_method(responses_mod.Responses, "create")
        unpatch_method(responses_mod.AsyncResponses, "create")
    except Exception:
        pass
