"""Patcher for the anthropic client: Messages API.

Covers sync and async ``messages.create`` (streaming and non-streaming),
tool use, prompt-cache usage fields, and the ``messages.stream()`` helper,
which bypasses ``create`` inside the SDK and needs its own wrapper. The
anthropic SDK is imported lazily.

For ``messages.stream()``, usage comes from the accumulated message
snapshot at context exit. If the caller abandoned the stream early the
snapshot is partial, so the span is marked ``usage_estimated``.
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

MESSAGES_PARAMS = (
    "model",
    "messages",
    "system",
    "tools",
    "tool_choice",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "stop_sequences",
    "thinking",
    "output_config",
)


def is_available() -> bool:
    return importlib.util.find_spec("anthropic") is not None


def normalized_request(kwargs: dict[str, Any]) -> dict[str, Any]:
    """The stored request payload; also the input to request_hash.

    Replay reproduces hashes through this same function, so its output for
    given kwargs is a compatibility surface.
    """
    request = clean_params(kwargs, MESSAGES_PARAMS)
    request["endpoint"] = "messages"
    return request


def _start_llm_span(kwargs: dict[str, Any], stream: bool):
    recorder = get_recorder()
    request = normalized_request(kwargs)
    model = kwargs.get("model") or "unknown"
    attributes = {
        "gen_ai.system": "anthropic",
        "gen_ai.request.model": model,
        "stream": stream,
        "request": request,
        "request_hash": request_hash(request),
    }
    return recorder.start_span(f"chat {model}", kind="llm", attributes=attributes)


def _set_usage(handle, usage: Any) -> None:
    if usage is None:
        return
    handle.set_attributes(
        {
            "gen_ai.usage.input_tokens": getattr(usage, "input_tokens", 0) or 0,
            "gen_ai.usage.output_tokens": getattr(usage, "output_tokens", 0) or 0,
            "cached_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
            "cache_write_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        }
    )


def _normalize_content(message: Any) -> dict[str, Any]:
    text_parts = []
    thinking_parts = []
    tool_calls = []
    for block in getattr(message, "content", None) or []:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(getattr(block, "text", "") or "")
        elif block_type == "thinking":
            thinking_parts.append(getattr(block, "thinking", "") or "")
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": getattr(block, "id", None),
                    "name": getattr(block, "name", None),
                    "arguments": getattr(block, "input", None),
                }
            )
    out = {"text": "\n".join(text_parts) or None, "tool_calls": tool_calls}
    if thinking_parts:
        out["thinking"] = "\n".join(thinking_parts)
    return out


def _capture_message(handle, message: Any) -> None:
    try:
        raw = message.model_dump(mode="json")
    except Exception:
        raw = None
    handle.set_attributes(
        {
            "gen_ai.response.model": getattr(message, "model", None),
            "gen_ai.response.id": getattr(message, "id", None),
            "finish_reason": getattr(message, "stop_reason", None),
            "response": _normalize_content(message),
            "response_raw": raw,
        }
    )
    _set_usage(handle, getattr(message, "usage", None))


def _request_text(kwargs: dict[str, Any]) -> str:
    parts = []
    system = kwargs.get("system")
    if isinstance(system, str):
        parts.append(system)
    try:
        for message in kwargs.get("messages") or []:
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


class MessagesStreamCollector:
    """Accumulates raw SSE events from ``create(stream=True)``."""

    def __init__(self, kwargs: dict[str, Any]):
        self._request_text = _request_text(kwargs)
        self.model = None
        self.response_id = None
        self.stop_reason = None
        self.input_usage = None
        self.output_tokens = None
        self.text = []
        self.thinking = []
        self.tool_blocks: dict[int, dict[str, Any]] = {}

    def add(self, event: Any) -> None:
        event_type = getattr(event, "type", None)
        if event_type == "message_start":
            message = getattr(event, "message", None)
            self.model = getattr(message, "model", None)
            self.response_id = getattr(message, "id", None)
            self.input_usage = getattr(message, "usage", None)
        elif event_type == "content_block_start":
            block = getattr(event, "content_block", None)
            if getattr(block, "type", None) == "tool_use":
                index = getattr(event, "index", 0) or 0
                self.tool_blocks[index] = {
                    "id": getattr(block, "id", None),
                    "name": getattr(block, "name", None),
                    "arguments": "",
                }
        elif event_type == "content_block_delta":
            delta = getattr(event, "delta", None)
            delta_type = getattr(delta, "type", None)
            if delta_type == "text_delta":
                self.text.append(getattr(delta, "text", "") or "")
            elif delta_type == "thinking_delta":
                self.thinking.append(getattr(delta, "thinking", "") or "")
            elif delta_type == "input_json_delta":
                index = getattr(event, "index", 0) or 0
                slot = self.tool_blocks.get(index)
                if slot is not None:
                    slot["arguments"] += getattr(delta, "partial_json", "") or ""
        elif event_type == "message_delta":
            usage = getattr(event, "usage", None)
            if usage is not None:
                self.output_tokens = getattr(usage, "output_tokens", None)
            delta = getattr(event, "delta", None)
            if getattr(delta, "stop_reason", None):
                self.stop_reason = delta.stop_reason

    def finalize(self, handle, error: str | None) -> None:
        text = "".join(self.text)
        response = {"text": text or None, "tool_calls": list(self.tool_blocks.values())}
        if self.thinking:
            response["thinking"] = "".join(self.thinking)
        handle.set_attributes(
            {
                "gen_ai.response.model": self.model,
                "gen_ai.response.id": self.response_id,
                "finish_reason": self.stop_reason,
                "response": response,
            }
        )
        if self.input_usage is not None:
            _set_usage(handle, self.input_usage)
        if self.output_tokens is not None:
            handle.set_attribute("gen_ai.usage.output_tokens", self.output_tokens)
        else:
            handle.set_attributes(
                {
                    "gen_ai.usage.output_tokens": estimate_tokens(text),
                    "usage_estimated": True,
                }
            )
            if self.input_usage is None:
                handle.set_attribute(
                    "gen_ai.usage.input_tokens", estimate_tokens(self._request_text)
                )
        handle.end(error=error)


def _make_create_wrapper(is_async: bool):
    def factory(original):
        if is_async:

            async def wrapper(self, *args, **kwargs):
                try:
                    handle = _start_llm_span(kwargs, stream=bool(kwargs.get("stream")))
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
                    return TracedAsyncStream(
                        result, handle, MessagesStreamCollector(kwargs)
                    )
                try:
                    _capture_message(handle, result)
                except Exception:
                    logger.debug("response capture failed", exc_info=True)
                handle.end()
                return result

            return wrapper

        def wrapper(self, *args, **kwargs):
            try:
                handle = _start_llm_span(kwargs, stream=bool(kwargs.get("stream")))
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
                return TracedSyncStream(result, handle, MessagesStreamCollector(kwargs))
            try:
                _capture_message(handle, result)
            except Exception:
                logger.debug("response capture failed", exc_info=True)
            handle.end()
            return result

        return wrapper

    return factory


class _TracedStreamManager:
    """Wraps MessageStreamManager so the span ends at context exit."""

    def __init__(self, inner: Any, handle, request_text: str):
        self._inner = inner
        self._tb_handle = handle
        self._request_text = request_text
        self._stream = None

    def __enter__(self):
        try:
            self._stream = self._inner.__enter__()
        except BaseException as exc:
            self._tb_handle.end(error=f"{type(exc).__name__}: {exc}")
            raise
        return self._stream

    def __exit__(self, exc_type, exc, tb):
        try:
            return self._inner.__exit__(exc_type, exc, tb)
        finally:
            error = f"{type(exc).__name__}: {exc}" if exc is not None else None
            _finalize_from_snapshot(
                self._tb_handle, self._stream, error, self._request_text
            )

    def __getattr__(self, name):
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)


class _TracedAsyncStreamManager:
    def __init__(self, inner: Any, handle, request_text: str):
        self._inner = inner
        self._tb_handle = handle
        self._request_text = request_text
        self._stream = None

    async def __aenter__(self):
        try:
            self._stream = await self._inner.__aenter__()
        except BaseException as exc:
            self._tb_handle.end(error=f"{type(exc).__name__}: {exc}")
            raise
        return self._stream

    async def __aexit__(self, exc_type, exc, tb):
        try:
            return await self._inner.__aexit__(exc_type, exc, tb)
        finally:
            error = f"{type(exc).__name__}: {exc}" if exc is not None else None
            _finalize_from_snapshot(
                self._tb_handle, self._stream, error, self._request_text
            )

    def __getattr__(self, name):
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)


def _finalize_from_snapshot(
    handle, stream: Any, error: str | None, request_text: str = ""
) -> None:
    try:
        # The snapshot property asserts once any event has arrived; before
        # that it raises, which lands in the except branch below.
        snapshot = getattr(stream, "current_message_snapshot", None)
    except Exception:
        snapshot = None
    try:
        if snapshot is not None:
            _capture_message(handle, snapshot)
            if getattr(snapshot, "stop_reason", None) is None:
                # The caller left before the stream finished; counts are partial.
                handle.set_attribute("usage_estimated", True)
        else:
            # Nothing was consumed; the request still went out and was billed.
            handle.set_attributes(
                {
                    "gen_ai.usage.input_tokens": estimate_tokens(request_text),
                    "gen_ai.usage.output_tokens": 0,
                    "usage_estimated": True,
                }
            )
    except Exception:
        logger.debug("snapshot capture failed", exc_info=True)
    handle.end(error=error)


def _make_stream_wrapper(is_async: bool):
    def factory(original):
        def wrapper(self, *args, **kwargs):
            try:
                handle = _start_llm_span(kwargs, stream=True)
                handle.detach()
            except Exception:
                logger.debug("span start failed", exc_info=True)
                return original(self, *args, **kwargs)
            try:
                manager = original(self, *args, **kwargs)
            except BaseException as exc:
                handle.end(error=f"{type(exc).__name__}: {exc}")
                raise
            if is_async:
                return _TracedAsyncStreamManager(manager, handle, _request_text(kwargs))
            return _TracedStreamManager(manager, handle, _request_text(kwargs))

        return wrapper

    return factory


def patch() -> bool:
    """Wrap the anthropic client methods. Idempotent."""
    from anthropic.resources import messages as messages_mod

    changed = False
    changed |= patch_method(
        messages_mod.Messages, "create", _make_create_wrapper(is_async=False)
    )
    changed |= patch_method(
        messages_mod.AsyncMessages, "create", _make_create_wrapper(is_async=True)
    )
    changed |= patch_method(
        messages_mod.Messages, "stream", _make_stream_wrapper(is_async=False)
    )
    changed |= patch_method(
        messages_mod.AsyncMessages, "stream", _make_stream_wrapper(is_async=True)
    )
    return changed


def unpatch() -> None:
    from anthropic.resources import messages as messages_mod

    unpatch_method(messages_mod.Messages, "create")
    unpatch_method(messages_mod.AsyncMessages, "create")
    unpatch_method(messages_mod.Messages, "stream")
    unpatch_method(messages_mod.AsyncMessages, "stream")
