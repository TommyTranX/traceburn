"""Shared helpers for client patchers.

Everything here obeys one rule: instrumentation must never break the
instrumented program. Capture code is wrapped; when it fails, the call
proceeds untraced and the failure is logged at debug level.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Callable

from ..recorder import SpanHandle
from ..store import _sanitize

logger = logging.getLogger("traceburn")

_SENTINEL_TYPE_NAMES = ("NotGiven", "Omit")


def clean_params(params: dict[str, Any], allow: tuple[str, ...]) -> dict[str, Any]:
    """Keep only allowlisted params that carry a real value."""
    out = {}
    for key in allow:
        value = params.get(key)
        if value is None or type(value).__name__ in _SENTINEL_TYPE_NAMES:
            continue
        out[key] = value
    return out


def request_hash(payload: dict[str, Any]) -> str:
    """Stable hash of a normalized request, used by replay and dedup."""
    canonical = json.dumps(
        _sanitize(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    """Estimate tokens locally using roughly four characters per token.

    This deliberately uses no optional tokenizer: loading tokenizer data can
    download files and makes results depend on the caller's environment.
    The heuristic is rough, particularly for code and non-English text.
    Provider-reported usage takes precedence; fallback counts are marked
    ``usage_estimated`` by the instrumentation.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def messages_text(messages: Any) -> str:
    """Flatten a chat-messages list to plain text, for token estimation."""
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


def raw_dump(result: Any) -> Any:
    try:
        return result.model_dump(mode="json")
    except Exception:
        return None


def set_chat_usage(handle: SpanHandle, usage: Any) -> None:
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


def capture_chat_response(handle: SpanHandle, result: Any) -> None:
    """Record a Chat-Completions-shaped response onto a span.

    Shared by the openai and litellm patchers: litellm's ``ModelResponse``
    and streaming chunks are deliberately built to mirror the openai SDK's
    chat-completion shapes field for field, so one capture path covers both.
    """
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
            "response_raw": raw_dump(result),
        }
    )
    set_chat_usage(handle, getattr(result, "usage", None))


class ChatStreamCollector:
    """Accumulates chat-completion chunks into final span attributes.

    Shared by the openai and litellm patchers; see ``capture_chat_response``.
    """

    def __init__(self, kwargs: dict[str, Any]):
        self._request_text = messages_text(kwargs.get("messages"))
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

    def finalize(self, handle: SpanHandle, error: str | None) -> None:
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
            set_chat_usage(handle, self.usage)
        else:
            handle.set_attributes(
                {
                    "gen_ai.usage.input_tokens": estimate_tokens(self._request_text),
                    "gen_ai.usage.output_tokens": estimate_tokens(text),
                    "usage_estimated": True,
                }
            )
        handle.end(error=error)


class TracedSyncStream:
    """Wraps a provider stream so the span closes when the stream does.

    Chunks pass through untouched. ``collector.add`` sees each chunk;
    ``collector.finalize`` runs exactly once, on exhaustion, close, error,
    or context-manager exit, and is responsible for ending the span.
    """

    def __init__(self, inner: Any, handle: SpanHandle, collector: Any):
        self._inner = inner
        self._tb_handle = handle
        self._collector = collector
        self._iter = None
        self._done = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._iter is None:
            self._iter = iter(self._inner)
        try:
            chunk = next(self._iter)
        except StopIteration:
            self._finish()
            raise
        except BaseException as exc:
            self._finish(error=f"{type(exc).__name__}: {exc}")
            raise
        try:
            self._collector.add(chunk)
        except Exception:
            logger.debug("stream capture failed", exc_info=True)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            return self._inner.__exit__(exc_type, exc, tb)
        finally:
            if exc is not None:
                self._finish(error=f"{type(exc).__name__}: {exc}")
            else:
                self._finish()

    def close(self):
        try:
            self._inner.close()
        finally:
            self._finish()

    def _finish(self, error: str | None = None):
        if self._done:
            return
        self._done = True
        try:
            self._collector.finalize(self._tb_handle, error)
        except Exception:
            logger.debug("stream finalize failed", exc_info=True)
            try:
                self._tb_handle.end(error=error)
            except Exception:
                pass

    def __del__(self):
        # Last-resort finalize for streams abandoned without close(). The
        # span was detached at creation, so ending here never touches
        # another context's state.
        try:
            self._finish()
        except Exception:
            pass

    def __getattr__(self, name):
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)


class TracedAsyncStream:
    """Async twin of TracedSyncStream."""

    def __init__(self, inner: Any, handle: SpanHandle, collector: Any):
        self._inner = inner
        self._tb_handle = handle
        self._collector = collector
        self._iter = None
        self._done = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._iter is None:
            self._iter = self._inner.__aiter__()
        try:
            chunk = await self._iter.__anext__()
        except StopAsyncIteration:
            self._finish()
            raise
        except BaseException as exc:
            self._finish(error=f"{type(exc).__name__}: {exc}")
            raise
        try:
            self._collector.add(chunk)
        except Exception:
            logger.debug("stream capture failed", exc_info=True)
        return chunk

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        try:
            return await self._inner.__aexit__(exc_type, exc, tb)
        finally:
            if exc is not None:
                self._finish(error=f"{type(exc).__name__}: {exc}")
            else:
                self._finish()

    async def close(self):
        # Both SDKs' AsyncStream expose ``async def close()``; mirror it.
        try:
            await self._inner.close()
        finally:
            self._finish()

    async def aclose(self):
        try:
            inner_aclose = getattr(self._inner, "aclose", None)
            if inner_aclose is not None:
                await inner_aclose()
            else:
                await self._inner.close()
        finally:
            self._finish()

    def _finish(self, error: str | None = None):
        if self._done:
            return
        self._done = True
        try:
            self._collector.finalize(self._tb_handle, error)
        except Exception:
            logger.debug("stream finalize failed", exc_info=True)
            try:
                self._tb_handle.end(error=error)
            except Exception:
                pass

    def __del__(self):
        try:
            self._finish()
        except Exception:
            pass

    def __getattr__(self, name):
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)


def patch_method(cls: Any, name: str, wrapper_factory: Callable) -> bool:
    """Replace ``cls.name`` with ``wrapper_factory(original)``, once.

    ``cls`` is usually a class (an SDK client's method), but a module works
    too (litellm's ``completion``/``acompletion`` are module-level
    functions, not client methods); both support ``__dict__``/``setattr``.

    Returns True when the method was patched by this call, False when it was
    already patched. The original lands on the wrapper as
    ``__traceburn_original__`` so unpatching can restore it.
    """
    original = cls.__dict__.get(name)
    if original is None:
        original = getattr(cls, name)
    if getattr(original, "__traceburn_original__", None) is not None:
        return False
    wrapped = wrapper_factory(original)
    wrapped.__traceburn_original__ = original
    setattr(cls, name, wrapped)
    return True


def unpatch_method(cls: Any, name: str) -> None:
    current = getattr(cls, name, None)
    original = getattr(current, "__traceburn_original__", None)
    if original is not None:
        setattr(cls, name, original)
