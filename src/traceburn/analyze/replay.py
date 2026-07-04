"""Deterministic replay: re-run agent code against recorded responses.

Recording happens automatically whenever instrumentation is installed:
every non-streaming LLM span stores the provider's exact response payload
(``response_raw``) keyed by the hash of its normalized request. Replay
patches the client methods so a matching request returns the stored
response, reconstructed as a real SDK object, with no network and no
tokens spent.

    import traceburn
    from traceburn.analyze.replay import replay

    with replay(trace_id="..."):
        run_agent()          # every recorded call is served from the store

Semantics:
- Requests match by normalized-request hash (same normalization as the
  recorder). Repeats of the same request play back the recorded responses
  in order; once exhausted, the last one repeats.
- A request with no recording follows ``on_miss``: "raise" (default)
  raises ReplayMiss, "passthrough" performs the real call.
- Streaming requests (including the anthropic ``messages.stream()``
  helper) and raw-response calls are not replayable in this version and
  follow ``on_miss``.

Replay contexts nest and interleave safely: one wrapper is installed per
client method and consults a stack of active replayers, so the innermost
context wins and exiting one context never disturbs another. When no
context is active the wrapper is a transparent passthrough. Served calls
never reach the network and never record new spans.
"""

from __future__ import annotations

import importlib.util
import logging
import threading
from collections import deque
from contextlib import contextmanager
from typing import Any, Callable

from ..instrument._util import request_hash
from ..store import Store

logger = logging.getLogger("traceburn")


class ReplayMiss(Exception):
    """Raised when a request cannot be served and on_miss='raise'."""


class Replayer:
    """Index of recorded responses for one trace or session."""

    def __init__(
        self,
        store: Store | None = None,
        trace_id: str | None = None,
        session_id: str | None = None,
        on_miss: str = "raise",
    ):
        if on_miss not in ("raise", "passthrough"):
            raise ValueError("on_miss must be 'raise' or 'passthrough'")
        if not trace_id and not session_id:
            raise ValueError("pass trace_id or session_id")
        self.on_miss = on_miss
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        self._queues: dict[str, deque] = {}
        self._last: dict[str, Any] = {}
        self._unservable: set[str] = set()
        store = store or Store()
        if trace_id:
            trace_ids = [trace_id]
        else:
            trace_ids = [
                t.trace_id for t in store.list_traces(limit=10_000, session_id=session_id)
            ]
        spans = []
        for tid in trace_ids:
            spans.extend(store.get_spans(tid, hydrate=True))
        for span in sorted(spans, key=lambda s: s.start_ns):
            if span.kind != "llm" or span.status != "ok":
                continue
            key = span.attributes.get("request_hash")
            if not key:
                continue
            raw = span.attributes.get("response_raw")
            if raw is not None:
                self._queues.setdefault(key, deque()).append(raw)
            else:
                # Recorded before payload capture, or a streamed recording.
                self._unservable.add(key)
        self.recorded_calls = sum(len(q) for q in self._queues.values())

    def take(self, key: str) -> Any | None:
        with self._lock:
            queue = self._queues.get(key)
            if queue:
                raw = queue.popleft()
                self._last[key] = raw
                self.hits += 1
                return raw
            if key in self._last:
                # Exhausted: an identical request repeats the last recording.
                self.hits += 1
                return self._last[key]
            self.misses += 1
            return None

    def miss_reason(self, key: str) -> str:
        if key in self._unservable:
            return (
                "this request was recorded without a stored response payload "
                "(a streamed call, or a recording made before payload capture)"
            )
        return "the prompt, model, or params differ from the recording"


# One wrapper per client method, consulting a stack of active replayers.
_active: list[Replayer] = []
_patched: list[tuple[type, str]] = []
_state_lock = threading.Lock()


def _current() -> Replayer | None:
    return _active[-1] if _active else None


def _is_raw_response_call(kwargs: dict[str, Any]) -> bool:
    try:
        headers = kwargs.get("extra_headers") or {}
        return bool(headers.get("X-Stainless-Raw-Response"))
    except Exception:
        return False


def _serve(
    replayer: Replayer,
    kwargs: dict[str, Any],
    key_builder: Callable,
    reconstruct: Callable,
    describe: str,
):
    """Returns (served, value); unservable shapes follow on_miss."""
    if kwargs.get("stream"):
        if replayer.on_miss == "passthrough":
            return False, None
        raise ReplayMiss(
            f"streaming request not replayable ({describe}); "
            "record and replay it non-streaming, or use on_miss='passthrough'"
        )
    if _is_raw_response_call(kwargs):
        if replayer.on_miss == "passthrough":
            return False, None
        raise ReplayMiss(
            f"raw-response request not replayable ({describe}); "
            "use a plain call, or on_miss='passthrough'"
        )
    key = key_builder(kwargs)
    raw = replayer.take(key)
    if raw is None:
        if replayer.on_miss == "passthrough":
            return False, None
        raise ReplayMiss(
            f"no recorded response for this request ({describe}); "
            f"{replayer.miss_reason(key)}"
        )
    return True, reconstruct(raw)


def _make_wrapper(inner, key_builder, reconstruct, describe, is_async):
    if is_async:

        async def wrapper(self, *args, **kwargs):
            replayer = _current()
            if replayer is not None:
                served, value = _serve(replayer, kwargs, key_builder, reconstruct, describe)
                if served:
                    return value
            return await inner(self, *args, **kwargs)

        return wrapper

    def wrapper(self, *args, **kwargs):
        replayer = _current()
        if replayer is not None:
            served, value = _serve(replayer, kwargs, key_builder, reconstruct, describe)
            if served:
                return value
        return inner(self, *args, **kwargs)

    return wrapper


def _make_stream_helper_wrapper(inner, describe):
    """For helpers that are always streaming (anthropic messages.stream)."""

    def wrapper(self, *args, **kwargs):
        replayer = _current()
        if replayer is not None and replayer.on_miss == "raise":
            raise ReplayMiss(
                f"streaming request not replayable ({describe}); "
                "record and replay it non-streaming, or use on_miss='passthrough'"
            )
        return inner(self, *args, **kwargs)

    return wrapper


def _wrap(cls: type, name: str, factory: Callable) -> None:
    current = getattr(cls, name)
    if getattr(current, "__traceburn_replay__", False):
        return
    wrapped = factory(current)
    wrapped.__traceburn_replay__ = True
    wrapped.__traceburn_replay_inner__ = current
    setattr(cls, name, wrapped)
    entry = (cls, name)
    if entry not in _patched:
        _patched.append(entry)


def _unwrap_all() -> None:
    remaining = []
    for cls, name in _patched:
        current = getattr(cls, name, None)
        inner = getattr(current, "__traceburn_replay_inner__", None)
        if inner is not None:
            setattr(cls, name, inner)
        else:
            # Something wrapped on top of us (instrumentation installed
            # inside the replay context). Leave the wrapper in place; with
            # no active replayer it is a transparent passthrough.
            remaining.append((cls, name))
    _patched[:] = remaining


def _patch_openai() -> None:
    from openai.resources.chat import completions as chat_mod
    from openai.types.chat import ChatCompletion

    from ..instrument.openai import normalized_request

    _wrap(
        chat_mod.Completions,
        "create",
        lambda inner: _make_wrapper(
            inner,
            lambda kw: request_hash(normalized_request("chat.completions", kw)),
            ChatCompletion.model_validate,
            "chat.completions",
            is_async=False,
        ),
    )
    _wrap(
        chat_mod.AsyncCompletions,
        "create",
        lambda inner: _make_wrapper(
            inner,
            lambda kw: request_hash(normalized_request("chat.completions", kw)),
            ChatCompletion.model_validate,
            "chat.completions",
            is_async=True,
        ),
    )
    try:
        from openai.resources import responses as responses_mod
        from openai.types.responses import Response

        _wrap(
            responses_mod.Responses,
            "create",
            lambda inner: _make_wrapper(
                inner,
                lambda kw: request_hash(normalized_request("responses", kw)),
                Response.model_validate,
                "responses",
                is_async=False,
            ),
        )
        _wrap(
            responses_mod.AsyncResponses,
            "create",
            lambda inner: _make_wrapper(
                inner,
                lambda kw: request_hash(normalized_request("responses", kw)),
                Response.model_validate,
                "responses",
                is_async=True,
            ),
        )
    except Exception:
        logger.debug("openai Responses API not replay-patched", exc_info=True)


def _patch_anthropic() -> None:
    from anthropic.resources import messages as messages_mod
    from anthropic.types import Message

    from ..instrument.anthropic import normalized_request

    for cls, is_async in ((messages_mod.Messages, False), (messages_mod.AsyncMessages, True)):
        _wrap(
            cls,
            "create",
            lambda inner, is_async=is_async: _make_wrapper(
                inner,
                lambda kw: request_hash(normalized_request(kw)),
                Message.model_validate,
                "messages",
                is_async=is_async,
            ),
        )
        # messages.stream() posts directly, bypassing create; it must honor
        # on_miss rather than silently reaching the network.
        _wrap(
            cls,
            "stream",
            lambda inner: _make_stream_helper_wrapper(inner, "messages.stream"),
        )


@contextmanager
def replay(
    trace_id: str | None = None,
    session_id: str | None = None,
    store: Store | None = None,
    on_miss: str = "raise",
):
    """Serve recorded responses for the duration of the context."""
    replayer = Replayer(
        store=store, trace_id=trace_id, session_id=session_id, on_miss=on_miss
    )
    with _state_lock:
        if importlib.util.find_spec("openai") is not None:
            _patch_openai()
        if importlib.util.find_spec("anthropic") is not None:
            _patch_anthropic()
        _active.append(replayer)
    try:
        yield replayer
    finally:
        with _state_lock:
            try:
                _active.remove(replayer)
            except ValueError:
                pass
            if not _active:
                _unwrap_all()
