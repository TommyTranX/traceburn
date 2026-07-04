"""The recorder: session and parent tracking, timing, cost, and writes.

Context propagation uses ``contextvars``, so nesting is correct under
asyncio (tasks inherit their creator's context) and threads stay isolated
(a new thread starts with no active span; annotate inside the thread if you
need one). Spans are written to the store when they close.

A failure inside the recorder must never break the instrumented program.
Store writes are wrapped and logged, never raised. Exceptions from the
user's own code always propagate; the span just records the error first.

The public entry points live in the package root: ``span``, ``trace``,
``session``, and ``configure``. The ``Recorder`` class and ``SpanHandle``
are for instrumentation code (patchers open a handle before a client call
and close it when the response, or the last stream chunk, arrives).
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from .pricing import PricingTable
from .schema import (
    SPAN_KINDS,
    STATUS_ERROR,
    Session,
    Span,
    Trace,
    new_session_id,
    new_span_id,
    new_trace_id,
)
from .store import Store

logger = logging.getLogger("traceburn")

_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "traceburn_current_span", default=None
)
_current_session: contextvars.ContextVar[Session | None] = contextvars.ContextVar(
    "traceburn_current_session", default=None
)


def _now_ns() -> int:
    return time.time_ns()


class SpanHandle:
    """An open span. Close it exactly once with ``end``.

    ``detach`` releases the span from the ambient context without closing
    it, so streaming instrumentation can keep timing a response after the
    client call has returned and sibling calls no longer nest under it.

    If a handle will be ended from a different task or thread than the one
    that started it, call ``detach()`` in the starting context first.
    Otherwise that context keeps the ended span as its current parent until
    the enclosing span exits, and later spans nest under it.
    """

    def __init__(
        self,
        recorder: Recorder,
        span: Span,
        token: contextvars.Token,
        parent: Span | None = None,
    ):
        self._recorder = recorder
        self.span = span
        self._parent = parent
        self._token: contextvars.Token | None = token
        self._ended = False

    def set_attribute(self, key: str, value: Any) -> None:
        self.span.attributes[key] = value

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        self.span.attributes.update(attributes)

    def detach(self) -> None:
        if self._token is not None:
            try:
                _current_span.reset(self._token)
            except ValueError:
                # Ending from a different context than the one that started
                # the span. Only this context can be repaired; the starting
                # context still holds the span until its enclosing span exits.
                if _current_span.get() is self.span:
                    _current_span.set(self._parent)
                logger.warning(
                    "span %r (%s) was detached from a different context than "
                    "the one that started it; call detach() in the starting "
                    "context before handing the handle elsewhere",
                    self.span.name,
                    self.span.span_id,
                )
            self._token = None

    def end(
        self,
        status: str | None = None,
        error: str | None = None,
        end_ns: int | None = None,
    ) -> Span:
        if self._ended:
            return self.span
        self._ended = True
        self.detach()
        span = self.span
        end = end_ns if end_ns is not None else _now_ns()
        # Wall clocks can step backward (NTP); never persist a negative duration.
        span.end_ns = max(end, span.start_ns)
        if status is not None:
            span.status = status
        if error is not None:
            span.error = error
            span.status = STATUS_ERROR
        self._recorder._finalize_and_write(span)
        return span


class Recorder:
    """Creates spans, tracks context, computes cost, writes on close."""

    def __init__(self, store: Store | None = None, pricing: PricingTable | None = None):
        self._store = store
        self._pricing = pricing
        self._init_lock = threading.Lock()
        self._write_failure_warned = False

    @property
    def store(self) -> Store:
        if self._store is None:
            with self._init_lock:
                if self._store is None:
                    self._store = Store()
        return self._store

    @property
    def pricing(self) -> PricingTable:
        if self._pricing is None:
            with self._init_lock:
                if self._pricing is None:
                    self._pricing = PricingTable.load()
        return self._pricing

    # -- span lifecycle ---------------------------------------------------

    def start_span(
        self,
        name: str,
        kind: str = "custom",
        attributes: dict[str, Any] | None = None,
    ) -> SpanHandle:
        if kind not in SPAN_KINDS:
            raise ValueError(f"kind must be one of {SPAN_KINDS}, got {kind!r}")
        parent = _current_span.get()
        session = _current_session.get()
        start_ns = _now_ns()
        if parent is not None:
            trace_id = parent.trace_id
            parent_id = parent.span_id
            session_id = parent.session_id
        else:
            trace_id = new_trace_id()
            parent_id = None
            session_id = session.session_id if session else None
            trace_row = Trace(
                trace_id=trace_id,
                name=name,
                session_id=session_id,
                start_ns=start_ns,
            )
            self._safe_write(lambda store: store.insert_trace(trace_row))
        span = Span(
            span_id=new_span_id(),
            trace_id=trace_id,
            parent_id=parent_id,
            session_id=session_id,
            name=name,
            kind=kind,
            start_ns=start_ns,
            attributes=dict(attributes or {}),
        )
        token = _current_span.set(span)
        return SpanHandle(self, span, token, parent=parent)

    @contextmanager
    def span(
        self,
        name: str,
        kind: str = "custom",
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[SpanHandle]:
        handle = self.start_span(name, kind=kind, attributes=attributes)
        try:
            yield handle
        except BaseException as exc:
            handle.end(error=f"{type(exc).__name__}: {exc}")
            raise
        handle.end()

    def trace(
        self,
        name: str | Callable | None = None,
        kind: str = "agent",
        attributes: dict[str, Any] | None = None,
    ):
        """Decorator form of ``span``. Works on sync and async functions."""

        def decorate(fn: Callable) -> Callable:
            label = name if isinstance(name, str) and name else fn.__qualname__
            if inspect.isasyncgenfunction(fn):

                @functools.wraps(fn)
                async def asyncgen_wrapper(*args, **kwargs):
                    with self.span(label, kind=kind, attributes=attributes):
                        async for item in fn(*args, **kwargs):
                            yield item

                return asyncgen_wrapper

            if inspect.isgeneratorfunction(fn):

                @functools.wraps(fn)
                def gen_wrapper(*args, **kwargs):
                    with self.span(label, kind=kind, attributes=attributes):
                        yield from fn(*args, **kwargs)

                return gen_wrapper

            if inspect.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def async_wrapper(*args, **kwargs):
                    with self.span(label, kind=kind, attributes=attributes):
                        return await fn(*args, **kwargs)

                return async_wrapper

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                with self.span(label, kind=kind, attributes=attributes):
                    return fn(*args, **kwargs)

            return wrapper

        if callable(name):
            return decorate(name)
        return decorate

    # -- sessions ---------------------------------------------------------

    @contextmanager
    def session(self, name: str) -> Iterator[Session]:
        sess = Session(session_id=new_session_id(), name=name, start_ns=_now_ns())
        self._safe_write(lambda store: store.insert_session(sess))
        token = _current_session.set(sess)
        try:
            yield sess
        finally:
            try:
                _current_session.reset(token)
            except ValueError:
                # Exited from a different context than the one that entered.
                if _current_session.get() is sess:
                    _current_session.set(None)
            sess.end_ns = _now_ns()
            self._safe_write(lambda store: store.end_session(sess.session_id, sess.end_ns))

    # -- internals --------------------------------------------------------

    def _finalize_and_write(self, span: Span) -> None:
        attrs = span.attributes
        if span.kind == "llm" and "cost_usd" not in attrs:
            try:
                cost = self.pricing.cost_for_attributes(attrs)
            except Exception:
                logger.warning("cost computation failed", exc_info=True)
                cost = None
            if cost is not None:
                attrs["cost_usd"] = cost
        self._safe_write(lambda store: store.insert_span(span))
        if span.end_ns is not None:
            self._safe_write(
                lambda store: store.extend_trace_end(span.trace_id, span.end_ns)
            )

    def _safe_write(self, op: Callable[[Store], None]) -> None:
        """Run one store operation, resolving the store inside the guard.

        Nothing here may raise into the instrumented program: not a write
        error, and not a failure to open the store in the first place (an
        unwritable path, a newer schema). The first failure logs a warning
        with the traceback; later ones log at debug to avoid flooding.
        """
        try:
            op(self.store)
        except Exception:
            if not self._write_failure_warned:
                self._write_failure_warned = True
                logger.warning(
                    "trace write failed; tracing is degraded", exc_info=True
                )
            else:
                logger.debug("trace write failed", exc_info=True)


# -- module-level default recorder ---------------------------------------

_default_recorder: Recorder | None = None
_default_lock = threading.Lock()


def get_recorder() -> Recorder:
    global _default_recorder
    if _default_recorder is None:
        with _default_lock:
            if _default_recorder is None:
                _default_recorder = Recorder()
    return _default_recorder


def configure(
    db_path: str | None = None,
    pricing_path: str | None = None,
    store: Store | None = None,
    pricing: PricingTable | None = None,
) -> Recorder:
    """Replace the default recorder. Call before recording anything."""
    global _default_recorder
    if _current_span.get() is not None:
        logger.warning(
            "configure() called while a span is open; the open trace will be "
            "split across stores"
        )
    with _default_lock:
        _default_recorder = Recorder(
            store=store if store is not None else (Store(db_path) if db_path else None),
            pricing=pricing
            if pricing is not None
            else (PricingTable.load(pricing_path) if pricing_path else None),
        )
    return _default_recorder


def span(name: str, kind: str = "custom", attributes: dict[str, Any] | None = None):
    return get_recorder().span(name, kind=kind, attributes=attributes)


def trace(name=None, kind: str = "agent", attributes: dict[str, Any] | None = None):
    return get_recorder().trace(name, kind=kind, attributes=attributes)


def session(name: str):
    return get_recorder().session(name)


def current_span() -> Span | None:
    return _current_span.get()


def current_session() -> Session | None:
    return _current_session.get()
