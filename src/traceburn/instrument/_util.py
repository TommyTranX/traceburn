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


_ENCODER = None
_ENCODER_FAILED = False


def estimate_tokens(text: str) -> int:
    """Rough token count: tiktoken when importable, else a length heuristic.

    Only used when a provider omits usage (streaming without a usage
    payload). Spans carrying these numbers are marked usage_estimated.
    """
    if not text:
        return 0
    global _ENCODER, _ENCODER_FAILED
    if _ENCODER is None and not _ENCODER_FAILED:
        try:
            import tiktoken

            _ENCODER = tiktoken.get_encoding("o200k_base")
        except Exception:
            _ENCODER_FAILED = True
    if _ENCODER is not None:
        try:
            return len(_ENCODER.encode(text))
        except Exception:
            pass
    # Documented heuristic: about four characters per token for English text.
    return max(1, len(text) // 4)


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


def patch_method(cls: type, name: str, wrapper_factory: Callable) -> bool:
    """Replace ``cls.name`` with ``wrapper_factory(original)``, once.

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


def unpatch_method(cls: type, name: str) -> None:
    current = getattr(cls, name, None)
    original = getattr(current, "__traceburn_original__", None)
    if original is not None:
        setattr(cls, name, original)
