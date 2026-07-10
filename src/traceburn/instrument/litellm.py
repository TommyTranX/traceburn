"""Patcher for litellm's module-level completion functions.

litellm fronts dozens of providers behind one call surface,
``litellm.completion`` / ``litellm.acompletion``, rather than a client
class, so this patcher wraps those two functions directly instead of a
class method. The provider recorded on the span comes from
``litellm.get_llm_provider(model)``, since litellm itself is the thing that
knows which backend a given model string routes to; when litellm can't
classify the model, the span still records (as provider ``"litellm"``) but
gets no dollar figure, since the pricing table has nothing to key against.

litellm's ``ModelResponse`` and its streaming chunks are deliberately built
to mirror the openai SDK's chat-completion shapes field for field, so
response capture reuses the shared helpers in ``_util.py`` rather than
duplicating them.

Double-counting: verified against litellm 1.91.1 that this patcher and the
openai/anthropic patchers do not both fire for the same call. litellm's own
openai transport calls ``client.chat.completions.with_raw_response.create``,
which the openai patcher already detects and passes through untraced (see
``test_raw_response_calls_pass_through_untraced``); its anthropic transport
does not use the anthropic SDK at all, it speaks to the API directly over
httpx. A future litellm release could change either of those internals;
there's no code-level guard against that, just this test coverage.

Known gap, deliberate for now: only ``completion``/``acompletion`` are
patched. ``embedding``, ``image_generation``, ``text_completion``, and the
Router/Proxy entry points are not.

Network note: this is the one adapter where traceburn's own "no network
calls of its own" claim needs a caveat about a dependency, not itself.
litellm tries to fetch its model cost map from GitHub the first time it is
imported, falling back to a bundled copy if that fails; this happens
because the host program imported litellm for its own use, the same as it
would without traceburn installed, so it is not traffic this patcher adds.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any

from ..recorder import get_recorder
from ._util import (
    ChatStreamCollector,
    TracedAsyncStream,
    TracedSyncStream,
    capture_chat_response,
    clean_params,
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


def is_available() -> bool:
    return importlib.util.find_spec("litellm") is not None


def _normalized_kwargs(args: tuple, kwargs: dict[str, Any]) -> dict[str, Any]:
    """A kwargs-shaped view of a call for span construction.

    ``litellm.completion(model, messages, ...)`` accepts its first two
    arguments positionally; that pair is the stable part of the public
    signature, so it's safe to fold in without inspecting the rest.
    """
    merged = dict(kwargs)
    if args:
        merged.setdefault("model", args[0])
    if len(args) > 1:
        merged.setdefault("messages", args[1])
    return merged


def _resolve_provider(model: str) -> tuple[str, str]:
    """(bare_model, provider). Falls back to (model, "litellm") when litellm
    itself can't classify the model string, which is an expected case for
    providers or aliases litellm doesn't recognize."""
    try:
        import litellm as litellm_mod

        bare_model, custom_llm_provider, _, _ = litellm_mod.get_llm_provider(model)
        return bare_model, custom_llm_provider or "litellm"
    except Exception:
        return model, "litellm"


def normalized_request(kwargs: dict[str, Any]) -> dict[str, Any]:
    request = clean_params(kwargs, CHAT_PARAMS)
    request["endpoint"] = "completion"
    return request


def _start_llm_span(kwargs: dict[str, Any]):
    recorder = get_recorder()
    request = normalized_request(kwargs)
    raw_model = kwargs.get("model") or "unknown"
    bare_model, prov = _resolve_provider(raw_model)
    attributes = {
        "gen_ai.system": prov,
        "gen_ai.request.model": bare_model,
        "stream": bool(kwargs.get("stream")),
        "request": request,
        "request_hash": request_hash(request),
    }
    return recorder.start_span(f"chat {bare_model}", kind="llm", attributes=attributes)


def _make_wrapper(is_async: bool):
    def factory(original):
        if is_async:

            async def wrapper(*args, **kwargs):
                call_kwargs = _normalized_kwargs(args, kwargs)
                try:
                    handle = _start_llm_span(call_kwargs)
                except Exception:
                    logger.debug("span start failed", exc_info=True)
                    return await original(*args, **kwargs)
                try:
                    result = await original(*args, **kwargs)
                except BaseException as exc:
                    handle.end(error=f"{type(exc).__name__}: {exc}")
                    raise
                if call_kwargs.get("stream"):
                    handle.detach()
                    return TracedAsyncStream(result, handle, ChatStreamCollector(call_kwargs))
                try:
                    capture_chat_response(handle, result)
                except Exception:
                    logger.debug("response capture failed", exc_info=True)
                handle.end()
                return result

            return wrapper

        def wrapper(*args, **kwargs):
            call_kwargs = _normalized_kwargs(args, kwargs)
            try:
                handle = _start_llm_span(call_kwargs)
            except Exception:
                logger.debug("span start failed", exc_info=True)
                return original(*args, **kwargs)
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                handle.end(error=f"{type(exc).__name__}: {exc}")
                raise
            if call_kwargs.get("stream"):
                handle.detach()
                return TracedSyncStream(result, handle, ChatStreamCollector(call_kwargs))
            try:
                capture_chat_response(handle, result)
            except Exception:
                logger.debug("response capture failed", exc_info=True)
            handle.end()
            return result

        return wrapper

    return factory


def patch() -> bool:
    """Wrap litellm.completion / litellm.acompletion. Idempotent. Returns
    True if anything was newly patched."""
    import litellm as litellm_mod

    changed = False
    changed |= patch_method(litellm_mod, "completion", _make_wrapper(is_async=False))
    changed |= patch_method(litellm_mod, "acompletion", _make_wrapper(is_async=True))
    return changed


def unpatch() -> None:
    import litellm as litellm_mod

    unpatch_method(litellm_mod, "completion")
    unpatch_method(litellm_mod, "acompletion")
