# Adding an instrumentation adapter

An adapter turns a client library's calls into `llm` spans. The openai and
anthropic adapters in `src/traceburn/instrument/` are the reference
implementations for a client class with methods to patch; `litellm.py` is
the reference for a library that exposes plain module-level functions
instead. This page is the map.

When a client's response objects mirror the shape of one already
supported (litellm's `ModelResponse` is deliberately built to match the
openai SDK's chat-completion objects field for field), reuse the capture
helpers in `_util.py` (`capture_chat_response`, `ChatStreamCollector`,
`messages_text`, `set_chat_usage`, `raw_dump`) instead of duplicating them.
`patch_method` / `unpatch_method` work on a module object the same way
they work on a class, so a module-level call surface doesn't need its own
patching mechanism.

## The shape

One module with three functions:

```python
def is_available() -> bool: ...   # importlib.util.find_spec, no import
def patch() -> bool: ...          # wrap the client methods; True if newly patched
def unpatch() -> None: ...        # restore the originals
```

Register it in the `_PATCHERS` dict in `instrument/__init__.py` and
`traceburn.install()` picks it up.

## The rules that matter

1. **Never break the host program.** Wrap span creation and response
   capture in try/except and fall back to calling the original method
   untraced. The only exceptions that may propagate are the client's own.
2. **Lazy imports.** The module must import cleanly when the client
   library is absent; import the client inside `patch()`.
3. **Idempotent patching.** Use `patch_method` / `unpatch_method` from
   `instrument/_util.py`; they mark wrappers and refuse double-patching.
4. **Honest usage.** Map the provider's usage report into the traceburn
   convention (see `schema.py`): `gen_ai.usage.input_tokens` is full-rate
   input only, cache reads and writes are separate fields. When usage is
   missing, estimate with `estimate_tokens` and set `usage_estimated`.

## What to capture

Build the span at request time:

- name: `"chat {model}"`, kind `"llm"`
- `gen_ai.system` (provider), `gen_ai.request.model`
- `request`: the normalized payload from an explicit allowlist of
  output-affecting params (never headers, never credentials)
- `request_hash`: `request_hash(normalized_request)` from `_util.py`; keep
  the normalization function stable, replay depends on it

Fill in at response time: `gen_ai.response.model`, usage fields,
`finish_reason`, the normalized `response` (text plus tool calls), and
`response_raw` (the provider's exact payload via `model_dump`) so replay
can reconstruct real SDK objects.

## Streaming

Wrap the returned stream in `TracedSyncStream` / `TracedAsyncStream` with
a collector object: `add(chunk)` accumulates, `finalize(handle, error)`
sets attributes and ends the span. Call `handle.detach()` before returning
the wrapped stream so later spans do not nest under the in-flight call.
The wrappers already handle exhaustion, close, context-manager exit,
errors, and garbage-collection finalize.

## Tests

Use `httpx.MockTransport` (or your client's equivalent) so the real SDK
parses real response shapes with zero network. Cover: a plain call, an
async call, a streaming call with and without usage, a tool call, an API
error, and the capture-failure fallback. See
`tests/test_instrument_openai.py` for the pattern.
