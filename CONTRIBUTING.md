# Contributing

Contributions are welcome, and two kinds are especially wanted: new
instrumentation adapters and new waste rules. Both are deliberately small
interfaces; either is an afternoon of work.

## Setup

```
git clone https://github.com/TommyTranX/traceburn
cd traceburn
python -m venv .venv && . .venv/bin/activate
pip install -e ".[ui]" pytest openai anthropic litellm httpx
pytest
```

The test suite makes no network calls. Instrumentation tests run the real
provider SDKs over `httpx.MockTransport` (or, for litellm, its own
`mock_response` kwarg), so they exercise the exact objects the patchers see
in production without a key.

## Adding an instrumentation adapter

One module in `src/traceburn/instrument/`, three functions:
`is_available()`, `patch()`, `unpatch()`. The walkthrough with the span
attribute conventions, streaming wrappers, and the never-break rule is in
[docs/instrumentation.md](docs/instrumentation.md). Requirements:

- lazy imports, so the module is importable when the client library is not
- capture failures are swallowed and logged, never raised into the host
- tests via a mock transport: sync, async, streaming, tool calls, an error
- register it in `instrument/__init__.py`

## Adding a waste rule

One module in `src/traceburn/analyze/waste/` with `run(ctx) -> list[Finding]`.
The contract, helpers, and the precision bar are in
[docs/waste-rules.md](docs/waste-rules.md). The short version: a positive
test, a negative test, no dollar figure that does not follow from observed
tokens and the pricing table, and an honest confidence level.

## Ground rules

- `pytest` passes; new behavior comes with tests.
- The core package imports stdlib only. Anything heavier goes behind an
  extra.
- No telemetry, no network calls of the tool's own, ever.
- Plain prose in docs and messages. Cost figures are always labeled
  estimates.

## Updating the pricing table

`src/traceburn/pricing.json` carries provider list prices with an `as_of`
date. Corrections and new models are welcome; cite the public pricing page
in the PR and update `as_of`.
