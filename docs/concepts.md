# Concepts

## The data model

Three tables, plus content-addressed blobs, in one SQLite file:

- A **session** groups related traces (a batch run, a test suite pass).
- A **trace** is one run: everything under one root span.
- A **span** is one timed operation with a `kind`: `llm`, `tool`, `agent`,
  `retrieval`, or `custom`. Spans nest via `parent_id`.

LLM spans follow the OpenTelemetry GenAI attribute names where they exist
(`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, and
so on) plus traceburn extensions: cache token splits, estimated cost, the
normalized request and response, and a `request_hash` used by replay and
duplicate detection. The full conventions live in `schema.py` and are a
public, versioned interface.

## Token accounting

`gen_ai.usage.input_tokens` counts only tokens billed at the full input
rate. Cache reads (`cached_input_tokens`) and cache writes
(`cache_write_tokens`) are separate, so cost math stays auditable:

```
cost = input * input_rate + cached * cached_rate
     + cache_write * write_rate + output * output_rate
```

Rates come from a dated, user-overridable pricing table
(`pricing.json`, `TRACEBURN_PRICING`). When a provider omits usage (some
streaming shapes), counts are estimated and the span carries
`usage_estimated: true`.

## Storage

WAL mode, so the viewer reads while your agent writes. Large request and
response payloads are stored once in a blobs table, keyed by content hash;
identical payloads across calls cost one row, which is also what makes
duplicate detection and replay lookups cheap. The schema is versioned via
a meta row.

## Write path guarantees

A failure inside traceburn never breaks your agent: store writes are
guarded, patcher capture errors degrade to an untraced call, and spans for
abandoned streams finalize at garbage collection. Exceptions from your own
code always propagate; the span just records the error first.

## The analyzers

Everything in `traceburn/analyze/` reads the store and computes:
flamegraph folds (latency- or cost-weighted), waterfall timelines, run
diffs, deterministic replay, and the waste report. None of it makes a
network call. The CLI and the web viewer are two skins over the same
functions, and every JSON shape they emit is a public interface you can
consume without importing the analyzers.
