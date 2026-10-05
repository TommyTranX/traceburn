# Standalone reports

With version 0.2.0 or newer installed from PyPI, run:

```bash
traceburn demo
traceburn report -o report.html
traceburn report <trace-id-or-prefix> -o another-run.html --open
```

`demo` creates a synthetic example with a repeated request and opens it in the
browser. It uses a temporary database, which is deleted afterward, and leaves
one HTML file at `.traceburn/demo/report.html`. No API keys or model calls are
needed. Use `--no-browser` on a server. Both commands refuse to replace an existing
file unless `--force` is supplied.

`report` reads the latest trace by default. You can also use `latest` explicitly
with `report`, `show`, `waste`, `fix`, `check`, or either side of `diff`.

## What leaves the database

The export includes only numeric usage, elapsed durations, estimated costs, a
call hierarchy with new local call numbers, fixed span-kind/status labels, and
known finding categories. It contains no scripts, remote fonts, or external assets.
Expand each finding's caveat and follow its call links to inspect the evidence.

Names, model identifiers, raw trace/span IDs, absolute timestamps, prompts,
responses, tool arguments, error messages, paths, request hashes, and arbitrary
attributes are omitted. Rule-generated summaries can contain model or tool names,
so the export uses fixed descriptions instead. The local rules still inspect
recorded payloads when computing findings.

This does not change what the source database records. Do not share the database
assuming that it was redacted. Numeric usage, timing, topology, and costs can
also be sensitive. Open and review the HTML before sharing it.

## Reading findings

Each finding gives a next step and a reason it may be expected. A repeated request
can be intentional sampling. A short answer does not prove that a smaller model
will work. Try changes on representative inputs and evaluate task outcomes.

Potential savings are hypotheses, not measured improvements. Per-finding savings
can overlap, so do not add them together. Unknown costs stay unknown; when any
model call lacks a valid cost, the summary is labeled a known-cost subtotal.

The demo uses invented token counts, costs, and durations. It is a walkthrough,
not a benchmark or a savings claim. A real before/after comparison requires
recorded runs and task checks supplied by the developer.
