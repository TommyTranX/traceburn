# traceburn

A local-first tracer and efficiency profiler for AI agents. Record every
LLM and tool call into one SQLite file on your machine, then see where the
time and money went: a cost flamegraph, a waste report with quantified
avoidable spend, deterministic replay, and run diffs. No account, no
server, no Docker, no telemetry.

```
pip install "traceburn[ui]"      # tracer + the local web viewer
pip install traceburn            # core only (stdlib, no dependencies)
```

## A finding to start with

A realistic tool-using support agent (large policy prompt, one tool round
per ticket, claude-haiku-4-5) left 69 percent of its API spend on the
table by not caching its static prompt.

| run | cost | what changed |
|---|---|---|
| baseline | $0.0539 | ~4,700-token policy prefix re-sent uncached on all 10 calls |
| after fix | $0.0167 | one cache_control block, exactly what the waste report suggested |

`traceburn waste` flagged the uncached prefix, estimated the saving within
18 percent, and named the fix. The whole exchange is reproducible for a
few cents:
[examples/cache_before_after.py](https://github.com/TommyTranX/traceburn/blob/main/examples/cache_before_after.py)
(measured 2026-07-05; prices change, rerun it yourself).

## Quickstart

Three lines to instrument, one command to look:

```python
import traceburn
traceburn.install()          # patches openai and anthropic clients

# ... run your agent as usual ...
```

```
traceburn ui                 # flamegraph, waterfall, waste report, diffs
traceburn ls                 # or stay in the terminal
```

Or with zero code changes: `TRACEBURN=1 python your_agent.py`.

No API key handy? `python examples/offline_demo.py` records a simulated
agent run with realistic token counts, so you can see a trace, a
flamegraph, and a real waste finding in under a minute.

## What it does

- **Trace.** Every LLM call (openai and anthropic SDKs: sync, async,
  streaming, tool calls, prompt-cache usage) plus your own steps via
  `@trace`, `span()`, and `session()`. One local SQLite file. The viewer
  reads while your agent writes.
- **Flamegraph.** One run, width by wall-clock or by dollars, self-time
  separated. See the expensive step instead of scrolling for it.
- **Waste report.** Rules for duplicate calls, missed prompt caching,
  redundant context, oversized models on trivial steps, and retry loops.
  Findings carry their evidence spans and a confidence level, and claim
  dollars only when the figure follows from observed tokens and the dated
  pricing table; purely informational findings say so. Silent when your
  run is clean: precision over recall, by design.
- **Replay.** Recorded responses served back through the real SDK types.
  Re-run agent code deterministically with zero tokens spent; a test in
  this repo proves no request escapes to the network.
- **Diff.** Two runs, aligned step by step: token, cost, and latency
  deltas, plus text diffs of prompts and responses that changed.

## Framework support

Today: the raw `openai` and `anthropic` Python SDKs, patched automatically
by `install()`. Anything built directly on them is covered. LangChain,
LlamaIndex, and every framework that emits OpenTelemetry GenAI spans land
via the OTel ingest and OTLP export planned for v0.2 (see roadmap); the
explicit `span()` API works with any framework today.

## How the waste rules work

Each rule is one small module with one entry point, documented in
[docs/waste-rules.md](https://github.com/TommyTranX/traceburn/blob/main/docs/waste-rules.md). Two principles: a wrong finding
is worse than a missed one, and no dollar figure without observed tokens
and a listed price behind it. Every figure the tool prints is labeled an
estimate.

## Related work

The LLM observability space is busy; this tool takes one deliberate seat
in it. [Langfuse](https://github.com/langfuse/langfuse) is a full
open-source platform (server, Postgres, ClickHouse) with tracing, evals,
and prompt management. [Arize Phoenix](https://github.com/Arize-ai/phoenix)
is the closest neighbor: it runs locally with SQLite and no account, and is
excellent for tracing plus evals, though it runs as a server process with a
large dependency set and does not focus on waste detection, cost
flamegraphs, deterministic replay, or run diffs.
[MLflow](https://github.com/mlflow/mlflow) has been adding GenAI tracing,
trace comparison, and efficiency scoring to its tracking server.
[LangSmith](https://smith.langchain.com) is LangChain's hosted platform.
[OpenLLMetry](https://github.com/traceloop/openllmetry) is instrumentation
that exports OTel spans to a backend of your choice, and
[AgentSight](https://github.com/eunomia-bpf/agentsight) renders token
flamegraphs of coding agents from the system side on Linux.
[Helicone](https://github.com/Helicone/helicone),
[OpenLIT](https://github.com/openlit/openlit),
[Braintrust](https://braintrust.dev), and
[Logfire](https://pydantic.dev/logfire) each pair instrumentation with a
server or cloud backend.

traceburn's seat: strictly local (one file, no account, no server for
basic use), framework-agnostic at the SDK level, and efficiency-first
(the waste report, the dollar-weighted flamegraph, replay, and diff in one
small package). It aims to sit next to your existing stack, not replace it.

## Privacy and data

Everything stays on your machine. traceburn makes no network calls of its
own and sends no telemetry; the only network traffic is your agent's own
model API calls. A test
([tests/test_no_network.py](https://github.com/TommyTranX/traceburn/blob/main/tests/test_no_network.py))
blocks all socket access and exercises the recorder, the store, and every
analyzer to prove it. Auth headers are never recorded, so API keys cannot
end up in traces.

## Limitations

- Instrumentation covers the openai and anthropic Python SDKs. Helper
  paths not traced yet: `parse()` conveniences and `with_raw_response`
  (passed through untraced). Multi-choice (`n > 1`) responses record the
  first choice.
- Cost figures are estimates from a dated public pricing table
  ([src/traceburn/pricing.json](https://github.com/TommyTranX/traceburn/blob/main/src/traceburn/pricing.json)); long-context
  tiers and regional surcharges are not modeled. Token counts prefer
  provider-reported usage; estimated counts are flagged.
- Waste rules are heuristics tuned for precision. They will miss waste
  before they invent it, and every finding says how confident it is.
- Streaming calls are not replayable yet; replay serves non-streaming
  recordings.
- The viewer is read-only, bound to 127.0.0.1, and has no auth.

## Roadmap (v0.2)

OTel GenAI ingest and OTLP export (LangChain and LlamaIndex land here), a
pytest plugin built on replay for deterministic agent tests, litellm
instrumentation, and more waste rules.

## Contributing

The two easiest ways in: add an instrumentation adapter
([docs/instrumentation.md](https://github.com/TommyTranX/traceburn/blob/main/docs/instrumentation.md)) or a waste rule
([docs/waste-rules.md](https://github.com/TommyTranX/traceburn/blob/main/docs/waste-rules.md)). Both are designed to be an
afternoon of work. See [CONTRIBUTING.md](https://github.com/TommyTranX/traceburn/blob/main/CONTRIBUTING.md).

## Citation

If traceburn is useful in your work, see [CITATION.cff](https://github.com/TommyTranX/traceburn/blob/main/CITATION.cff).

MIT license.
