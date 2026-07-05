# traceburn

## The finding that made me build this

I wrote a small support ticket triage agent: five tickets, one tool call each, a roughly 4,700
token static policy prompt sent fresh on every call, model claude-haiku-4-5. Running it uncached
cost $0.0539. traceburn's waste report looked at the trace, noticed that same 4,700 token prefix
going out uncached on all 10 calls, and estimated that about 82 percent of that spend was
avoidable. So I added exactly the one cache_control block it suggested and reran the same five
tickets: $0.0167.

That's a 69 percent measured saving. The tool's estimate landed within 18 percent of what actually
happened, close enough to trust as a first signal, not close enough to treat as gospel. That's
roughly how I want a cost estimator to behave.

Prices change and models get repriced, so do not take my word for it: the reproduction is a few
cents and a couple of minutes, at
[examples/cache_before_after.py](https://github.com/TommyTranX/traceburn/blob/main/examples/cache_before_after.py).
Measured 2026-07-05.

That's the whole pitch in one run. traceburn is a local-first tracer and efficiency profiler for
AI agents, built on top of the openai and anthropic Python SDKs: a cost and latency flamegraph, a
waste report that quantifies avoidable spend instead of just gesturing at it, deterministic replay
of recorded calls, and run diffs, all backed by one SQLite file on disk. No account, no server to
stand up for basic use, no telemetry leaving your machine.

## Install

Core tracer, stdlib only, zero dependencies:

```bash
pip install traceburn
```

Tracer plus the local web viewer (adds starlette and uvicorn):

```bash
pip install "traceburn[ui]"
```

Requires Python 3.10 or later.

## Quickstart

Patch the SDKs at the top of your agent:

```python
import traceburn
traceburn.install()

# your existing openai / anthropic code, unchanged
```

Every sync call, async call, streaming response, tool call, and prompt cache hit on either SDK now
gets recorded as a span. Traces live in one SQLite file, `./.traceburn/traces.db` by default; set
the `TRACEBURN_DB` environment variable if you want it somewhere else. If you'd rather not touch
the source at all, wrap the run instead:

```bash
TRACEBURN=1 python your_agent.py
```

Then look at what happened:

```bash
traceburn ui             # opens the web viewer at 127.0.0.1:8765
traceburn ls              # list recorded traces
traceburn show <id>       # inspect one trace
traceburn waste <id>      # run the waste report on one trace
traceburn diff <a> <b>    # compare two traces span by span
```

No API keys and nothing to configure: `python examples/offline_demo.py` records a simulated agent
run with realistic token counts, including one deliberate duplicate call, so you can see a real
trace, a real flamegraph, and a real waste finding inside a minute.

## What it actually does

**Trace.** `traceburn.install()` patches both SDKs so every call becomes a span with tokens,
latency, and cost attached, no code changes required past that one line. Want manual control
instead, or you're using a framework outside the two supported SDKs? The explicit API, `@trace`,
`span()`, and `session()`, works by hand with anything.

**Flamegraph.** Spans render as a flamegraph you can size two ways: by wall-clock time or by
dollars spent, with self-time kept separate from time spent in children, so a slow parent span
doesn't hide which child call actually burned the seconds or the money.

**Waste report.** Heuristics look for duplicate calls, unused cache opportunities, bloated
prompts, model overkill, and retry loops. Each finding ships with a confidence level and, where the
numbers support it, a dollar figure. More on this below.

**Replay.** `traceburn.analyze.replay.replay()` plays recorded provider responses back through the
real SDK types, so your agent code runs again exactly as before with zero tokens spent. That's
useful for tests and for debugging without burning a budget. Streaming calls aren't replayable
yet; replay currently only serves non-streaming recordings.

**Diff.** `traceburn diff <a> <b>` lines up two traces span by span and reports the delta in
tokens, cost, and latency, alongside text diffs of the prompts and responses that changed between
the runs. Good for answering "did that prompt tweak actually help."

## The web viewer

`traceburn ui` starts a small, read-only Starlette API in front of a vendored single-page app: no
build step, no CDN, everything ships in the package. It gives you the trace tree, the flamegraph,
a waterfall timeline, the waste report, and the run diff view.

It binds to 127.0.0.1 only and checks the request's host header against DNS rebinding, but it has
no authentication of any kind. That's a deliberate tradeoff: the viewer isn't meant to be reachable
from anywhere but your own machine.

## Framework support

Today, that means the raw `openai` and `anthropic` Python SDKs, patched automatically by
`traceburn.install()`. If you're on something else, the explicit `span()` / `trace()` / `session()`
API works with any framework right now, by hand, since it doesn't care what's making the call.
LangChain, LlamaIndex, and anything already emitting OpenTelemetry GenAI spans aren't instrumented
automatically yet. That's real, planned work for v0.2, not something already built and just
undocumented, and it's covered in the roadmap below.

## How the waste rules work

The rules live in `traceburn/analyze/waste/` and are documented in full at
[docs/waste-rules.md](https://github.com/TommyTranX/traceburn/blob/main/docs/waste-rules.md). Five
kinds of waste get checked for: duplicates (exact and near-duplicate repeated calls), cache (a
stable prompt prefix resent without ever hitting a provider cache), context_bloat (duplicate
blocks inside one prompt, or a huge prompt for a tiny output), model_overkill (a frontier-priced
model spent on trivial short calls, phrased as a suggestion and kept at low confidence on purpose),
and loops (retry storms, repeated identical tool calls, or a runaway step count). Every finding
also carries a confidence level: high, medium, low, or info.

Two principles govern all of them. A wrong finding does more damage than a missed one, so the
rules are tuned for precision over recall and would rather stay quiet than guess. And no dollar
figure is ever printed without observed tokens behind it and a price in the pricing table to
multiply against; everything the tool prints is labeled an estimate, because it is one.

## Privacy

traceburn makes no network calls of its own and sends no telemetry anywhere. The only traffic on
the wire is your own calls to your own model provider, exactly as they'd happen without traceburn
installed. A test,
[tests/test_no_network.py](https://github.com/TommyTranX/traceburn/blob/main/tests/test_no_network.py),
blocks all socket access at the interpreter level and then runs the recorder, the store, every
analyzer, and the CLI against that blockade, to prove the point rather than just assert it. Auth
headers are never recorded in a trace, so an API key cannot end up sitting in a stored span.

## Limitations

Instrumentation currently covers only the `openai` and `anthropic` Python SDKs. Within those,
`parse()` convenience methods and `with_raw_response` calls pass through untraced rather than being
recorded incorrectly, and multi-choice requests (`n > 1`) only record the first choice.

Cost figures come from a dated public pricing table
([pricing.json](https://github.com/TommyTranX/traceburn/blob/main/src/traceburn/pricing.json)) and
don't model long-context pricing tiers or regional surcharges. Token counts prefer whatever the
provider itself reports as usage; anything estimated is flagged as estimated rather than presented
as measured.

The waste rules are heuristics tuned for precision over recall, which means they'll miss real
waste sooner than they'll invent fake waste, and every finding states its own confidence so you
can judge it accordingly. Streaming calls aren't replayable yet; only non-streaming recordings are.

The web viewer is read-only, bound to 127.0.0.1 only, and has no authentication.

## Roadmap: v0.2

- OpenTelemetry GenAI span ingest plus OTLP export. This is also the path for capturing LangChain
  and LlamaIndex traces, since it rides on their existing OTel instrumentation rather than
  requiring bespoke adapters for each.
- A pytest plugin built on replay, for deterministic, token-free agent tests.
- litellm instrumentation.
- More waste rules.

## Related work

[Langfuse](https://github.com/langfuse/langfuse) is a full open-source LLM platform: tracing,
evals, and prompt management, backed by Postgres and ClickHouse and meant to run as a server.

[Arize Phoenix](https://github.com/Arize-ai/phoenix) is the closest neighbor here. It runs locally
against SQLite with no account needed, and it's strong on tracing and evals, but it runs as a
server process with a fairly large dependency set, and it doesn't focus on waste detection, a
cost-weighted flamegraph, deterministic replay, or run diffs.

MLflow has been adding GenAI tracing, trace comparison, and efficiency scoring to its tracking
server; see [mlflow/mlflow](https://github.com/mlflow/mlflow).

[LangSmith](https://smith.langchain.com) is LangChain's hosted, proprietary platform.
[OpenLLMetry](https://github.com/traceloop/openllmetry) takes a different approach: it instruments
your code and exports OpenTelemetry spans to whatever backend you choose to point it at, rather
than shipping a backend of its own.

[AgentSight](https://github.com/eunomia-bpf/agentsight) renders token flamegraphs of coding agents
from the system side using eBPF, a genuinely different vantage point, though it's Linux only.

Helicone ([Helicone/helicone](https://github.com/Helicone/helicone)), OpenLIT
([openlit/openlit](https://github.com/openlit/openlit)), Braintrust
([braintrust.dev](https://braintrust.dev)), and Logfire
([pydantic.dev/logfire](https://pydantic.dev/logfire)) each pair instrumentation with a server or a
cloud backend of their own.

traceburn's own position is narrower than most of the above: strictly local, one file, no account
and no server needed for basic use, framework-agnostic at the SDK level, and built around
efficiency first, meaning the waste report, the dollar-weighted flamegraph, replay, and diff all
live together in one small package. It's meant to sit next to whatever observability stack you
already run, not replace it.

## Contributing

There are two extension points, each documented and each meant to be roughly an afternoon of work:
an [instrumentation adapter](https://github.com/TommyTranX/traceburn/blob/main/docs/instrumentation.md)
for a new SDK or framework, and a
[waste rule](https://github.com/TommyTranX/traceburn/blob/main/docs/waste-rules.md) for a new
pattern of avoidable spend. The full guide is at
[CONTRIBUTING.md](https://github.com/TommyTranX/traceburn/blob/main/CONTRIBUTING.md).

## Citation

A citation file is included at
[CITATION.cff](https://github.com/TommyTranX/traceburn/blob/main/CITATION.cff).

## License

MIT. Full text at
[LICENSE](https://github.com/TommyTranX/traceburn/blob/main/LICENSE).

Written by Tommy Tran (tommy.tranxhec@gmail.com).
