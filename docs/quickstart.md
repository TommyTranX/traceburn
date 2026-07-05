# Quickstart

Ten minutes from install to a flamegraph and a waste finding. No account,
no server to configure.

## Install

```
pip install traceburn        # core: stdlib only
pip install "traceburn[ui]"  # adds the local web viewer
```

## See it work without an API key

```
python examples/offline_demo.py
traceburn ui
```

The demo records a simulated research agent with realistic token counts
(including a deliberate duplicate call), so the viewer has a trace tree, a
cost flamegraph, and a waste finding to show immediately.

## Instrument your own agent

```python
import traceburn
traceburn.install()
```

That is the whole integration for openai and anthropic SDK code: every
call is recorded with model, messages, usage, cache reads and writes,
latency, and an estimated cost. Add your own steps where the patchers
cannot see:

```python
from traceburn import trace, span, session

@trace("plan", kind="agent")
def plan(state): ...

with session("nightly-run"):
    with span("retrieve", kind="retrieval"):
        docs = retriever(query)
    answer = plan(state)
```

Or skip the code change entirely:

```
TRACEBURN=1 python your_agent.py
```

## Look at a run

```
traceburn ui                     # web viewer at 127.0.0.1:8765
traceburn ls                     # recent sessions and traces
traceburn show <trace-prefix>    # span tree in the terminal
traceburn waste <trace-prefix>   # the efficiency report
traceburn diff <a> <b>           # compare two runs
```

Traces land in `./.traceburn/traces.db` (override with `TRACEBURN_DB`).
It is one SQLite file; take it, copy it, query it with the sqlite3 CLI.

## Replay a run with zero tokens

```python
from traceburn.analyze.replay import replay

with replay(trace_id="<id>"):
    run_agent()      # every recorded call served from the local store
```

See [replay.md](replay.md) for matching semantics and limits.

## Next

- [concepts.md](concepts.md): the data model in five minutes
- [waste-rules.md](waste-rules.md): what each rule flags and why
- [instrumentation.md](instrumentation.md): add an adapter for another client
