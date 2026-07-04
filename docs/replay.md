# Deterministic replay

Recording is automatic: with instrumentation installed, every non-streaming
LLM span stores the provider's exact response payload, deduplicated by
content hash, keyed by the hash of its normalized request. Replay serves
those responses back to your unmodified agent code, reconstructed as real
SDK objects, with no network and no tokens spent.

```python
import traceburn
from traceburn.analyze.replay import replay

traceburn.install()
run_agent()                       # recorded run, real API calls

with replay(trace_id="<id>"):
    run_agent()                   # identical run, zero tokens
```

## Matching

A request matches a recording when its normalized form hashes identically:
same model, same messages, same allowlisted parameters. The normalization
is the same code path the recorder uses, so anything the recorder captured
the replayer can match. Changing the prompt, the model, or a sampling
parameter changes the hash, and the call becomes a miss.

Repeats of an identical request play back the recorded responses in order.
When the recordings for a hash run out, the last one repeats: an identical
request is assumed to keep getting the same answer.

## Misses

`on_miss="raise"` (default) raises `ReplayMiss` naming the endpoint. This is
what you want in tests: a miss means your agent sent something the
recording never saw, which is itself a signal the behavior changed.
`on_miss="passthrough"` performs the real call instead, useful when
iterating on one step of a larger recorded run.

## Limitations

- Streaming requests are not replayable in this version; they follow the
  `on_miss` policy. Record the call non-streaming to replay it.
- Replayed calls do not write new spans; the replayed run is not itself a
  recorded trace.
- Recordings made before response payloads were stored (or with capture
  errors) are skipped at index build time; `Replayer.recorded_calls` says
  how many calls are servable.
