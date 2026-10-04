# Fixes and the CI gate

Two commands sit on top of the waste report: `traceburn fix`, which shows the
mechanical patch for a finding when one is honest to give, and
`traceburn check`, a threshold gate meant to run in CI.

## `traceburn fix <trace_id>`

Runs the waste report, then for each finding checks whether it carries
structured, machine-readable fix data (`Finding.fix` in the schema). If it
does, prints the literal code change. If it doesn't, the finding is left out
of the fix list entirely (`traceburn waste` still has the full explanation).

Two kinds of fix exist today, both narrow on purpose:

- **`cache` (anthropic only)**: when the shared, uncached prefix a `cache`
  finding flags is verifiably the caller's entire `system` string, not
  something that spans into the messages too, the fix shows the exact
  wrapping pattern: turn `system="..."` into a list with one text block
  carrying `cache_control: {"type": "ephemeral"}`. The pattern is shown
  generically, with a placeholder standing in for the real prompt text,
  since the transformation is what's actionable and a real prompt can be
  long or sensitive enough that dumping it to a terminal by default is the
  wrong call.
- **`model_overkill`**: the literal model-string swap, from the model the
  finding flagged to the provider's cheapest listed model. Still only a
  suggestion; the finding's own confidence is deliberately low, since
  quality on the cheaper model is something only a human can verify.

Every other rule (`duplicates`, `context_bloat`, `loops`) needs a decision
only visible in the caller's own source code (cache the result where? hoist
which call out of which loop?), so no fix is offered for them; inventing one
would mean guessing at code traceburn cannot see, which breaks the same
precision-first rule the waste rules themselves are held to.

## `traceburn check [trace_id]`

A threshold gate for CI: exits 0 if a trace's cost and waste stay within
bounds, exits 1 and prints why if not. Meant to run right after a test suite
records a traced agent call.

```bash
traceburn check                                    # the most recently recorded trace
traceburn check <trace_id> --max-cost 0.10          # fail if this run cost more than $0.10
traceburn check --max-avoidable-pct 20              # fail if over 20% of spend looks avoidable
traceburn check <trace_id> --baseline <baseline_id> --max-regression-pct 15
                                                     # fail if cost rose more than 15% vs baseline
```

If no `trace_id` is given, `check` uses the most recently recorded trace,
which is the common case: run your test, then call `traceburn check` right
after. `--baseline` compares against a second, named trace (say, one you
recorded and committed a reference for) rather than a fixed dollar figure,
useful for catching a prompt change that quietly made an agent step more
expensive.

Example CI step:

```yaml
- run: python your_test_that_records_a_trace.py
- run: traceburn check --max-cost 0.05 --max-avoidable-pct 25
```

`check` never modifies anything; it only reads the store and reports.

A cost check also exits nonzero when an LLM span has no valid recorded cost.
Unknown model prices or missing usage are insufficient data, not zero spend.
Non-LLM spans do not need a cost. Explicitly recorded zero costs remain valid.
A positive cost compared with a zero-cost baseline cannot produce a percentage;
that check fails with an explanation. Use an absolute `--max-cost` budget for
that case. Two confirmed zero-cost traces pass the relative comparison.
`--max-regression-pct` requires `--baseline`.
