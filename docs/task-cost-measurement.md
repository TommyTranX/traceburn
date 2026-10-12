# Measure cost per successful agent task

An inexpensive model call is not necessarily an inexpensive completed task.
This guide joins independently checked outcomes to all model/API spend in a
declared workload, including unsuccessful attempts and retries. It works with
TraceBurn traces or records exported from another system.

## Run the examples without API keys

The `outcomes` command is in the development source, after PyPI version 0.2.0.
From a checkout of this repository, using Python 3.10 or later:

```bash
PYTHONPATH=src python3 -m traceburn.cli outcomes examples/task_costs/retries.csv --manifest examples/task_costs/retries.manifest.json
PYTHONPATH=src python3 examples/task_costs/reproduce.py
PYTHONPATH=src python3 -m traceburn.cli outcomes examples/task_costs/realm_attempts.jsonl --manifest examples/task_costs/realm.manifest.json --format csv
```

These commands only read local files and print reports. They do not call a model
or incur API charges. Use `--format json` for machine-readable results. To install
the development CLI in a virtual environment, run `python3 -m pip install .` from
the checkout. The examples are repository files, not bundled wheel resources.

There are two deliberately different examples:

- **Synthetic retry example:** invented costs and outcomes. Three tasks, four
  attempts, two successful tasks. Total spend is $0.12, so cost per success is
  $0.06. Failed attempts account for $0.06, including one task that later passed.
  The cheaper-looking condition spends $0.09 but solves only one task, so its
  cost per success is $0.09. This tests accounting, not agent behavior.
- **Recorded study reanalysis:** 288 existing model runs, representing baseline
  and 600-character tool-response truncation on the same 144 synthetic tasks.
  The report recomputes recorded usage cost and imports the original harness's
  independent grades. SDK-internal retry billing is not observable in the saved
  logs, so full cost per success is **undefined**, not silently treated as complete.
  See the [data, provenance, and limitations](../examples/task_costs/README.md).

## Define the measurement before collecting data

1. Declare a fixed cohort of task IDs and the conditions to compare. Use the same
   workload, stopping rule, attempt budget, model settings, and quality test in
   each condition. Keep the random seed and environment version with the run.
2. Define success outside the agent: an exact answer, verified state change,
   passing test, or reviewed artifact against a written rubric. A normal API
   response, an agent saying "done," or a span status of `ok` is not success.
3. Record every attempt, including timeout, cancellation, refusal, and exhausted
   budget. Grade these according to the declared task requirement; use
   `ungraded` if the evidence is unavailable. Never drop failed attempts.
4. Capture every billable call. Disable SDK-internal retries during a controlled
   experiment, or collect their usage separately and reconcile billing. An
   absent usage record is unknown cost. Instrumentation alone cannot establish
   complete capture, and disabled retries do not guarantee every failed request
   has observable billing.
5. Set the quality floor before inspecting results. Compare cost only alongside
   success rate, failure severity, latency, and human intervention. A lower cost
   is not an improvement if the required quality floor is missed.

For a fixed cohort:

```text
cost per successful task = sum(model/API cost of ALL attempts)
                           / number of distinct tasks that pass
```

A retrying task is counted once in the denominator. Every billable retry remains
in the numerator. If all tasks fail, the metric is undefined, not zero. If any
task is missing, outcome ungraded, or billing incomplete, this implementation
also leaves the metric undefined and reports the reason. A known zero cost is
different from an unknown cost.

## Portable input contract

Use one JSON object per line or the same named columns in CSV:

| Field | Meaning |
| --- | --- |
| `condition` | Identifier such as `baseline` or `candidate`. |
| `task_id` | Logical task identifier shared across conditions. |
| `attempt` | Positive integer starting at 1; restarted attempts increment it. |
| `outcome` | `pass`, `fail`, or `ungraded`, determined independently. |
| `cost_usd` | Nonnegative known model/API spend for this attempt, or null. |
| `billing_complete` | Explicit assertion that every billable model call is accounted for. |
| `model_calls` | Count of observed model calls, or null if unknown. |
| `retry_calls` | Known subset of calls that are retries, or null if unknown. |
| `latency_s` | Elapsed attempt duration in seconds, or null if unknown. |

In CSV, blank numeric cells mean unknown and boolean values are `true` or
`false`. In JSONL, use `null` for unknown. An incomplete record can retain its
known subtotal with `billing_complete: false`. Missing rows are detected using
the manifest, not guessed from observed data. Identifiers contain letters,
digits, dots, hyphens, or underscores and start with a letter or digit.

The [example manifest](../examples/task_costs/retries.manifest.json) specifies
`schema_version: 1`, `conditions`, `task_ids`, a written `quality_check`, a
`pricing_ref`, `cost_scope: "model_api"`, `cost_basis` (`estimated` or `invoiced`),
and `evidence` (`measured`, `synthetic`, or `replayed`). Use separate manifests
for different billing bases or evidence types. Preserve the manifest with outputs.

No attempt may follow a successful attempt. A separate repeated trial gets a new
task ID, such as `ticket-17-rep2`. An ungraded earlier attempt prevents the tool
from classifying the logical task, even if a later attempt passes, because the
earlier attempt might already have succeeded. Human-assisted and autonomous
outcomes should use separate conditions or cohorts with explicit quality rules.

The JSON and CSV reports include spend on failed attempts, restarted attempts,
known call counts, missing tasks, ungraded outcomes, and undefined reasons.
`summed_attempt_latency_s` adds durations; it is not end-to-end batch wall time
when attempts run concurrently. Unknown retry counts do not mean zero retries.

## Join TraceBurn traces to an independent evaluator

```python
import json
from traceburn.store import Store
from traceburn.outcomes import attempt_from_trace

store = Store(".traceburn/traces.db")
try:
    row = attempt_from_trace(
        store, trace_id="YOUR_TRACE_ID",
        condition="baseline", task_id="ticket-17", attempt=1,
        outcome="pass",  # Replace with your independent evaluator's result.
        capture_complete=False,  # Set true only after checking full usage capture.
        retry_calls=None,  # Replace when your runner counts retries.
    )
    print(json.dumps(row))
finally:
    store.close()
```

Create one trace per attempt and finish it before exporting. The helper sums LLM
span costs only, avoiding double-counting parent/tool spans. A missing model
price or invalid cost forces incomplete billing regardless of the caller's
assertion. Keep every failed call in the trace. If a retry receives no usage,
reconcile it before setting `capture_complete=True`. Save the exported row as
JSONL and run `traceburn outcomes` with your manifest. The helper imports no
grader and cannot determine task success for you.

## Pricing and boundaries

Preserve provider, model version, uncached input, cache reads/writes, output,
pricing date, endpoint, and discounts with raw records. The bundled table has
per-model verification dates where refreshed; the remaining entries retain
the original snapshot date. New estimated spans record `cost_basis`,
`pricing_as_of`, and `pricing_source`. Previously recorded costs are not
repriced. Use `TRACEBURN_PRICING` for your rates and keep that file with the run.

The reference report covers model/API costs only. Tool fees, infrastructure,
cache storage, human review, and business harm are excluded. Do not label it
total cost of ownership or ROI. To measure those, maintain a separate reconciled
ledger with explicit allocations and no double-counting. Price overrides may be
needed for long context, nonstandard service tiers, regions, and cache duration.

Descriptive cohort ratios do not establish statistical equivalence. For an
optimization decision, use paired tasks, repeated runs, predeclared quality
thresholds, and uncertainty estimates appropriate to the sampling design.
This first implementation does not calculate confidence intervals or make
an automated "ship this optimization" decision.

## Relationship to other measurement work

The [Tokenomics Big-T framework](https://www.tokeneconomics.com/docs/projects/big-t/big-t-notation-paper/)
describes how token consumption grows with workload characteristics. This guide
addresses a complementary empirical question: how much recorded spend produces
successful tasks under a stated quality check?
The [Tokenomics cache explainer](https://www.tokeneconomics.com/cache-explainer/)
shows why cache billing categories matter, while its
[AI Value Classification](https://www.tokeneconomics.com/ai-value-classification/)
discusses cost and value boundaries. These are references, not endorsements or
claims of adoption. TraceBurn's implementation and example results are independent.
