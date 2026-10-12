# Task outcome and cost examples

Start with the [measurement guide](../../docs/task-cost-measurement.md).
Everything in this directory runs offline. No API key or model call is needed.

## Synthetic retry fixture

`retries.csv` and `retries.manifest.json` are a small portable template with
invented costs and labels. A restarted attempt stays attached to its logical
task; its failed first attempt still contributes to cost. This demonstrates
the computation, not empirical savings or agent performance.

```bash
PYTHONPATH=src python3 -m traceburn.cli outcomes examples/task_costs/retries.csv --manifest examples/task_costs/retries.manifest.json
```

Expected: baseline passes 2/3 tasks, spends $0.12, and costs $0.06 per success.
Candidate passes 1/3, spends $0.09, and costs $0.09 per success.

## Reanalysis of recorded model runs

The data comes from Tommy Tran's study, **Where the Tokens Go: A Measured Anatomy
of LLM Agent Trajectory Costs**, listed among the
[accepted REALM workshop papers](https://realm-workshop.github.io/accepted_papers/).
These are historical measured model responses on synthetic tasks, not new runs,
production adoption evidence, or a claim that TraceBurn collected the original data.

The selection includes every baseline and `truncate` row from repeat 0: 144
paired tasks, 288 attempts, three task families, a 12-step cap, and one model,
`claude-haiku-4-5-20251001`. Truncation limits tool responses to 600 characters.
No rows were selected according to outcome. Additional repeated baseline runs
and other conditions are outside this comparison.

```bash
PYTHONPATH=src python3 examples/task_costs/reproduce.py
```

This checks the observations' SHA-256, reconciles per-step token totals with
per-run totals, recomputes the original price estimate ($1/M input, $5/M output),
imports the runtime grades, and verifies the committed derived reports. It does
not invoke the agent or independently rerun the original grader. Use `--write`
only when intentionally regenerating the four derived files after reviewing a change.

Inspect the [rendered report](realm_report.md), [JSON](realm_report.json),
[CSV](realm_report.csv), and [provenance](provenance.json).
`realm_observations.jsonl` retains token usage, model IDs, answers, and runtime
grades. `realm_attempts.jsonl` is the portable input generated from those records.

### What can and cannot be concluded

- Baseline passed 142/144 tasks; truncation passed 68/144. The same workload and
  original grade rule are used for both. This is a one-model, bounded synthetic
  study; it does not establish that all compression harms all agents.
- Recorded usage estimates include unsuccessful runs and all saved model steps.
  They are not invoices. The original client enabled up to five SDK-internal
  retries, which the saved step logs do not expose. `billing_complete` is therefore
  false, retry counts are unknown, and full cost per success is undefined.
- The [original grader excerpt](original_grader.py) checked a normalized answer
  or numeric tolerance under 0.01, plus exact required file content for tasks
  that wrote files. The exported grades were computed outside the model at run
  time. Final filesystem state was not saved in these observations, so the full
  grade cannot be independently reconstructed from this export.
- Step latency is not complete attempt duration. It is left unknown rather than
  presenting summed API latency as end-to-end latency. No production cost,
  business impact, or quality-equivalence claim follows from this reanalysis.

The explicit missing evidence is part of the example. A new controlled study
should save final state, disable or expose SDK retries, log every attempt and
usage gap, and reconcile billing before claiming complete cost per success.
