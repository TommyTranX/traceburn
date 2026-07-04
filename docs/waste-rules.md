# Waste rules

`traceburn waste <trace_id>` runs every rule in `traceburn/analyze/waste/`
over one trace and prints a ranked report. This page documents what each
rule detects, when it stays silent, and how the numbers are computed.

Two principles govern every rule:

1. Precision over recall. A finding that is wrong erodes trust faster than
   a missed finding builds it. Rules are tuned to stay silent unless the
   waste is real and visible in the recorded spans.
2. No invented numbers. Every dollar figure derives from the observed token
   counts and the bundled pricing table (dated, overridable, documented in
   `pricing.json`). When pricing is unknown, the rule reports tokens or
   seconds instead of dollars. All figures are labeled estimates.

Every finding carries: the rule id, a severity (`high`, `medium`, `low`,
`info`), the offending span ids, a plain-language explanation, quantified
avoidable tokens, dollars, or seconds where defensible, and a `confidence`
field (`high`, `medium`, `low`) saying how likely the flagged waste is real.
The `Finding` JSON schema is a public interface (see `schema.py`).

## duplicates

Fires when the same normalized request was sent more than once and every
attempt succeeded. Exact duplicates share a request hash; the repeats are
counted at their recorded cost, confidence high. Near-duplicates share a
model and at least 90 percent of their prompt vocabulary (token-set Jaccard)
without sharing a hash; confidence medium.

Silent when: requests differ, repeats involve errors (that is a retry storm,
see loops), or prompts are under 100 tokens (too small to matter and too
easy to collide).

## cache

Fires when two or more calls to the same model, within a ten-minute window,
share a stable prefix of at least 1024 estimated tokens and none of them
read from or wrote to a prompt cache. The saving is the repeats' prefix
tokens priced at the input rate minus the cached rate; where the provider
bills cache writes, the first call's write premium is subtracted. If the
pricing table has no cache rates for the model, no dollar figure is claimed.

Silent when: any call in the window used the cache, the shared prefix is
short, the calls are far apart, or there is only one call.

## context_bloat

Two checks. First, duplicate blocks inside one request: the same text block
(200+ characters) sent twice in a single prompt, usually history
re-appending or a retriever returning the same chunk twice; the repeated
tokens are priced at the model's input rate, confidence high. Second,
prompt dominance: a call with at least 8000 prompt tokens producing under
2 percent of that in output; informational only, no dollar claim, because
how much context the output actually needed is a judgment call.

## model_overkill

Fires when a group of small calls (at most 800 prompt tokens, at most 150
output tokens, no tool use) ran on a model priced at least five times above
the provider's cheapest listed model, and moving them would save at least
$0.0005. The figure is what the same tokens cost on the cheapest model, so
it is an upper bound. Deliberately low confidence and phrased as a
suggestion: quality on the cheaper model must be verified by a human.

## loops

Three checks. Retry storms: identical requests attempted more than once
with at least one error among them; reports the wall-clock burned by failed
attempts, and dollars only if the error spans carry cost. Repeated tool
calls: a tool span with identical recorded arguments three or more times.
Runaway step count: more than 50 LLM calls in one trace, informational.

## Adding a rule

A rule is one module in `traceburn/analyze/waste/` with a single entry
point:

```python
from ...schema import Finding
from ._common import RuleContext

RULE_ID = "my_rule"

def run(ctx: RuleContext) -> list[Finding]:
    ...
```

`RuleContext` gives you the trace id, all spans (hydrated, so request and
response payloads are loaded), the LLM spans, and the pricing table.
Helpers for provider, model, cost, token counts, and message-text
extraction live in `_common.py`.

Requirements for a merged rule:

- a positive test (the rule fires on a constructed case) and a negative
  test (it stays silent on a legitimate case) in `tests/test_waste.py`
- every dollar figure derived from observed tokens and the pricing table
- a `confidence` value that honestly reflects false-positive risk
- an entry in `ALL_RULES` in `waste/__init__.py` and a section on this page

A rule that raises is skipped and logged; it never takes the report down.
