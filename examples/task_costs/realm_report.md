# Cost per successful task

Evidence: measured. Cost basis: estimated. Scope: model/API only.

Quality check: Original harness check_success: normalized answer or numeric tolerance &lt;0.01, plus required file content where applicable. Runtime grades imported; final filesystem states are not in saved logs.

Pricing reference: Frozen recorded-usage estimate: claude-haiku-4-5-20251001, USD 1/M input and 5/M output. No recorded cache tokens. Not an invoice.

| Condition | Passed / planned | Success rate | Known spend | Total spend | Cost / success |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 142 / 144 | 98.61% | $1.708166 | undefined | undefined |
| truncate | 68 / 144 | 47.22% | $3.627289 | undefined | undefined |

Failed attempts remain in spend. A success is counted once per task.
Cost changes do not establish equivalent quality. These are descriptive results.
baseline: incomplete_billing.
truncate: incomplete_billing.
