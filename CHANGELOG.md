# Changelog

## 0.3.0.dev0 (development source, not a PyPI release)

- Add `traceburn outcomes` and a Python trace-to-attempt helper. Join independent
  task outcomes to all recorded model/API costs, including failed attempts and
  retries; import JSONL/CSV and export Markdown/JSON/CSV without a database.
- Add a measurement guide, a synthetic retry fixture, and an offline reanalysis
  of 288 existing model runs with explicit provenance and evidence limitations.
- Report incomplete cost as unknown in CLI waste reports, trace lists, and trace
  summaries. A known zero is displayed as zero. **API change:** waste report
  `total_cost_usd` can now be null; use `known_cost_usd` and `unknown_cost_calls`
  to distinguish a subtotal from a complete total. Consumers must handle null.
- Keep per-model price dates and sources. Recheck the existing Anthropic entries,
  GPT-4.1, and GPT-5.5 against provider pages on October 11, 2026. Other entries
  retain their July 4 snapshot dates. Historical recorded costs are unchanged.
- Clarify that the five-ticket cache example did not independently establish
  equivalent task quality.
