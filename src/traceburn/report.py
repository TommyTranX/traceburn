"""Standalone report with an explicit allowlist of shareable fields.

Raw names, IDs, models, timestamps, payloads, errors, and rule-generated
prose never enter the export. Rules inspect local payloads to find evidence;
only fixed labels and validated numeric metadata cross the export boundary.
"""
from __future__ import annotations

import math
from html import escape
from pathlib import Path

from .analyze import waste
from .schema import SPAN_KINDS
from .store import Store

GUIDANCE = {
    "duplicates": (
        "Repeated model requests",
        "Inspect the linked calls locally to check whether the requests and answers repeat.",
        "Try reusing a result or moving the call out of the loop when its inputs are stable.",
        "Repeated sampling can be intentional. Check task outcomes before removing a call.",
    ),
    "cache": (
        "Potential prompt-cache reuse",
        "Inspect the linked prompts for repeated prefixes and their recorded cache usage.",
        "Keep stable instructions together and check the provider's cache requirements.",
        "Cache eligibility, expiry, minimum prefix length, and pricing vary by provider.",
    ),
    "context_bloat": (
        "Growing or large context",
        "Inspect how the prompt changes across the linked calls.",
        "Try shortening repeated history or retrieving only the context needed for the task.",
        "Large context can be necessary. Test answer quality after changing it.",
    ),
    "loops": (
        "Repeated work or a long run",
        "Inspect the linked calls for repeated arguments, failed attempts, or a long sequence.",
        "Check stop conditions, retry limits, and whether deterministic tool results can be reused.",
        "Polling and retries can be necessary. Verify the underlying condition before changing them.",
    ),
    "model_overkill": (
        "Candidate for a smaller-model experiment",
        "Inspect the short responses in the linked calls and the task they perform.",
        "Evaluate a smaller model on representative inputs and compare cost and task success.",
        "A short answer does not prove a task is easy. This finding is a hypothesis to test.",
    ),
}


def number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            if math.isfinite(value) and value >= 0:
                return value
        except OverflowError:
            pass
    return None


def _display(value, *, money=False):
    value = number(value)
    if value is None:
        return "unknown"
    return f"${value:.4f}" if money else f"{value:,.0f}"


def report_data(store: Store, trace_id: str) -> dict:
    """Return allowlisted metadata, with local call numbers replacing raw IDs."""
    trace = store.get_trace(trace_id)
    if trace is None:
        raise ValueError("trace not found")
    spans = store.get_spans(trace_id, hydrate=False)
    labels = {span.span_id: i + 1 for i, span in enumerate(spans)}
    calls = []
    for span in spans:
        attrs = span.attributes
        calls.append({
            "call": labels[span.span_id],
            "parent": labels.get(span.parent_id),
            "kind": span.kind if span.kind in SPAN_KINDS else "custom",
            "status": span.status if span.status in ("ok", "error") else "unknown",
            "duration_ms": number(span.duration_ns / 1e6) if span.duration_ns is not None else None,
            "input_tokens": number(attrs.get("gen_ai.usage.input_tokens")),
            "cached_tokens": number(attrs.get("cached_input_tokens")),
            "cache_write_tokens": number(attrs.get("cache_write_tokens")),
            "output_tokens": number(attrs.get("gen_ai.usage.output_tokens")),
            "estimated_cost_usd": number(attrs.get("cost_usd")),
        })
    llm_calls = [c for c in calls if c["kind"] == "llm"]
    unknown = sum(c["estimated_cost_usd"] is None for c in llm_calls)
    cost = sum(c["estimated_cost_usd"] or 0 for c in llm_calls)
    findings = []
    for finding in waste.report(store, trace_id)["findings"]:
        rule = finding["rule_id"]
        if rule not in GUIDANCE:
            continue
        findings.append({
            "rule": rule,
            "calls": [labels[s] for s in finding["span_ids"] if s in labels],
            "potential_savings_usd": number(finding.get("avoidable_usd")),
            "confidence": finding["confidence"] if finding["confidence"] in ("low", "medium", "high") else "unknown",
        })
    return {"calls": calls, "findings": findings, "known_cost_usd": cost,
            "unknown_cost_calls": unknown, "llm_calls": len(llm_calls)}


def render_report(store: Store, trace_id: str, *, synthetic: bool = False) -> str:
    data = report_data(store, trace_id)
    cards = []
    for finding in data["findings"]:
        title, evidence, action, caveat = GUIDANCE[finding["rule"]]
        links = ", ".join(f'<a href="#call-{i}">Call {i}</a>' for i in finding["calls"])
        amount = finding["potential_savings_usd"]
        potential = (f"<p>Potential savings: {_display(amount, money=True)}. Estimate only; "
                     "findings may overlap and should not be added together.</p>") if amount else ""
        cards.append(f'<article><h3>{title}</h3><p class="eyebrow">'
                     f'{len(finding["calls"])} linked calls | {finding["confidence"]} confidence</p>'
                     f'<p>{evidence}</p><p>{links or "Applies to the whole run."}</p>'
                     f'<p><strong>Try next:</strong> {action}</p><details><summary>When this may be expected</summary>'
                     f'<p>{caveat}</p></details>{potential}</article>')
    rows = []
    for call in data["calls"]:
        parent = f'Call {call["parent"]}' if call["parent"] else "root"
        duration = f'{call["duration_ms"]:,.1f}' if call["duration_ms"] is not None else "unknown"
        cells = [f'Call {call["call"]}', parent, call["kind"], call["status"], duration,
                 _display(call["input_tokens"]), _display(call["cached_tokens"]),
                 _display(call["cache_write_tokens"]), _display(call["output_tokens"]),
                 _display(call["estimated_cost_usd"], money=True) if call["kind"] == "llm" else "n/a"]
        rows.append(f'<tr id="call-{call["call"]}">' + "".join(f"<td>{escape(v)}</td>" for v in cells) + "</tr>")
    label = "Synthetic demo" if synthetic else "Trace report"
    demo = ('<aside>This is a synthetic example with invented tokens, durations, and costs. '
            'It demonstrates a repeated request; it is not a measured benchmark or a savings claim.</aside>') if synthetic else ""
    unknown = data["unknown_cost_calls"]
    cost_label = "Known-cost subtotal" if unknown else "Estimated model cost"
    cost_note = f'{unknown} model call(s) have unknown cost.' if unknown else "Costs are estimates from recorded metadata."
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>TraceBurn | {label}</title><style>
:root{{color-scheme:light;--ink:#182334;--muted:#55657a;--accent:#146655;--line:#dce5e6}}
*{{box-sizing:border-box}}body{{margin:0;background:#f5f8f8;color:var(--ink);font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1080px;margin:auto;padding:48px 24px}}.brand,.eyebrow{{font-size:13px;letter-spacing:.05em;color:var(--accent);font-weight:700}}
h1{{font-size:clamp(32px,5vw,48px);line-height:1.1;margin:12px 0}}h2{{margin-top:36px}}h3{{margin:0;font-size:21px}}p{{margin:12px 0}}
.muted,footer{{color:var(--muted)}}aside{{background:#e6f2ed;border-left:4px solid var(--accent);padding:16px;margin:24px 0}}
.metrics{{display:flex;gap:16px;flex-wrap:wrap;margin:24px 0}}.metric{{flex:1;min-width:160px;border:1px solid var(--line);background:white;padding:20px;border-radius:12px}}
.metric strong{{display:block;font-size:30px}}article{{background:white;border:1px solid var(--line);border-radius:12px;padding:24px;margin:16px 0}}
a{{color:var(--accent)}}summary{{cursor:pointer;color:var(--muted)}}.scroll{{overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:white}}
table{{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}}th,td{{text-align:left;padding:12px;border-bottom:1px solid var(--line)}}tr:target{{background:#e6f2ed}}
footer{{font-size:14px;margin-top:32px;border-top:1px solid var(--line);padding-top:20px}}
</style></head><body><main><div class="brand">TRACEBURN / {label.upper()}</div>
<h1>Where did this run spend?</h1><p class="muted">Inspect the calls. Test a change. Check that the task still succeeds.</p>{demo}
<div class="metrics"><div class="metric">{cost_label}<strong>{_display(data['known_cost_usd'], money=True)}</strong></div>
<div class="metric">Model calls<strong>{data['llm_calls']}</strong></div><div class="metric">Findings to investigate<strong>{len(cards)}</strong></div></div>
<p class="muted">{cost_note} Cost changes alone do not establish equivalent task quality.</p>
<h2>What to investigate</h2>{''.join(cards) or '<article>No patterns flagged by the current rules. This does not establish that the run is optimal.</article>'}
<h2>Calls and recorded usage</h2><p class="muted">Call numbers are local to this report. Input excludes cache reads and writes, which have separate columns. Missing usage is shown as unknown.</p>
<div class="scroll"><table><thead><tr>{''.join('<th>'+h+'</th>' for h in ['Call','Parent','Kind','Status','Time (ms)','Input','Cache read','Cache write','Output','Est. cost'])}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<footer><strong>Export scope:</strong> numeric usage, timing, cost, topology, fixed status labels, and findings categories only. Prompts, responses, tool arguments, names, models, raw IDs, timestamps, errors, and paths are omitted. Numeric metadata can still be sensitive; review before sharing. The source database retains its recorded payloads. This standalone file contains no scripts, external assets, or telemetry.</footer>
</main></body></html>'''


def write_report(store: Store, trace_id: str, output: str | Path, *, synthetic=False, force=False) -> Path:
    path = Path(output).expanduser().resolve()
    if path == Path(store.path).expanduser().resolve():
        raise ValueError("report output must not overwrite the trace database")
    html = render_report(store, trace_id, synthetic=synthetic)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if force else "x", encoding="utf-8") as file:
        file.write(html)
    return path
