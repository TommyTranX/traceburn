"""Terminal viewer: the fast path into a trace database.

    traceburn ui                   open the web viewer at 127.0.0.1:8765
    traceburn ls                   recent sessions and traces
    traceburn show <trace_id>      span tree with timing, tokens, and cost
    traceburn waste <trace_id>     efficiency report with avoidable spend
    traceburn fix <trace_id>       mechanical patches for fixable findings
    traceburn check [trace_id]     CI gate: exit nonzero on a cost or waste threshold
    traceburn diff <a> <b>         compare two runs step by step

Trace ids can be abbreviated to any unique prefix. The database path comes
from --db, the TRACEBURN_DB environment variable, or ./.traceburn/traces.db,
in that order. Stdlib only; no server, no network.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from . import __version__
from .schema import Span, Trace
from .store import Store, default_db_path


def _format_duration(ns: int | None) -> str:
    if ns is None:
        return "-"
    ms = ns / 1e6
    if ms < 9.95:
        return f"{ms:.1f}ms"
    if ms < 999.5:
        return f"{ms:.0f}ms"
    seconds = ms / 1_000
    if round(seconds, 1) < 60:
        return f"{seconds:.1f}s"
    total_seconds = round(seconds)
    minutes, remainder = divmod(total_seconds, 60)
    return f"{minutes}m{remainder}s"


def _format_cost(cost: float | None) -> str:
    if not cost:
        return "-"
    if cost >= 1:
        return f"${cost:.2f}"
    return f"${cost:.4f}"


def _trace_duration(trace: Trace) -> int | None:
    if trace.end_ns is None:
        return None
    return trace.end_ns - trace.start_ns


def _resolve_trace(store: Store, prefix: str) -> Trace:
    if prefix == "latest":
        traces = store.list_traces(limit=1)
        if not traces:
            raise SystemExit("no traces recorded yet")
        return traces[0]
    exact = store.get_trace(prefix)
    if exact is not None:
        return exact
    matches = store.find_traces(prefix)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(f"no trace found matching {prefix!r} (try: traceburn ls)")
    ids = ", ".join(t.trace_id[:12] for t in matches[:8])
    raise SystemExit(f"trace id prefix {prefix!r} is ambiguous: {ids}")


def cmd_ls(store: Store, args: argparse.Namespace) -> None:
    sessions = store.list_sessions(limit=args.limit)
    if sessions:
        print("sessions")
        for sess in sessions:
            count = store.count_traces(sess.session_id)
            print(
                f"  {sess.session_id}  {sess.name:<28}  "
                f"{count} traces  {_format_duration(sess.end_ns - sess.start_ns if sess.end_ns else None)}"
            )
        print()
    traces = store.list_traces(limit=args.limit)
    if not traces:
        print("no traces recorded yet")
        return
    print("traces")
    print(f"  {'id':<14} {'name':<28} {'spans':>5} {'llm':>4} {'errors':>6} {'time':>8} {'est. cost':>10}")
    for trace in traces:
        stats = store.trace_stats(trace.trace_id)
        print(
            f"  {trace.trace_id[:12]:<14} {trace.name[:28]:<28} "
            f"{stats['span_count']:>5} {stats['llm_count'] or 0:>4} "
            f"{stats['error_count'] or 0:>6} "
            f"{_format_duration(_trace_duration(trace)):>8} "
            f"{_format_cost(stats['cost_usd']):>10}"
        )


def _span_line(span: Span, depth: int) -> str:
    indent = "  " * depth
    duration = _format_duration(span.duration_ns)
    label = f"{indent}{span.name}"
    extra = ""
    if span.kind == "llm":
        attrs = span.attributes
        input_tokens = attrs.get("gen_ai.usage.input_tokens", 0) or 0
        cached = attrs.get("cached_input_tokens", 0) or 0
        written = attrs.get("cache_write_tokens", 0) or 0
        output_tokens = attrs.get("gen_ai.usage.output_tokens", 0) or 0
        token_bits = [f"{input_tokens + cached + written}in"]
        if cached:
            token_bits.append(f"{cached}cached")
        if written:
            token_bits.append(f"{written}cachew")
        token_bits.append(f"{output_tokens}out")
        extra = f"  {'/'.join(token_bits)}  {_format_cost(attrs.get('cost_usd'))}"
        if attrs.get("usage_estimated"):
            extra += " (est. tokens)"
    status = "" if span.status == "ok" else f"  ERROR: {span.error or ''}"
    return f"  {label:<44} {span.kind:<9} {duration:>8}{extra}{status}"


def cmd_show(store: Store, args: argparse.Namespace) -> None:
    trace = _resolve_trace(store, args.trace_id)
    spans = store.get_spans(trace.trace_id, hydrate=False)
    stats = store.trace_stats(trace.trace_id)
    print(
        f"trace {trace.trace_id}  '{trace.name}'"
        + (f"  session {trace.session_id}" if trace.session_id else "")
    )
    print(
        f"{stats['span_count']} spans, {stats['llm_count'] or 0} llm calls, "
        f"{stats['error_count'] or 0} errors, "
        f"{_format_duration(_trace_duration(trace))}, "
        f"{_format_cost(stats['cost_usd'])} estimated"
    )
    print()
    children: dict[str | None, list[Span]] = {}
    for span in spans:
        children.setdefault(span.parent_id, []).append(span)
    seen: set[str] = set()

    def render(parent_id: str | None, depth: int) -> None:
        for span in sorted(children.get(parent_id, []), key=lambda s: s.start_ns):
            if span.span_id in seen:
                continue
            seen.add(span.span_id)
            print(_span_line(span, depth))
            render(span.span_id, depth + 1)

    render(None, 0)
    # Orphans: spans whose recorded parent was never written (an unfinished
    # or abandoned run). Render each orphan subtree from its topmost span so
    # the hierarchy underneath survives.
    all_ids = {s.span_id for s in spans}
    orphan_roots = [
        s for s in spans if s.span_id not in seen and s.parent_id not in all_ids
    ]
    if orphan_roots:
        print("\n  (spans with no recorded parent)")
        for span in orphan_roots:
            seen.add(span.span_id)
            print(_span_line(span, 1))
            render(span.span_id, 2)
    leftovers = [s for s in spans if s.span_id not in seen]
    for span in leftovers:
        seen.add(span.span_id)
        print(_span_line(span, 1))


def cmd_waste(store: Store, args: argparse.Namespace) -> None:
    from .analyze import waste

    trace = _resolve_trace(store, args.trace_id)
    result = waste.report(store, trace.trace_id)
    print(f"trace {trace.trace_id[:12]}  '{trace.name}'")
    print(result["headline"])
    if not result["findings"]:
        return
    print()
    for finding in result["findings"]:
        amounts = []
        if finding["avoidable_usd"]:
            amounts.append(f"~{_format_cost(finding['avoidable_usd'])} avoidable")
        if finding["avoidable_tokens"]:
            amounts.append(f"{finding['avoidable_tokens']} tokens")
        if finding["avoidable_seconds"]:
            amounts.append(f"{finding['avoidable_seconds']:.1f}s")
        amount_txt = f"  [{', '.join(amounts)}]" if amounts else ""
        print(
            f"{finding['severity'].upper():<6} {finding['rule_id']:<15} "
            f"{finding['summary']}{amount_txt}"
        )
        print(f"       {finding['explanation']}")
        if finding["span_ids"]:
            shown = ", ".join(s[:10] for s in finding["span_ids"][:6])
            more = len(finding["span_ids"]) - 6
            print(f"       spans: {shown}{f' (+{more} more)' if more > 0 else ''}")
        print(f"       confidence: {finding['confidence']}; figures are estimates")
        print()


def cmd_fix(store: Store, args: argparse.Namespace) -> None:
    from .analyze import waste
    from .analyze.fix import render_fix
    from .schema import Finding

    trace = _resolve_trace(store, args.trace_id)
    result = waste.report(store, trace.trace_id)
    print(f"trace {trace.trace_id[:12]}  '{trace.name}'")
    if not result["findings"]:
        print("no waste found, nothing to fix")
        return

    findings = [Finding.from_dict(f) for f in result["findings"]]
    rendered = [(f, render_fix(f, store)) for f in findings]
    fixable = [(f, patch) for f, patch in rendered if patch]
    unfixable = len(findings) - len(fixable)

    summary = f"{len(fixable)} fix(es) available of {len(findings)} finding(s)"
    if unfixable:
        summary += f"; {unfixable} have no automatic fix (run `traceburn waste` for those)"
    print(summary)

    for finding, patch in fixable:
        print(f"\n--- {finding.rule_id}: {finding.summary} ---")
        print(patch)


def _checked_trace_cost(store: Store, trace: Trace) -> float:
    """Refuse a cost gate when any recorded model call has unknown cost."""
    unknown = []
    for span in store.get_spans(trace.trace_id, hydrate=False):
        if span.kind != "llm":
            continue
        cost = span.attributes.get("cost_usd")
        if (
            not isinstance(cost, (int, float))
            or isinstance(cost, bool)
            or not math.isfinite(cost)
            or cost < 0
        ):
            unknown.append(span.span_id)
    if unknown:
        print("FAIL: insufficient cost data")
        print(
            f"  trace {trace.trace_id[:12]} has {len(unknown)} LLM span(s) with "
            "missing or invalid cost. This can indicate an unknown model price "
            "or missing usage. Record usage and configure model pricing before "
            "rerunning the check."
        )
        print("  spans: " + ", ".join(s[:12] for s in unknown[:5]))
        raise SystemExit(1)
    return store.trace_stats(trace.trace_id)["cost_usd"] or 0.0


def cmd_check(store: Store, args: argparse.Namespace) -> None:
    from .analyze import waste

    if args.trace_id:
        trace = _resolve_trace(store, args.trace_id)
    else:
        traces = store.list_traces(limit=1)
        if not traces:
            raise SystemExit("no traces recorded yet; nothing to check")
        trace = traces[0]

    if args.max_regression_pct is not None and not args.baseline:
        raise SystemExit("--max-regression-pct requires --baseline")

    cost = _checked_trace_cost(store, trace)
    print(f"trace {trace.trace_id[:12]}  '{trace.name}'  {_format_cost(cost)}")

    violations = []

    if args.max_cost is not None and cost > args.max_cost:
        violations.append(
            f"cost {_format_cost(cost)} exceeds --max-cost {_format_cost(args.max_cost)}"
        )

    if args.max_avoidable_pct is not None:
        report = waste.report(store, trace.trace_id)
        pct = (report["avoidable_usd"] / cost * 100) if cost > 0 else 0.0
        print(
            f"avoidable: {pct:.0f}% "
            f"({_format_cost(report['avoidable_usd'])} of {_format_cost(cost)})"
        )
        if pct > args.max_avoidable_pct:
            violations.append(
                f"{pct:.0f}% avoidable exceeds --max-avoidable-pct {args.max_avoidable_pct:.0f}%"
            )

    if args.baseline:
        baseline = _resolve_trace(store, args.baseline)
        baseline_cost = _checked_trace_cost(store, baseline)
        if baseline_cost == 0 and cost > 0:
            violations.append(
                "cannot compute percentage regression from a zero-cost baseline "
                "to a positive current cost; use --max-cost for an absolute budget"
            )
        elif baseline_cost == 0:
            print(f"vs baseline {baseline.trace_id[:12]}: both costs are zero")
        else:
            regression_pct = (cost - baseline_cost) / baseline_cost * 100
            print(
                f"vs baseline {baseline.trace_id[:12]}: {regression_pct:+.0f}% "
                f"({_format_cost(baseline_cost)} -> {_format_cost(cost)})"
            )
            if (
                args.max_regression_pct is not None
                and regression_pct > args.max_regression_pct
            ):
                violations.append(
                    f"{regression_pct:+.0f}% cost regression vs baseline exceeds "
                    f"--max-regression-pct {args.max_regression_pct:.0f}%"
                )

    if violations:
        print("\nFAIL")
        for v in violations:
            print(f"  - {v}")
        raise SystemExit(1)
    print("\nPASS")


def cmd_diff(store: Store, args: argparse.Namespace) -> None:
    from .analyze.diff import diff_traces

    trace_a = _resolve_trace(store, args.trace_a)
    trace_b = _resolve_trace(store, args.trace_b)
    result = diff_traces(store, trace_a.trace_id, trace_id_b=trace_b.trace_id)

    totals_a, totals_b = result["totals_a"], result["totals_b"]
    print(f"a: {trace_a.trace_id[:12]} '{trace_a.name}'   b: {trace_b.trace_id[:12]} '{trace_b.name}'")
    print(
        f"{'':>14}{'spans':>8}{'llm':>6}{'errors':>8}{'time':>10}{'tokens in/out':>16}{'est. cost':>11}"
    )
    for label, totals in (("a", totals_a), ("b", totals_b)):
        print(
            f"{label:>14}{totals['spans']:>8}{totals['llm_calls']:>6}{totals['errors']:>8}"
            f"{totals['duration_ms'] / 1000:>9.1f}s"
            f"{str(totals['input_tokens']) + '/' + str(totals['output_tokens']):>16}"
            f"{_format_cost(totals['cost_usd']):>11}"
        )
    delta_cost = totals_b["cost_usd"] - totals_a["cost_usd"]
    delta_ms = totals_b["duration_ms"] - totals_a["duration_ms"]
    cost_txt = f"{delta_cost:+.4f}$" if abs(delta_cost) >= 5e-5 else "no change"
    print(f"{'delta':>14}{'':>8}{'':>6}{'':>8}{delta_ms / 1000:>+9.1f}s{'':>16}{cost_txt:>11}")
    print()

    def _significant(m):
        deltas = m["deltas"]
        return (
            abs(deltas["duration_ms"]) >= 50
            or abs(deltas["cost_usd"]) >= 5e-5
            or deltas["input_tokens"] != 0
            or deltas["output_tokens"] != 0
            or "request_diff" in m
            or "response_diff" in m
        )

    changed = [m for m in result["matched"] if _significant(m)]
    if changed:
        print("changed steps")
        for m in changed:
            deltas = m["deltas"]
            bits = []
            if deltas["duration_ms"]:
                bits.append(f"{deltas['duration_ms'] / 1000:+.2f}s")
            if deltas["cost_usd"]:
                bits.append(f"{deltas['cost_usd']:+.4f}$")
            if deltas["input_tokens"]:
                bits.append(f"{deltas['input_tokens']:+d} in")
            if deltas["output_tokens"]:
                bits.append(f"{deltas['output_tokens']:+d} out")
            marks = []
            if "request_diff" in m:
                marks.append("prompt changed")
            if "response_diff" in m:
                marks.append("output changed")
            suffix = f"  ({'; '.join(marks)})" if marks else ""
            print(f"  {m['name']:<36} {', '.join(bits) if bits else 'no metric change'}{suffix}")
        if args.verbose:
            for m in changed:
                for key in ("request_diff", "response_diff"):
                    if key in m:
                        print(f"\n--- {m['name']} {key.replace('_', ' ')} ---")
                        for line in m[key][:60]:
                            print(f"  {line}")
    for label, items in (("only in b (added)", result["added"]), ("only in a (removed)", result["removed"])):
        if items:
            print(f"\n{label}")
            for brief in items:
                print(
                    f"  {brief['name']:<36} {brief['kind']:<9} "
                    f"{_format_duration(int(brief['duration_ms'] * 1e6)) if brief['duration_ms'] else '-':>8} "
                    f"{_format_cost(brief['cost_usd']):>10}"
                )


def cmd_ui(store: Store, args: argparse.Namespace) -> None:
    store.close()
    try:
        from .ui.server import serve
    except ImportError:
        raise SystemExit(
            "the web viewer needs the [ui] extra: pip install 'traceburn[ui]'"
        )
    serve(db_path=store.path, port=args.port, open_browser=not args.no_browser)


def cmd_report(store: Store, args: argparse.Namespace) -> None:
    from .report import write_report

    trace = _resolve_trace(store, args.trace_id)
    try:
        path = write_report(store, trace.trace_id, args.output, force=args.force)
    except (OSError, ValueError) as error:
        raise SystemExit(f"cannot write report: {error}")
    print(f"Report: {path}")
    print("Payloads and names are omitted. Review numeric metadata before sharing.")
    if args.open:
        import webbrowser
        webbrowser.open(path.as_uri())


def cmd_demo(args: argparse.Namespace) -> None:
    import tempfile
    import webbrowser

    from .demo import record_demo
    from .report import write_report

    # A separate temporary database keeps synthetic data out of user traces.
    with tempfile.TemporaryDirectory(prefix="traceburn-demo-") as temp:
        store = Store(Path(temp) / "demo.db")
        try:
            trace_id = record_demo(store)
            path = write_report(store, trace_id, args.output, synthetic=True, force=args.force)
        except (OSError, ValueError) as error:
            raise SystemExit(f"cannot write demo: {error}")
        finally:
            store.close()
    print("Synthetic demo: invented usage and costs, no API calls.")
    print(f"Report: {path}")
    print("Inspect the repeated request, expand its caveat, and follow its call links.")
    if not args.no_browser:
        webbrowser.open(path.as_uri())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="traceburn",
        description="Local-first tracer and efficiency profiler for AI agents.",
    )
    parser.add_argument("--db", help="path to the trace database", default=None)
    parser.add_argument("--version", action="version", version=f"traceburn {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo_parser = sub.add_parser("demo", help="open a synthetic standalone demo; no API keys")
    demo_parser.add_argument("-o", "--output", default=".traceburn/demo/report.html")
    demo_parser.add_argument("--no-browser", action="store_true")
    demo_parser.add_argument("--force", action="store_true", help="replace an existing report")

    report_parser = sub.add_parser("report", help="export a standalone HTML report with payloads omitted")
    report_parser.add_argument("trace_id", nargs="?", default="latest", help="trace id, prefix, or latest (default)")
    report_parser.add_argument("-o", "--output", default="traceburn-report.html")
    report_parser.add_argument("--open", action="store_true", help="open the file in your browser")
    report_parser.add_argument("--force", action="store_true", help="replace an existing report")
    report_parser.set_defaults(func=cmd_report)

    ui_parser = sub.add_parser("ui", help="open the local web viewer")
    ui_parser.add_argument("-p", "--port", type=int, default=8765)
    ui_parser.add_argument("--no-browser", action="store_true")
    ui_parser.set_defaults(func=cmd_ui)

    ls_parser = sub.add_parser("ls", help="list recent sessions and traces")
    ls_parser.add_argument("-n", "--limit", type=int, default=20)
    ls_parser.set_defaults(func=cmd_ls)

    show_parser = sub.add_parser("show", help="print one trace as a span tree")
    show_parser.add_argument("trace_id", nargs="?", default="latest", help="trace id, prefix, or latest (default)")
    show_parser.set_defaults(func=cmd_show)

    waste_parser = sub.add_parser("waste", help="efficiency report for one trace")
    waste_parser.add_argument("trace_id", nargs="?", default="latest", help="trace id, prefix, or latest (default)")
    waste_parser.set_defaults(func=cmd_waste)

    fix_parser = sub.add_parser(
        "fix", help="show mechanical patches for fixable waste findings"
    )
    fix_parser.add_argument("trace_id", nargs="?", default="latest", help="trace id, prefix, or latest (default)")
    fix_parser.set_defaults(func=cmd_fix)

    check_parser = sub.add_parser(
        "check", help="CI gate: exit nonzero if cost or waste crosses a threshold"
    )
    check_parser.add_argument(
        "trace_id", nargs="?", default=None,
        help="trace id or unique prefix (default: the most recently recorded trace)",
    )
    check_parser.add_argument(
        "--max-cost", type=float, default=None,
        help="fail if the trace's estimated cost exceeds this many dollars",
    )
    check_parser.add_argument(
        "--max-avoidable-pct", type=float, default=None,
        help="fail if the waste report's avoidable share exceeds this percent",
    )
    check_parser.add_argument(
        "--baseline", default=None, help="compare cost against this trace id or prefix"
    )
    check_parser.add_argument(
        "--max-regression-pct", type=float, default=None,
        help="with --baseline, fail if cost rose by more than this percent",
    )
    check_parser.set_defaults(func=cmd_check)

    diff_parser = sub.add_parser("diff", help="compare two traces")
    diff_parser.add_argument("trace_a", help="baseline trace id or prefix")
    diff_parser.add_argument("trace_b", help="comparison trace id or prefix")
    diff_parser.add_argument(
        "-v", "--verbose", action="store_true", help="include prompt and response text diffs"
    )
    diff_parser.set_defaults(func=cmd_diff)

    args = parser.parse_args(argv)
    if args.command == "demo":
        cmd_demo(args)
        return 0
    db_path = args.db or default_db_path()
    if not Path(db_path).exists():
        raise SystemExit(
            f"no trace database at {db_path} (record a trace first, "
            "or point --db or TRACEBURN_DB at one)"
        )
    store = Store(db_path)
    try:
        args.func(store, args)
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
