"""Terminal viewer: the fast path into a trace database.

    traceburn ls                   recent sessions and traces
    traceburn show <trace_id>      span tree with timing, tokens, and cost

Trace ids can be abbreviated to any unique prefix. The database path comes
from --db, the TRACEBURN_DB environment variable, or ./.traceburn/traces.db,
in that order. Stdlib only; no server, no network.
"""

from __future__ import annotations

import argparse
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="traceburn",
        description="Local-first tracer and efficiency profiler for AI agents.",
    )
    parser.add_argument("--db", help="path to the trace database", default=None)
    parser.add_argument("--version", action="version", version=f"traceburn {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    ls_parser = sub.add_parser("ls", help="list recent sessions and traces")
    ls_parser.add_argument("-n", "--limit", type=int, default=20)
    ls_parser.set_defaults(func=cmd_ls)

    show_parser = sub.add_parser("show", help="print one trace as a span tree")
    show_parser.add_argument("trace_id", help="trace id or unique prefix")
    show_parser.set_defaults(func=cmd_show)

    args = parser.parse_args(argv)
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
