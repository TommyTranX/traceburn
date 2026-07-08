"""Read-only JSON API over the trace store, plus the static viewer.

Binds to 127.0.0.1 only; there is no auth because there is no remote
access. Every endpoint is a read; the viewer cannot modify a trace. The
JSON shapes served here are the same public interfaces the analyzers
return (span dicts, flamegraph folds, waste reports, diffs).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .. import __version__
from ..analyze import waste
from ..analyze.diff import diff_traces
from ..analyze.flamegraph import fold, waterfall
from ..store import Store

STATIC_DIR = Path(__file__).parent / "static"

# The standard DNS-rebinding defense for a localhost tool: a page on an
# attacker's domain that rebinds to 127.0.0.1 still sends its own Host.
ALLOWED_HOSTS = ["127.0.0.1", "localhost", "testserver"]


def _limit(request: Request, default: int, ceiling: int = 1000) -> int:
    try:
        value = int(request.query_params.get("limit", default))
    except (TypeError, ValueError):
        return default
    return max(1, min(value, ceiling))


def create_app(db_path: str | None = None, on_startup: Callable[[], None] | None = None) -> Starlette:
    """Build the Starlette app.

    ``on_startup``, when given, runs once the ASGI server has actually
    started (Starlette's only supported hook for this is the ``lifespan``
    context manager; the old ``add_event_handler``/``on_event`` API this
    version once used was removed). ``serve()`` uses it to open a browser
    only after a successful bind, never on a port that failed to open.
    """
    store = Store(db_path)

    def meta(request: Request) -> JSONResponse:
        return JSONResponse({"version": __version__, "db": store.path})

    def sessions(request: Request) -> JSONResponse:
        items = []
        for sess in store.list_sessions(limit=_limit(request, 50)):
            row = sess.to_dict()
            row["trace_count"] = store.count_traces(sess.session_id)
            items.append(row)
        return JSONResponse({"sessions": items})

    def traces(request: Request) -> JSONResponse:
        limit = _limit(request, 100)
        session_id = request.query_params.get("session_id")
        items = []
        for trace in store.list_traces(limit=limit, session_id=session_id):
            row = trace.to_dict()
            row["stats"] = store.trace_stats(trace.trace_id)
            items.append(row)
        return JSONResponse({"traces": items})

    def _resolve(trace_id: str):
        trace = store.get_trace(trace_id)
        if trace is None:
            matches = store.find_traces(trace_id)
            trace = matches[0] if len(matches) == 1 else None
        return trace

    def trace_detail(request: Request) -> JSONResponse:
        trace = _resolve(request.path_params["trace_id"])
        if trace is None:
            return JSONResponse({"error": "trace not found"}, status_code=404)
        row = trace.to_dict()
        row["stats"] = store.trace_stats(trace.trace_id)
        return JSONResponse(row)

    def spans(request: Request) -> JSONResponse:
        trace = _resolve(request.path_params["trace_id"])
        if trace is None:
            return JSONResponse({"error": "trace not found"}, status_code=404)
        hydrated = store.get_spans(
            trace.trace_id, hydrate=True, hydrate_keys=("request", "response")
        )
        return JSONResponse({"spans": [s.to_dict() for s in hydrated]})

    def flamegraph_view(request: Request) -> JSONResponse:
        trace = _resolve(request.path_params["trace_id"])
        if trace is None:
            return JSONResponse({"error": "trace not found"}, status_code=404)
        weight = request.query_params.get("weight", "latency")
        if weight not in ("latency", "cost"):
            return JSONResponse({"error": "weight must be latency or cost"}, status_code=400)
        tree = fold(store.get_spans(trace.trace_id, hydrate=False), weight=weight)
        return JSONResponse(tree)

    def waterfall_view(request: Request) -> JSONResponse:
        trace = _resolve(request.path_params["trace_id"])
        if trace is None:
            return JSONResponse({"error": "trace not found"}, status_code=404)
        rows = waterfall(store.get_spans(trace.trace_id, hydrate=False))
        return JSONResponse({"rows": rows})

    def waste_view(request: Request) -> JSONResponse:
        trace = _resolve(request.path_params["trace_id"])
        if trace is None:
            return JSONResponse({"error": "trace not found"}, status_code=404)
        return JSONResponse(waste.report(store, trace.trace_id))

    def diff_view(request: Request) -> JSONResponse:
        a, b = request.query_params.get("a"), request.query_params.get("b")
        if not a or not b:
            return JSONResponse({"error": "pass a and b trace ids"}, status_code=400)
        trace_a, trace_b = _resolve(a), _resolve(b)
        if trace_a is None or trace_b is None:
            return JSONResponse({"error": "trace not found"}, status_code=404)
        return JSONResponse(diff_traces(store, trace_a.trace_id, trace_id_b=trace_b.trace_id))

    def index(request: Request) -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @asynccontextmanager
    async def lifespan(app: Starlette):
        if on_startup is not None:
            on_startup()
        yield

    return Starlette(
        routes=[
            Route("/", index),
            Route("/api/meta", meta),
            Route("/api/sessions", sessions),
            Route("/api/traces", traces),
            Route("/api/traces/{trace_id}", trace_detail),
            Route("/api/traces/{trace_id}/spans", spans),
            Route("/api/traces/{trace_id}/flamegraph", flamegraph_view),
            Route("/api/traces/{trace_id}/waterfall", waterfall_view),
            Route("/api/traces/{trace_id}/waste", waste_view),
            Route("/api/diff", diff_view),
            Mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static"),
        ],
        middleware=[Middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)],
        lifespan=lifespan,
    )


def serve(db_path: str | None = None, port: int = 8765, open_browser: bool = True) -> None:
    """Run the viewer on 127.0.0.1. Blocks until interrupted."""
    import webbrowser

    import uvicorn

    url = f"http://127.0.0.1:{port}"
    # Fires only once uvicorn has actually bound and started; a failed
    # bind must not open a browser at someone else's server.
    on_startup = (lambda: webbrowser.open(url)) if open_browser else None
    app = create_app(db_path, on_startup=on_startup)
    print(f"traceburn viewer at {url} (ctrl-c to stop)")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
