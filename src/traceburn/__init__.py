"""traceburn: a local-first tracer and efficiency profiler for AI agents.

Record every LLM and tool call an agent makes into one local SQLite file,
then see where the time and money went. No account, no server, no telemetry.

Quick use:

    import traceburn
    from traceburn import trace, span, session

    @trace("plan", kind="agent")
    def plan(state): ...

    with session("nightly-run"):
        with span("retrieve", kind="retrieval"):
            docs = retriever(query)
        answer = plan(state)

Or zero-config: ``traceburn.install()`` patches whichever of the openai and
anthropic clients are installed, and every call they make is recorded.
"""

from .instrument import install, uninstall
from .pricing import ModelPrice, PricingTable
from .recorder import (
    Recorder,
    SpanHandle,
    configure,
    current_session,
    current_span,
    get_recorder,
    session,
    span,
    trace,
)
from .schema import SPAN_KINDS, Finding, Session, Span, Trace
from .store import Store

__version__ = "0.1.0.dev0"

__all__ = [
    "SPAN_KINDS",
    "Finding",
    "ModelPrice",
    "PricingTable",
    "Recorder",
    "Session",
    "Span",
    "SpanHandle",
    "Store",
    "Trace",
    "__version__",
    "configure",
    "current_session",
    "current_span",
    "get_recorder",
    "install",
    "session",
    "span",
    "trace",
    "uninstall",
]
