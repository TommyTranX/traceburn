"""The no-telemetry guarantee, as a test.

The README states traceburn makes no network calls of its own. This test
blocks socket access at the interpreter level and exercises the full local
surface: recording, storage, the pricing table, every analyzer, and the
terminal viewer. Any connection attempt fails the test.
"""

import socket

import pytest

from traceburn.analyze import waste
from traceburn.analyze.diff import diff_traces
from traceburn.analyze.flamegraph import fold, waterfall
from traceburn.cli import main
from traceburn.pricing import PricingTable
from traceburn.recorder import Recorder
from traceburn.store import Store


@pytest.fixture
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("traceburn attempted a network call")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)


def test_entire_local_surface_makes_no_network_calls(no_network, tmp_path, capsys):
    db = str(tmp_path / "traces.db")
    recorder = Recorder(store=Store(db), pricing=PricingTable.load())

    trace_ids = []
    for run in range(2):
        with recorder.session(f"run-{run}"):
            with recorder.span("agent", kind="agent") as root:
                with recorder.span(
                    "chat gpt-4o",
                    kind="llm",
                    attributes={
                        "gen_ai.system": "openai",
                        "gen_ai.request.model": "gpt-4o",
                        "gen_ai.usage.input_tokens": 1200,
                        "gen_ai.usage.output_tokens": 80,
                        "request": {"messages": [{"role": "user", "content": "hi"}]},
                        "response": {"text": "hello", "tool_calls": []},
                        "request_hash": "same-hash",
                    },
                ):
                    pass
                with recorder.span("lookup", kind="tool"):
                    pass
            trace_ids.append(root.span.trace_id)

    store = recorder.store
    spans = store.get_spans(trace_ids[0])
    assert fold(spans, weight="cost")["children"]
    assert waterfall(spans)
    report = waste.report(store, trace_ids[0])
    assert "headline" in report
    assert diff_traces(store, trace_ids[0], trace_id_b=trace_ids[1])["matched"]
    store.close()

    for argv in (["ls"], ["show", trace_ids[0][:8]], ["waste", trace_ids[0][:8]],
                 ["diff", trace_ids[0][:10], trace_ids[1][:10]]):
        assert main(["--db", db] + argv) == 0
    capsys.readouterr()
