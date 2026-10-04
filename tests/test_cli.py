import pytest

from traceburn.cli import main
from traceburn.recorder import Recorder
from traceburn.store import Store


@pytest.fixture
def populated_db(tmp_path):
    db = str(tmp_path / "traces.db")
    recorder = Recorder(store=Store(db))
    with recorder.session("nightly"):
        with recorder.span("research-agent", kind="agent"):
            with recorder.span(
                "chat gpt-test",
                kind="llm",
                attributes={
                    "gen_ai.system": "openai",
                    "gen_ai.request.model": "gpt-test",
                    "gen_ai.usage.input_tokens": 1000,
                    "gen_ai.usage.output_tokens": 200,
                    "cost_usd": 0.0123,
                },
            ):
                pass
            with recorder.span("search", kind="tool"):
                pass
    recorder.store.close()
    return db


def test_ls(populated_db, capsys):
    assert main(["--db", populated_db, "ls"]) == 0
    out = capsys.readouterr().out
    assert "nightly" in out
    assert "research-agent" in out
    assert "$0.0123" in out


def test_show_with_prefix(populated_db, capsys):
    store = Store(populated_db)
    trace_id = store.list_traces()[0].trace_id
    store.close()

    assert main(["--db", populated_db, "show", trace_id[:8]]) == 0
    out = capsys.readouterr().out
    assert "research-agent" in out
    assert "chat gpt-test" in out
    assert "1000in" in out
    assert "200out" in out
    assert "search" in out


def test_show_unknown_prefix_exits(populated_db):
    with pytest.raises(SystemExit, match="no trace found"):
        main(["--db", populated_db, "show", "zzzzzz"])


def test_ls_empty_db(tmp_path, capsys):
    db = str(tmp_path / "empty.db")
    Store(db).close()
    assert main(["--db", db, "ls"]) == 0
    assert "no traces recorded yet" in capsys.readouterr().out


def test_missing_db_exits_cleanly(tmp_path):
    with pytest.raises(SystemExit, match="no trace database"):
        main(["--db", str(tmp_path / "absent.db"), "ls"])


def test_duration_formatting_carries_units():
    from traceburn.cli import _format_duration

    assert _format_duration(119_600_000_000) == "2m0s"
    assert _format_duration(999_600_000) == "1.0s"
    assert _format_duration(59_960_000_000) == "1m0s"
    assert _format_duration(5_000_000) == "5.0ms"
    assert _format_duration(None) == "-"


@pytest.fixture
def db_with_fixable_finding(tmp_path):
    """A trace with a model_overkill finding, which carries a renderable fix."""
    db = str(tmp_path / "traces.db")
    recorder = Recorder(store=Store(db))
    with recorder.session("s"):
        with recorder.span("agent", kind="agent"):
            for i in range(3):
                with recorder.span(
                    "chat gpt-5",
                    kind="llm",
                    attributes={
                        "gen_ai.system": "openai",
                        "gen_ai.request.model": "gpt-5",
                        "gen_ai.usage.input_tokens": 100,
                        "gen_ai.usage.output_tokens": 10,
                        "cost_usd": (100 * 1.25 + 10 * 10.0) / 1e6,
                        "request_hash": f"h{i}",
                        "response": {"text": "ok", "tool_calls": []},
                    },
                ):
                    pass
    recorder.store.close()
    return db


@pytest.fixture
def db_with_no_waste(tmp_path):
    db = str(tmp_path / "traces.db")
    recorder = Recorder(store=Store(db))
    with recorder.session("s"):
        with recorder.span("agent", kind="agent"):
            with recorder.span(
                "chat gpt-test", kind="llm",
                attributes={
                    "gen_ai.system": "openai", "gen_ai.request.model": "gpt-test",
                    "gen_ai.usage.input_tokens": 50, "gen_ai.usage.output_tokens": 10,
                    "cost_usd": 0.001,
                },
            ):
                pass
    recorder.store.close()
    return db


def test_fix_shows_model_swap_patch(db_with_fixable_finding, capsys):
    store = Store(db_with_fixable_finding)
    trace_id = store.list_traces()[0].trace_id
    store.close()

    assert main(["--db", db_with_fixable_finding, "fix", trace_id[:8]]) == 0
    out = capsys.readouterr().out
    assert "fix(es) available" in out
    assert "model_overkill" in out
    assert "gpt-5" in out


def test_fix_reports_nothing_to_fix_when_clean(db_with_no_waste, capsys):
    store = Store(db_with_no_waste)
    trace_id = store.list_traces()[0].trace_id
    store.close()

    assert main(["--db", db_with_no_waste, "fix", trace_id[:8]]) == 0
    assert "nothing to fix" in capsys.readouterr().out


def test_check_passes_under_threshold(populated_db, capsys):
    assert main(["--db", populated_db, "check", "--max-cost", "1.0"]) == 0
    assert "PASS" in capsys.readouterr().out


def test_check_fails_over_cost_threshold(populated_db, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--db", populated_db, "check", "--max-cost", "0.001"])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "exceeds --max-cost" in out


def test_check_defaults_to_latest_trace(populated_db, capsys):
    assert main(["--db", populated_db, "check", "--max-cost", "1.0"]) == 0
    out = capsys.readouterr().out
    assert "research-agent" in out


def test_check_no_traces_exits_cleanly(tmp_path):
    db = str(tmp_path / "empty.db")
    Store(db).close()
    with pytest.raises(SystemExit, match="no traces recorded"):
        main(["--db", db, "check"])


def test_check_baseline_regression(db_with_fixable_finding, capsys):
    store = Store(db_with_fixable_finding)
    recorder = Recorder(store=store)
    with recorder.session("s2"):
        with recorder.span("agent-expensive", kind="agent"):
            with recorder.span(
                "chat gpt-5", kind="llm",
                attributes={
                    "gen_ai.system": "openai", "gen_ai.request.model": "gpt-5",
                    "gen_ai.usage.input_tokens": 100_000, "gen_ai.usage.output_tokens": 10_000,
                    "cost_usd": 0.5,
                },
            ):
                pass
    traces = store.list_traces()
    baseline_id = traces[-1].trace_id  # the small, first-recorded trace
    expensive_id = traces[0].trace_id  # the newest, most expensive trace
    store.close()

    with pytest.raises(SystemExit) as exc:
        main([
            "--db", db_with_fixable_finding, "check", expensive_id[:8],
            "--baseline", baseline_id[:8], "--max-regression-pct", "10",
        ])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "regression" in out


def record_check_trace(db, model="unpriced-model", cost=None, usage=True):
    store = Store(db)
    recorder = Recorder(store=store)
    attrs = {"gen_ai.system": "openai", "gen_ai.request.model": model}
    if usage:
        attrs["gen_ai.usage.input_tokens"] = 10
    if cost is not None:
        attrs["cost_usd"] = cost
    with recorder.span("agent", kind="agent") as root:
        with recorder.span("tool", kind="tool"):
            pass
        with recorder.span("model", kind="llm", attributes=attrs):
            pass
    store.close()
    return root.span.trace_id


@pytest.mark.parametrize("flags", [
    ["--max-cost", "1"], ["--max-avoidable-pct", "25"], [],
])
def test_check_fails_when_a_model_has_unknown_price(tmp_path, capsys, flags):
    db = str(tmp_path / "check.db")
    trace_id = record_check_trace(db)
    with pytest.raises(SystemExit) as exc:
        main(["--db", db, "check", trace_id] + flags)
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "insufficient cost data" in out
    assert "unknown model price" in out
    assert "PASS" not in out


def test_check_rejects_known_model_with_missing_usage(tmp_path, capsys):
    db = str(tmp_path / "check.db")
    trace_id = record_check_trace(db, model="gpt-4o", usage=False)
    with pytest.raises(SystemExit) as exc:
        main(["--db", db, "check", trace_id, "--max-cost", "1"])
    assert exc.value.code == 1
    assert "missing usage" in capsys.readouterr().out


def test_check_rejects_unknown_baseline_cost(tmp_path, capsys):
    db = str(tmp_path / "check.db")
    baseline = record_check_trace(db)
    current = record_check_trace(db, cost=0.01)
    with pytest.raises(SystemExit) as exc:
        main(["--db", db, "check", current, "--baseline", baseline,
              "--max-regression-pct", "10"])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert baseline[:12] in out
    assert "insufficient cost data" in out


def test_check_fails_positive_cost_against_zero_baseline(tmp_path, capsys):
    db = str(tmp_path / "check.db")
    baseline = record_check_trace(db, cost=0)
    current = record_check_trace(db, cost=0.01)
    with pytest.raises(SystemExit) as exc:
        main(["--db", db, "check", current, "--baseline", baseline,
              "--max-regression-pct", "10"])
    assert exc.value.code == 1
    assert "zero-cost baseline" in capsys.readouterr().out


def test_check_accepts_explicit_zero_cost_and_unpriced_tools(tmp_path, capsys):
    db = str(tmp_path / "check.db")
    baseline = record_check_trace(db, cost=0)
    current = record_check_trace(db, cost=0)
    assert main(["--db", db, "check", current, "--baseline", baseline,
                 "--max-regression-pct", "10", "--max-cost", "0"]) == 0
    assert "PASS" in capsys.readouterr().out


def test_check_requires_baseline_for_relative_threshold(populated_db):
    with pytest.raises(SystemExit, match="requires --baseline"):
        main(["--db", populated_db, "check", "--max-regression-pct", "10"])


@pytest.mark.parametrize("invalid_cost", [float("nan"), float("inf"), -0.1, "unknown"])
def test_check_rejects_invalid_cost_values(tmp_path, capsys, invalid_cost):
    db = str(tmp_path / "check.db")
    trace_id = record_check_trace(db, cost=invalid_cost)
    with pytest.raises(SystemExit) as exc:
        main(["--db", db, "check", trace_id, "--max-cost", "1"])
    assert exc.value.code == 1
    assert "insufficient cost data" in capsys.readouterr().out
