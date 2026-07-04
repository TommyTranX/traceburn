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
