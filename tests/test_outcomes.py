import csv
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from traceburn.cli import main
from traceburn.outcomes import attempt_from_trace, load_attempts, render_summary, summarize
from traceburn.schema import Span, Trace

ROOT = Path(__file__).resolve().parents[1]


def manifest(tasks=("a",), **kwargs):
    return dict(schema_version=1, conditions=["baseline"], task_ids=list(tasks),
                quality_check="independent exact match", pricing_ref="test rates",
                cost_scope="model_api", evidence="synthetic", cost_basis="estimated", **kwargs)


def row(task="a", attempt=1, outcome="pass", cost=1.0, complete=True, **kwargs):
    result = dict(condition="baseline", task_id=task, attempt=attempt, outcome=outcome,
                  cost_usd=cost, billing_complete=complete, model_calls=1,
                  retry_calls=0, latency_s=2.0)
    result.update(kwargs)
    return result


def test_failed_attempts_and_restarts_stay_in_cost():
    rows = [row(outcome="fail", cost=2), row(attempt=2, cost=3), row(task="b", outcome="fail", cost=4)]
    result = summarize(rows, manifest(("a", "b")))["conditions"][0]
    assert result["cost_per_success_usd"] == 9
    assert result["success_rate"] == .5
    assert result["failed_attempt_cost_usd"] == 6
    assert result["restarted_attempts"] == 1
    assert result["attempts"] == 3


@pytest.mark.parametrize("rows,tasks,reason", [
    ([row(cost=None, complete=False)], ("a",), "incomplete_billing"),
    ([row(cost=2, complete=False)], ("a",), "incomplete_billing"),
    ([row(outcome="fail")], ("a",), "zero_successes"),
    ([row(outcome="ungraded")], ("a",), "ungraded_outcomes"),
    ([row()], ("a", "b"), "missing_tasks"),
    ([], ("a",), "missing_tasks"),
])
def test_incomplete_inputs_cannot_produce_cost_per_success(rows, tasks, reason):
    result = summarize(rows, manifest(tasks))["conditions"][0]
    assert result["cost_per_success_usd"] is None
    assert reason in result["undefined_reasons"]


def test_explicit_zero_is_known():
    result = summarize([row(cost=0)], manifest())["conditions"][0]
    assert result["total_cost_usd"] == result["cost_per_success_usd"] == 0
    assert not result["undefined_reasons"]


def test_ungraded_first_attempt_is_not_silently_overwritten_by_success():
    result = summarize([row(outcome="ungraded"), row(attempt=2)], manifest())["conditions"][0]
    assert result["ungraded_tasks"] == 1
    assert result["success_rate"] is None


@pytest.mark.parametrize("rows", [
    [row(), row()],
    [row(), row(attempt=2)],
    [row(attempt=2)],
    [row(outcome="fail"), row(attempt=3)],
    [row(task="undeclared")],
    [row(attempt=True)],
    [row(cost=-1)],
    [row(cost=True)],
    [row(cost=float("nan"))],
    [row(cost=float("inf"))],
    [row(cost=None)],
    [row(complete="true")],
    [row(model_calls=1, retry_calls=2)],
    [row(latency_s=-1)],
    [row(outcome="ok")],
])
def test_reject_ambiguous_or_invalid_accounting(rows):
    with pytest.raises(ValueError):
        summarize(rows, manifest())


@pytest.mark.parametrize("value", [[], None, {}, {**manifest(), "schema_version": True},
                                         {**manifest(), "conditions": ["=evil"]}])
def test_invalid_manifest(value):
    with pytest.raises(ValueError):
        summarize([row()], value)


def test_csv_and_jsonl_agree_and_unknowns_remain_unknown(tmp_path):
    rows = [row(outcome="fail", cost=None, complete=False, model_calls=None,
                retry_calls=None, latency_s=None), row(attempt=2)]
    jsonl = tmp_path / "attempts.jsonl"
    jsonl.write_text("".join(json.dumps(r) + "\n" for r in rows))
    csv_path = tmp_path / "attempts.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert load_attempts(csv_path) == load_attempts(jsonl) == rows


def test_csv_bad_row_is_a_clear_error(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("condition,attempt\nbaseline,1\n")
    with pytest.raises(ValueError, match="columns"):
        load_attempts(path)


def test_serializations_preserve_undefined(tmp_path):
    report = summarize([row(cost=None, complete=False)], manifest())
    assert json.loads(render_summary(report, "json"))["conditions"][0]["cost_per_success_usd"] is None
    csv_row = next(csv.DictReader(io.StringIO(render_summary(report, "csv"))))
    assert csv_row["cost_per_success_usd"] == ""
    assert csv_row["undefined_reasons"] == "incomplete_billing"
    assert "undefined" in render_summary(report)


def test_csv_metadata_cannot_become_a_formula():
    report = summarize([row()], {**manifest(), "quality_check": "=1+1"})
    result = next(csv.DictReader(io.StringIO(render_summary(report, "csv"))))
    assert result["quality_check"] == "'=1+1"


def test_trace_join_excludes_parent_cost_and_requires_capture_assertion(store):
    store.insert_trace(Trace(trace_id="t", name="task", start_ns=0, end_ns=10_000_000_000))
    store.insert_span(Span(span_id="parent", trace_id="t", name="task", kind="agent", start_ns=0,
                           attributes={"cost_usd": 100}))
    store.insert_span(Span(span_id="known", trace_id="t", name="call", kind="llm", start_ns=0,
                           attributes={"cost_usd": 2}))
    args = dict(condition="baseline", task_id="a", attempt=1, outcome="fail", capture_complete=False)
    record = attempt_from_trace(store, "t", **args)
    assert record["cost_usd"] == 2
    assert record["model_calls"] == 1
    assert not record["billing_complete"]
    assert record["outcome"] == "fail"
    assert record["latency_s"] == 10
    args["capture_complete"] = True
    assert attempt_from_trace(store, "t", **args)["billing_complete"]
    store.insert_span(Span(span_id="unknown", trace_id="t", name="error", kind="llm", start_ns=0))
    record = attempt_from_trace(store, "t", **args)
    assert not record["billing_complete"]
    assert record["cost_usd"] == 2


def test_cli_needs_no_trace_database_and_preserves_formats(capsys, tmp_path):
    assert main(["--db", str(tmp_path / "absent.db"), "outcomes",
                 str(ROOT / "examples/task_costs/retries.csv"), "--manifest",
                 str(ROOT / "examples/task_costs/retries.manifest.json"), "--format", "json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["conditions"][0]["cost_per_success_usd"] == pytest.approx(.06)
    assert report["conditions"][1]["cost_per_success_usd"] == pytest.approx(.09)
    assert not (tmp_path / "absent.db").exists()


def test_cli_invalid_input_has_no_traceback(capsys, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("[]")
    with pytest.raises(SystemExit) as error:
        main(["outcomes", str(ROOT / "examples/task_costs/retries.csv"), "--manifest", str(path)])
    assert error.value.code == 2
    assert "Traceback" not in capsys.readouterr().err


def test_frozen_reanalysis_is_reproducible_without_overclaiming():
    spec = importlib.util.spec_from_file_location("reproduce", ROOT / "examples/task_costs/reproduce.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows, report = module.reproduce()
    assert len(rows) == 288
    baseline, truncated = report["conditions"]
    assert (baseline["successful_tasks"], truncated["successful_tasks"]) == (142, 68)
    assert baseline["known_cost_usd"] == pytest.approx(1.708166)
    assert truncated["known_cost_usd"] == pytest.approx(3.627289)
    assert all(r["cost_per_success_usd"] is None for r in report["conditions"])
    subprocess.run([sys.executable, "-B", str(ROOT / "examples/task_costs/reproduce.py")], check=True, capture_output=True)
