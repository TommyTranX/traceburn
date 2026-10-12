"""Outcome-aware model/API cost accounting, independent of a tracing backend.

The manifest declares the cohort before aggregation. Each input row is one
attempt at a task, including failed attempts. A retry within an attempt stays
in that attempt's cost; a restarted attempt gets the next attempt number.
"""
from __future__ import annotations

import csv
import html
import io
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from .costs import model_costs, valid_cost

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}\Z")
FIELDS = ("condition", "task_id", "attempt", "outcome", "cost_usd",
          "billing_complete", "model_calls", "retry_calls", "latency_s")


def _identifier(value):
    return isinstance(value, str) and bool(_ID.fullmatch(value))


def _integer(value, minimum=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def load_attempts(path: str | Path) -> list[dict]:
    """Read JSONL or CSV. Blank optional CSV cells mean unknown, never zero."""
    path = Path(path)
    if path.suffix.lower() != ".csv":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if any(row.get(f) is None for f in FIELDS) or None in row:
            raise ValueError("CSV needs all attempt columns and the same number of cells per row")
        for field in ("attempt", "model_calls", "retry_calls"):
            raw = row.get(field, "")
            row[field] = int(raw) if raw != "" else None
        for field in ("cost_usd", "latency_s"):
            raw = row.get(field, "")
            row[field] = float(raw) if raw != "" else None
        raw = row.get("billing_complete", "").lower()
        if raw not in ("true", "false"):
            raise ValueError("CSV billing_complete must be true or false")
        row["billing_complete"] = raw == "true"
    return rows


def summarize(attempts: list[dict], manifest: dict) -> dict:
    """Return descriptive cohort statistics, refusing ambiguous accounting."""
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("manifest schema_version must be 1")
    for field in ("conditions", "task_ids"):
        values = manifest.get(field)
        if (not isinstance(values, list) or not values
                or not all(_identifier(v) for v in values) or len(set(values)) != len(values)):
            raise ValueError(f"manifest {field} must be a nonempty list of unique identifiers")
    for field in ("quality_check", "pricing_ref"):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise ValueError(f"manifest requires {field}")
    if manifest.get("cost_scope") != "model_api":
        raise ValueError("schema v1 supports cost_scope=model_api only; not total business cost")
    if manifest.get("evidence") not in ("measured", "synthetic", "replayed"):
        raise ValueError("declare evidence as measured, synthetic, or replayed")
    if manifest.get("cost_basis") not in ("estimated", "invoiced"):
        raise ValueError("declare cost_basis as estimated or invoiced")

    if not isinstance(attempts, list):
        raise ValueError("attempts must be a list of objects")
    groups = defaultdict(list)
    seen = set()
    for row in attempts:
        if not isinstance(row, dict) or any(f not in row for f in FIELDS):
            raise ValueError(f"every attempt needs fields: {', '.join(FIELDS)}")
        if row["condition"] not in manifest["conditions"] or row["task_id"] not in manifest["task_ids"]:
            raise ValueError("attempt outside the declared cohort or conditions")
        if not _integer(row["attempt"], 1):
            raise ValueError("attempt numbers must be positive integers")
        key = (row["condition"], row["task_id"], row["attempt"])
        if key in seen:
            raise ValueError(f"duplicate attempt: {key}")
        seen.add(key)
        if row["outcome"] not in ("pass", "fail", "ungraded"):
            raise ValueError("outcome must be pass, fail, or ungraded")
        if type(row["billing_complete"]) is not bool:
            raise ValueError("billing_complete must be a boolean")
        if row["cost_usd"] is not None and not valid_cost(row["cost_usd"]):
            raise ValueError("cost_usd must be finite, nonnegative, or null")
        if row["billing_complete"] and row["cost_usd"] is None:
            raise ValueError("complete billing requires a cost, including explicit zero")
        for field in ("model_calls", "retry_calls"):
            if row[field] is not None and not _integer(row[field]):
                raise ValueError(f"{field} must be a nonnegative integer or null")
        if (row["retry_calls"] is not None and row["model_calls"] is not None
                and row["retry_calls"] > row["model_calls"]):
            raise ValueError("retry_calls cannot exceed model_calls")
        if row["latency_s"] is not None and not valid_cost(row["latency_s"]):
            raise ValueError("latency_s must be finite, nonnegative, or null")
        groups[(row["condition"], row["task_id"])].append(row)

    summaries = []
    for condition in manifest["conditions"]:
        rows = [r for r in attempts if r["condition"] == condition]
        passed = failed = ungraded = missing = 0
        for task in manifest["task_ids"]:
            history = sorted(groups[(condition, task)], key=lambda r: r["attempt"])
            if not history:
                missing += 1
                continue
            if [r["attempt"] for r in history] != list(range(1, len(history) + 1)):
                raise ValueError("attempt numbers must be contiguous from 1")
            if any(r["outcome"] == "pass" for r in history[:-1]):
                raise ValueError("attempt after task already passed; use a new task_id for a new trial")
            if any(r["outcome"] == "ungraded" for r in history):
                ungraded += 1
            elif history[-1]["outcome"] == "pass":
                passed += 1
            else:
                failed += 1
        known = math.fsum(r["cost_usd"] or 0 for r in rows)
        billing_complete = bool(rows) and not missing and all(r["billing_complete"] for r in rows)
        reasons = []
        if missing:
            reasons.append("missing_tasks")
        if ungraded:
            reasons.append("ungraded_outcomes")
        if not billing_complete:
            reasons.append("incomplete_billing")
        if not passed:
            reasons.append("zero_successes")
        total = known if billing_complete and not missing else None
        failed_rows = [r for r in rows if r["outcome"] == "fail"]
        failed_cost = math.fsum(r["cost_usd"] or 0 for r in failed_rows)
        def complete_sum(field):
            return (sum(r[field] for r in rows)
                    if rows and not missing and all(r[field] is not None for r in rows) else None)
        summaries.append({
            "condition": condition, "expected_tasks": len(manifest["task_ids"]),
            "observed_tasks": len(manifest["task_ids"]) - missing,
            "successful_tasks": passed, "failed_tasks": failed,
            "ungraded_tasks": ungraded, "missing_tasks": missing,
            "attempts": len(rows), "restarted_attempts": len(rows) - (passed + failed + ungraded),
            "success_rate": passed / len(manifest["task_ids"]) if not missing and not ungraded else None,
            "known_cost_usd": known, "total_cost_usd": total,
            "billing_complete": billing_complete,
            "failed_attempt_known_cost_usd": failed_cost,
            "failed_attempt_cost_usd": failed_cost if not missing and all(r["billing_complete"] for r in failed_rows) else None,
            "model_calls": complete_sum("model_calls"), "retry_calls": complete_sum("retry_calls"),
            "summed_attempt_latency_s": complete_sum("latency_s"),
            "cost_per_success_usd": known / passed if not reasons else None,
            "undefined_reasons": reasons,
        })
    return {"schema_version": 1, "evidence": manifest["evidence"],
            "cost_scope": manifest["cost_scope"], "cost_basis": manifest["cost_basis"],
            "quality_check": manifest["quality_check"], "pricing_ref": manifest["pricing_ref"],
            "conditions": summaries}


def attempt_from_trace(store, trace_id: str, *, condition: str, task_id: str,
                       attempt: int, outcome: str, capture_complete: bool,
                       retry_calls: int | None = None) -> dict:
    """Join an independently assessed outcome to recorded model costs.

    capture_complete is the caller's assertion that all billable calls are
    represented. A trace cannot establish this or infer business success.
    """
    trace = store.get_trace(trace_id)
    if trace is None:
        raise ValueError("trace not found")
    if type(capture_complete) is not bool:
        raise ValueError("capture_complete must be a boolean")
    costs = model_costs(store.get_spans(trace_id, hydrate=False))
    complete = capture_complete and trace.end_ns is not None and costs["unknown_cost_calls"] == 0
    return {"condition": condition, "task_id": task_id, "attempt": attempt,
            "outcome": outcome, "cost_usd": costs["known_cost_usd"],
            "billing_complete": complete,
            "model_calls": costs["llm_calls"], "retry_calls": retry_calls,
            "latency_s": (trace.end_ns - trace.start_ns) / 1e9 if trace.end_ns is not None else None}


def render_summary(report: dict, output_format: str = "markdown") -> str:
    if output_format == "json":
        return json.dumps(report, indent=2, allow_nan=False) + "\n"
    if output_format == "csv":
        rows = [{**{k: v for k, v in report.items() if k != "conditions"},
                 **r, "undefined_reasons": ";".join(r["undefined_reasons"])}
                for r in report["conditions"]]
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        # Do not turn user-supplied metadata into spreadsheet formulas.
        writer.writerows({k: "'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v
                         for k, v in row.items()} for row in rows)
        return stream.getvalue()
    if output_format != "markdown":
        raise ValueError("output format must be json, csv, or markdown")
    def money(value):
        return "undefined" if value is None else f"${value:.6f}"
    lines = ["# Cost per successful task", "",
             f"Evidence: {report['evidence']}. Cost basis: {report['cost_basis']}. Scope: model/API only.", "",
             "Quality check: " + html.escape(report["quality_check"]).replace("\n", " "), "",
             "Pricing reference: " + html.escape(report["pricing_ref"]).replace("\n", " "), "",
             "| Condition | Passed / planned | Success rate | Known spend | Total spend | Cost / success |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["conditions"]:
        rate = "undefined" if row["success_rate"] is None else f"{100 * row['success_rate']:.2f}%"
        lines.append(f"| {row['condition']} | {row['successful_tasks']} / {row['expected_tasks']} | {rate} | "
                     f"{money(row['known_cost_usd'])} | {money(row['total_cost_usd'])} | {money(row['cost_per_success_usd'])} |")
    lines += ["", "Failed attempts remain in spend. A success is counted once per task.",
              "Cost changes do not establish equivalent quality. These are descriptive results."]
    for row in report["conditions"]:
        if row["undefined_reasons"]:
            lines.append(f"{row['condition']}: {', '.join(row['undefined_reasons'])}.")
    return "\n".join(lines) + "\n"
