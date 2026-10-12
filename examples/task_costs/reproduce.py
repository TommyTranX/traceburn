"""Reanalyze frozen observations offline. No provider SDK, credentials, or calls.

Default: print the report and check committed derived files for drift.
--write: deliberately regenerate only realm_attempts.jsonl and realm_report.*.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from traceburn.outcomes import render_summary, summarize

HERE = Path(__file__).resolve().parent


def reproduce():
    source = (HERE / "realm_observations.jsonl").read_bytes()
    provenance = json.loads((HERE / "provenance.json").read_text())
    if hashlib.sha256(source).hexdigest() != provenance["published_observations_sha256"]:
        raise ValueError("observations checksum changed; review provenance before reanalysis")
    manifest = json.loads((HERE / "realm.manifest.json").read_text())
    attempts = []
    for raw in source.decode().splitlines():
        row = json.loads(raw)
        steps = row["steps"]
        if row["repeat"] != 0 or row["condition"] not in ("baseline", "truncate"):
            raise ValueError("unexpected cohort member")
        if (len(steps) != row["n_steps"] or
                any(s["model"] != "claude-haiku-4-5-20251001" for s in steps) or
                any(s["cache_creation_input_tokens"] or s["cache_read_input_tokens"] for s in steps)):
            raise ValueError("unexpected step/model/cache accounting")
        inputs = sum(s["input_tokens"] for s in steps)
        outputs = sum(s["output_tokens"] for s in steps)
        cost = round((inputs + 5 * outputs) / 1_000_000, 6)
        if (inputs != row["total_input_tokens"] or outputs != row["total_output_tokens"]
                or abs(cost - row["cost_usd"]) > 1e-10):
            raise ValueError("recorded token totals or cost do not reconcile")
        if type(row["success"]) is not bool or (row["success"] and row["outcome"] != "submitted"):
            raise ValueError("invalid recorded grade")
        attempts.append({
            "condition": row["condition"], "task_id": row["task_id"], "attempt": 1,
            "outcome": "pass" if row["success"] else "fail", "cost_usd": cost,
            # The original SDK could retry internally; saved usage is not an invoice.
            "billing_complete": False, "model_calls": len(steps),
            "retry_calls": None, "latency_s": None,
        })
    report = summarize(attempts, manifest)
    if len(attempts) != 288 or any(r["missing_tasks"] for r in report["conditions"]):
        raise ValueError("incomplete paired cohort")
    return attempts, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate committed derived files")
    args = parser.parse_args()
    attempts, report = reproduce()
    files = {"realm_attempts.jsonl": "".join(json.dumps(r, sort_keys=True) + "\n" for r in attempts)}
    for format_name, suffix in (("markdown", "md"), ("json", "json"), ("csv", "csv")):
        files[f"realm_report.{suffix}"] = render_summary(report, format_name)
    for name, content in files.items():
        if args.write:
            (HERE / name).write_text(content, encoding="utf-8")
        elif (HERE / name).read_bytes().decode("utf-8") != content:
            raise SystemExit(f"derived file differs: {name}; inspect before using --write")
    print(render_summary(report), end="")
    print("Verified frozen token accounting and derived files. No model calls or fresh grades.")


if __name__ == "__main__":
    main()
