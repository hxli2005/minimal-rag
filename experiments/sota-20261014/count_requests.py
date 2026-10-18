"""Count persisted generation/judge attempts, including failures and judge retries.

model_runs counts generation. judge_attempts[*].runs counts ALL judge attempts;
judge_runs is only a legacy fallback, never added again to judge_attempts.
Model attribution comes from config.model/config.judge_model. Historical astra
generation is included in total_requests, but is not mislabeled as sol usage.
Completion requires finished generation plus a grade (or saved Hotpot EM/F1).
SDK-internal network retries and not-yet-persisted in-flight calls are not exposed
by these traces; this is application-attempt accounting, a live lower bound.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def error_text(record):
    return " ".join(str(record.get(k, "")) for k in ("error", "error_type", "detail"))


def timeout(record):
    return bool(re.search(r"timeout|timed out", error_text(record), re.I))


def failed(run):
    return bool(run.get("error") or run.get("error_type") or run.get("status") not in (None, "started", "completed"))


def count(output):
    output = Path(output)
    config = read(output / "config.json")
    summary = read(output / "summary.json", {})
    rows = {r["id"]: r for r in summary.get("cases", [])}
    requested = len(config["ids"])
    totals, statuses = Counter(), Counter()
    infra, incidents, timeouts = [], [], []
    scores = []
    for path in sorted((output / "traces").glob("*.json")):
        trace = read(path)
        result = trace.get("result", {})
        row = rows.get(trace["id"], {})
        generation = trace.get("model_runs", [])
        attempts = trace.get("judge_attempts", [])
        judges = ([r for attempt in attempts for r in attempt.get("runs", [])]
                  if "judge_attempts" in trace else trace.get("judge_runs", []))
        runs = generation + judges
        totals["traces"] += 1
        totals["generation_requests"] += len(generation)
        totals["judge_requests"] += len(judges)
        totals["judge_retry_requests"] += sum(len(a.get("runs", [])) for a in attempts[1:])
        totals["failed_requests"] += sum(failed(r) for r in runs)
        totals["timeout_requests"] += sum(timeout(r) for r in runs)
        totals["started_requests"] += sum(r.get("status") == "started" for r in runs)
        statuses[result.get("status", "in_progress")] += 1
        generation_ok = result.get("status") == "finished" and not row.get("format_or_runtime_error")
        totals["generation_finished"] += generation_ok
        if config["dataset"] == "hotpotqa":
            scored = "em" in row and "f1" in row
            if scored:
                scores.append({"em": row["em"], "f1": row["f1"]})
        else:
            judgement = trace.get("judgement", {})
            scored = "correct" in judgement
            if scored:
                scores.append({"correct": float(judgement["correct"])})
        totals["completed"] += generation_ok and scored
        errors = [result, *runs, *attempts]
        tool_errors = [e["content"] for e in trace.get("events", [])
                       if e.get("role") == "tool" and isinstance(e.get("content"), dict)
                       and e["content"].get("error")]
        errors.extend(tool_errors)
        totals["tool_errors"] += len(tool_errors)
        totals["protocol_errors"] += sum(e.get("role") == "protocol_error" for e in trace.get("events", []))
        incident = result.get("status") == "runtime_error" or any(error_text(e).strip() for e in errors) or any(failed(r) for r in runs)
        if incident:
            incidents.append(trace["id"])
            if not (generation_ok and scored):
                infra.append(trace["id"])
        if any(timeout(e) for e in errors):
            timeouts.append(trace["id"])
    generation = totals["generation_requests"]
    judges = totals["judge_requests"]
    by_model = Counter()
    by_model[config["model"]] += generation
    by_model[config["judge_model"]] += judges
    metrics = {"scored_cases": len(scores)}
    for metric in ("correct", "em", "f1"):
        values = [r[metric] for r in scores if metric in r]
        if values:
            metrics[metric + "_sum"] = sum(values)
            metrics[metric + "_mean_scored"] = sum(values) / len(values)
            metrics[metric + "_mean_requested"] = sum(values) / requested
    if "correct_sum" in metrics:
        metrics["correct_count"] = int(metrics["correct_sum"])
    startup = read(output / "startup-failure.json", {})
    return {"output": str(output), "dataset": config["dataset"], "requested": requested,
            "completed": totals["completed"], "generation_finished": totals["generation_finished"],
            "persisted_traces": totals["traces"], "statuses": dict(statuses), **metrics,
            "generation_requests": generation, "judge_requests": judges,
            "total_requests": generation + judges, "sol_requests": by_model["gpt-6.1-sol"],
            "requests_by_model": dict(by_model), "judge_retry_requests": totals["judge_retry_requests"],
            "failed_requests": totals["failed_requests"], "started_requests": totals["started_requests"],
            "infrastructure_failed_cases": len(infra), "infrastructure_failed_ids": infra,
            "infrastructure_incident_cases": len(incidents), "timeout_cases": len(timeouts),
            "timeout_ids": timeouts, "timeout_requests": totals["timeout_requests"],
            "startup_infrastructure_failures": startup.get("startup_infrastructure_failures", 0),
            "tool_errors": totals["tool_errors"], "protocol_errors": totals["protocol_errors"],
            "accounting": "persisted application attempts; failed/retried runs included; judge_runs not double-counted",
            "limitations": ["Live counts exclude calls not yet saved to a trace; SDK-internal HTTP retries are not recorded.",
                            "completed excludes generation failures even when runner assigns an automatic incorrect grade.",
                            "Missing/unscored cases remain in the requested denominator; mean_scored uses only saved grades.",
                            "Startup failures can only be counted when a startup-failure.json record exists."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(count(args.output_directory), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
