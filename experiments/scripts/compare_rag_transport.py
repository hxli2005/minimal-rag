"""Summarize the same development questions before/after the transport change."""
import hashlib
import argparse
import inspect
import json
from pathlib import Path

from minimal_rag.loop import run_agent
from minimal_rag_experiments.rag_optimization_eval import aggregate

from minimal_rag.paths import ROOT
BASE = ROOT / "data/benchmarks/optimization-20261011"
PAIRS = [
    ("hotpotqa", "hotpot-dev100-v5", "hotpot-dev20-v5-lean"),
    ("financebench", "finance-dev50-v5-original", "finance-dev15-v5-lean"),
    ("enterprise", "enterprise-dev20-v5", "enterprise-dev20-v5-lean"),
]


def main(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("historical evidence is not overwritten; choose a new output")
    probe = json.loads((BASE / "transport/prompt-overhead.json").read_text())
    extra = json.loads((BASE / "transport/prompt-overhead-disabled-mcp.json").read_text())
    arms = probe["arms"] + [extra]
    report = {
        "change": "Override Codex base instructions, disable unrelated context/tools locally, reuse one ephemeral thread per question, append only observations, count cumulative usage once per thread.",
        "core_lines": len(inspect.getsource(run_agent).rstrip().splitlines()),
        "core_sha256": hashlib.sha256(inspect.getsource(run_agent).rstrip().encode()).hexdigest(),
        "short_request_probe": [{"arm": row["mode"], **row["usage"]["total"]} for row in arms],
        "regressions": [],
        "limitations": [
            "Short-request probe has one request per arm; its token reduction is not a universal reduction for long RAG tasks.",
            "Input includes cached tokens. Cached tokens are reported separately, not treated as free or as a known subscription-price equivalent.",
            "Incremental transport does not remove history from model context. Earlier context still contributes to per-turn input usage.",
            "Development samples are small and reused for tuning; no heldout or frontier-quality claim.",
            "The judge also uses the revised adapter. Semantic scores across old/new runs are regression diagnostics, not a controlled judge-isolated experiment.",
            "Resource timeouts and process concurrency differ as recorded; latency is descriptive, not a speed benchmark.",
        ],
    }
    for dataset, old_name, new_name in PAIRS:
        previous = json.loads((BASE / "runs" / old_name / "summary.json").read_text())
        current_path = BASE / "runs" / new_name / "summary.json"
        if not current_path.exists():
            report["regressions"].append({"dataset": dataset, "complete": False})
            continue
        current = json.loads(current_path.read_text())
        wanted = set(current["config"]["ids"])
        before = [row for row in previous["cases"] if row["id"] in wanted]
        after = current["cases"]
        if {row["id"] for row in before} != wanted:
            raise ValueError("baseline questions are missing")
        complete = current["completed"] == len(wanted) and (
            dataset == "hotpotqa" or all(row.get("graded") for row in after))
        row = {"dataset": dataset, "complete": complete, "n_requested": len(wanted),
               "before_run": old_name, "after_run": new_name,
               "before": aggregate(before), "after": aggregate(after),
               "same_retriever": previous["config"]["retriever_sha256"] == current["config"]["retriever_sha256"],
               "same_model": previous["config"]["model"] == current["config"]["model"],
               "same_format": previous["config"]["format_request"] == current["config"]["format_request"],
               "resource_settings": {label: {k: config[k] for k in ("timeout", "case_timeout", "concurrency")}
                                     for label, config in (("before", previous["config"]), ("after", current["config"]))}}
        row["attempts_without_usage"] = {}
        for label, run_name in (("before", old_name), ("after", new_name)):
            missing = 0
            for qid in wanted:
                trace = BASE / "runs" / run_name / "traces" / f"{qid}.json"
                if trace.exists():
                    data = json.loads(trace.read_text())
                    missing += sum(not call.get("usage") for call in data.get("model_runs", []))
            row["attempts_without_usage"][label] = missing
        if complete:
            if not any(row["attempts_without_usage"].values()):
                row["input_reduction_fraction"] = 1 - row["after"]["input_tokens"] / row["before"]["input_tokens"]
                uncached_before = row["before"]["input_tokens"] - row["before"]["cached_input_tokens"]
                uncached_after = row["after"]["input_tokens"] - row["after"]["cached_input_tokens"]
                row["uncached_input_reduction_fraction"] = 1 - uncached_after / uncached_before
            old_map = {x["id"]: x for x in before}
            metric = "em" if dataset == "hotpotqa" else "correct"
            row["answer_changes"] = [{"id": x["id"], "before": old_map[x["id"]][metric], "after": x[metric]}
                                     for x in after if old_map[x["id"]][metric] != x[metric]]
        report["regressions"].append(row)
    report["complete"] = all(row["complete"] for row in report["regressions"])
    path = output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"path": str(path), "complete": report["complete"],
                      "regressions": [{k: r.get(k) for k in ("dataset", "complete", "before", "after", "input_reduction_fraction")}
                                      for r in report["regressions"]]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    main(parser.parse_args().output)
