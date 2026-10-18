"""Portable phase summaries; reuse the repository's fixed paired bootstrap."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

from count_requests import count
from freeze import verify
from run_frozen import ROOT

sys.path.insert(0, str(ROOT / "experiments/scripts"))
from compare_rag_holdout import paired

BASE = ROOT / "data/benchmarks"
OUT = BASE / "sota-20261014"
HERE = Path(__file__).resolve().parent
REFERENCES = {
    "enterprise-dev": {
        "astra_cascade": BASE / "optimization-20261012/enterprise-dev100-bge-cascade-preview",
        "cdl": BASE / "optimization-20261011/judged/enterprise-dev100-cdl-lean",
    },
    "enterprise-test": {
        "historical_v6": BASE / "optimization-20261011/holdout/enterprise",
        "cdl": BASE / "optimization-20261011/holdout/public-comparisons/cdl",
    },
    "financebench-test": {
        "historical_v6": BASE / "optimization-20261011/holdout/financebench",
        "dewey": BASE / "optimization-20261011/holdout/public-comparisons/dewey",
    },
    "hotpotqa-test": {"historical_v6": BASE / "optimization-20261011/holdout/hotpotqa"},
}


def read(path):
    return json.loads(path.read_text())


def metrics(rows):
    result = {"n": len(rows)}
    for key in ("correct", "completeness", "overall", "em", "f1", "document_recall", "retrieved_document_recall", "model_calls"):
        values = [r[key] for r in rows if r.get(key) is not None]
        if values:
            result[key] = statistics.mean(values)
            if key in ("document_recall", "retrieved_document_recall"):
                result[key + "_n"] = len(values)
    return result


def compare(left, right, keys):
    output = {}
    for key in keys:
        ids = {r["id"] for r in left if r.get(key) is not None} & {r["id"] for r in right if r.get(key) is not None}
        if ids:
            output[key] = paired([r for r in left if r["id"] in ids], [r for r in right if r["id"] in ids], key)
    return output


def summarize(name):
    directory = OUT / name
    summary = read(directory / "summary.json")
    config = summary["config"]
    dataset = config["dataset"]
    n = len(config["ids"])
    assert summary["completed"] == n and summary["metrics"]["graded"] == n, "finish scoring before summarizing"
    rows = summary["cases"]
    assert len(rows) == len({r["id"] for r in rows}) == n
    assert {r["id"] for r in rows} == set(config["ids"])
    audit = verify(dataset, directory)
    keys = ["em", "f1"] if dataset == "hotpotqa" else ["correct", "completeness", "overall", "document_recall"]
    categories = sorted({r["category"] for r in rows})
    comparisons = {}
    sources = [directory / "summary.json", directory / "config.json"]
    for label, reference in REFERENCES[name].items():
        baseline = read(reference / "summary.json")
        right = baseline["cases"]
        assert len(right) == len({r["id"] for r in right}) == n
        assert {r["id"] for r in rows} == {r["id"] for r in right}
        ref_config = read(reference / "config.json")
        assert config["sample_sha256"] == ref_config.get("sample_sha256", ref_config.get("reference_sha256"))
        if dataset != "hotpotqa":
            assert config["judge_model"] == ref_config.get("judge_model", ref_config.get("judge"))
            assert config["judge_prompt"] == ref_config.get("judge_prompt", ref_config.get("prompt"))
        comparisons[label] = {"metrics": baseline["metrics"], "paired": compare(rows, right, keys),
            "by_category": {category: {"candidate": metrics([r for r in rows if r["category"] == category]),
                "reference": metrics([r for r in right if r["category"] == category]),
                "paired": compare([r for r in rows if r["category"] == category], [r for r in right if r["category"] == category], keys)} for category in categories}}
        primary = "em" if dataset == "hotpotqa" else "correct"
        by_id = {r["id"]: r for r in right}
        comparisons[label]["discordant_ids"] = {
            "wins": [r["id"] for r in rows if r[primary] > by_id[r["id"]][primary]],
            "losses": [r["id"] for r in rows if r[primary] < by_id[r["id"]][primary]],
        }
        sources += [reference / "summary.json", reference / "config.json"]
    requests = count(directory)
    costs = {"generation_requests": requests["generation_requests"], "judge_requests": requests["judge_requests"],
             "sol_requests": requests["sol_requests"], "generation_statuses": requests["statuses"],
             "failed_requests": requests["failed_requests"], "judge_retry_requests": requests["judge_retry_requests"],
             "mean_generation_calls": requests["generation_requests"] / n,
             "generation_input_tokens": summary["metrics"]["input_tokens"],
             "mean_generation_input_tokens": summary["metrics"]["input_tokens"] / n,
             "counting_note": requests["accounting"], "counting_limitations": requests["limitations"]}
    judge_input = missing_usage = 0
    for path in (directory / "traces").glob("*.json"):
        trace = read(path)
        attempts = trace.get("judge_attempts", [])
        runs = [r for a in attempts for r in a.get("runs", [])] if attempts else trace.get("judge_runs", [])
        missing_usage += sum((r.get("usage") or {}).get("total") is None for r in trace.get("model_runs", []) + runs)
        latest = {}
        for i, run in enumerate(runs):
            usage = (run.get("usage") or {}).get("total")
            if usage is not None:
                latest[run.get("thread_id", str(i))] = usage
        judge_input += sum(u.get("input_tokens", 0) or 0 for u in latest.values())
    costs["judge_input_tokens"] = judge_input
    costs["total_input_tokens"] = costs["generation_input_tokens"] + judge_input
    costs["requests_without_returned_token_usage"] = missing_usage
    if missing_usage:
        costs["counting_limitations"].append("Token totals include returned usage only; failed requests without usage may have consumed additional tokens.")
    if name == "enterprise-dev":
        passed = sum(r["correct"] for r in rows) >= 82 and requests["infrastructure_failed_cases"] <= 5
        criterion = "Gate only: dev correct >=82/100 and infrastructure failures <=5"
    elif name == "enterprise-test":
        low, high = comparisons["cdl"]["paired"]["correct"]["paired_bootstrap_95_interval"]
        passed, criterion = low <= 0 <= high, "Paired correctness interval versus CDL contains zero"
    elif name == "financebench-test":
        ref = comparisons["dewey"]
        passed = summary["metrics"]["correct"] >= ref["metrics"]["correct"] and ref["paired"]["correct"]["paired_bootstrap_95_interval"][0] >= -0.05
        criterion = "Correctness >= Dewey and paired interval lower bound >= -5pp"
    else:
        passed = summary["metrics"]["em"] >= 0.78 and summary["metrics"]["f1"] >= 0.87
        criterion = "Answer EM >=78% and F1 >=87% on local test100 distractor"
    result = {"phase": name, "completed": requests["completed"], "requested": n,
        "scored_cases": summary["metrics"]["graded"], "accounted_cases": n,
        "complete": requests["completed"] == n, "evaluation_finished": True,
        "evaluation_kind": config["evaluation_kind"], "metrics": summary["metrics"],
        "by_category": summary["by_category"], "costs": costs, "comparisons": comparisons,
        "infrastructure_failed_cases": requests["infrastructure_failed_cases"],
        "criterion": criterion, "passed": bool(passed), "config_audit": audit,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "limitations": ["Local sol judge, known regression tests, not official leaderboard results.",
            "Paired bootstrap: 10,000 samples, seed 20261012; captures question variation, not repeated-run randomness.",
            "Enterprise test changes retrieval fixed -> cascade plus model astra -> sol; no isolated model-effect claim.",
            "Finance corpus 368 PDFs versus Dewey 85 filings; five historical baseline-metadata exposures retained.",
            "Hotpot local test100 comes from public distractor dev data; official test leaderboard is scale reference only.",
            "Citation document recall and observed document recall are multi-turn outcomes, not retrieval recall@k."]}
    path = HERE / (name + "-summary.json")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"phase": name, "passed": passed, "metrics": metrics(rows), "costs": costs}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=REFERENCES)
    summarize(parser.parse_args().name)
