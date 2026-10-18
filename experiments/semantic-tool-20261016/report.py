"""Apply the prespecified paired gate and describe observed tool attribution."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from minimal_rag_experiments.benchmark import load_cases

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/semantic-tool-20261016"


def read(path):
    return json.loads(path.read_text())


def results(folder, cases, dataset):
    rows = {}
    scores = {}
    if dataset == "browsecomp":
        scores = {str(r["query_id"]): r["correct"] for r in read(folder / "evaluation_summary.json")["per_query_metrics"]}
    for case in cases:
        qid = case["id"]
        trace = read(folder / "traces" / f"{qid}.json")
        if "result" not in trace:
            raise ValueError(f"unfinished answer: {folder}/{qid}")
        if dataset == "browsecomp" and trace["result"]["status"] == "finished":
            judgement = read(folder / "judgements" / f"{qid}.json")
            if (judgement["judge_result"].get("parse_error", False)
                    or type(judgement["judge_result"].get("correct")) is not bool):
                raise ValueError(f"missing valid judge output: {folder}/{qid}")
        value = scores[qid] if dataset == "browsecomp" else trace["judgement"]["correct"]
        if type(value) is not bool:
            raise ValueError(f"missing valid judgement: {folder}/{qid}")
        rows[qid] = (int(value), trace)
    return rows


def behavior(rows, golds, aliases):
    counts, cases_with, exclusive, first, details = Counter(), Counter(), [], [], []
    for qid, (_, trace) in rows.items():
        found = defaultdict(set); called = Counter(); first_found = {}
        for event in trace.get("events", []):
            if event.get("role") != "tool": continue
            name = event["name"]; called[name] += 1
            if name not in ("search", "semantic_search"): continue
            for hit in event["content"].get("hits", []):
                doc = str(hit["document_id"]); doc = aliases.get(doc, doc)
                found[name].add(doc); first_found.setdefault(doc, name)
        counts.update(called); cases_with.update(called.keys())
        gold = golds.get(qid, set())
        unique = sorted((found["semantic_search"] - found["search"]) & gold)
        new = sorted(d for d in gold if first_found.get(d) == "semantic_search")
        if unique: exclusive.append(qid)
        if new: first.append(qid)
        details.append({"id": qid, "calls": dict(called), "semantic_exclusive_reference_ids": unique,
                        "semantic_first_reference_ids": new,
                        "all_reference_documents_found": gold <= (found["search"] | found["semantic_search"]) if gold else None})
    return {"calls": dict(counts), "cases_with_tool": dict(cases_with),
            "semantic_calls_per_question": counts["semantic_search"]/len(rows),
            "semantic_question_fraction": cases_with["semantic_search"]/len(rows),
            "semantic_tool_call_fraction": counts["semantic_search"]/sum(counts.values()) if counts else 0,
            "semantic_exclusive_reference_question_count": len(exclusive),
            "semantic_exclusive_reference_questions": exclusive, "semantic_first_reference_questions": first,
            "per_question": details}


def main(enterprise_only=False):
    groups, deltas = {}, []
    datasets = (("enterprise", 400),) if enterprise_only else (("enterprise", 400), ("browsecomp", 830))
    for dataset, size in datasets:
        _, cases = load_cases(dataset, "test" if dataset == "enterprise" else "all")
        assert len(cases) == size
        a_path = ROOT / "runs/search-description-20261016/enterprise-test" if dataset == "enterprise" else OUT / "browsecomp-A"
        a, b = results(a_path, cases, dataset), results(OUT / f"{dataset}-B", cases, dataset)
        if any(t["result"]["status"] in ("runtime_error", "interrupted") for _, t in b.values()):
            raise ValueError(f"{dataset}: technical generation retries are not complete")
        golds = {c["id"]: set(c["gold_ids"]) for c in cases}
        aliases = {}
        if dataset == "enterprise":
            aliases = read(ROOT / "data/benchmarks/enterprise/duplicate-id-aliases.json")
        else:
            for line in (ROOT / "data/benchmarks/browsecomp-plus/official/topics-qrels/qrel_evidence.txt").read_text().splitlines():
                qid, _, docid, relevance = line.split()
                if qid in golds and int(relevance) > 0: golds[qid].add(docid)
        differences = np.array([b[c["id"]][0]-a[c["id"]][0] for c in cases])
        deltas.append(differences)
        categories = {}
        for category in sorted({c["category"] for c in cases}):
            ids = [c["id"] for c in cases if c["category"] == category]
            categories[category] = {"n": len(ids), "A_correct": sum(a[q][0] for q in ids),
                                    "B_correct": sum(b[q][0] for q in ids),
                                    "wins": sum(b[q][0] > a[q][0] for q in ids),
                                    "losses": sum(b[q][0] < a[q][0] for q in ids)}
        groups[dataset] = {"n": size, "A_correct": sum(x[0] for x in a.values()),
                           "B_correct": sum(x[0] for x in b.values()),
                           "by_category": categories,
                           "A_generation_status": dict(Counter(t["result"]["status"] for _, t in a.values())),
                           "B_generation_status": dict(Counter(t["result"]["status"] for _, t in b.values())),
                           "wins_with_A_generation_failure": [c["id"] for c, d in zip(cases, differences)
                               if d > 0 and a[c["id"]][1]["result"]["status"] != "finished"],
                           "losses_with_B_generation_failure": [c["id"] for c, d in zip(cases, differences)
                               if d < 0 and b[c["id"]][1]["result"]["status"] != "finished"],
                           "wins": [c["id"] for c, d in zip(cases, differences) if d>0],
                           "losses": [c["id"] for c, d in zip(cases, differences) if d<0],
                           "ties": int((differences == 0).sum()), "B_behavior": behavior(b, golds, aliases)}
        groups[dataset]["generation_retry_accounting"] = {
            label: read(path / "generation-retry-accounting.json")
            for label, path in (("A", a_path), ("B", OUT / f"{dataset}-B"))
            if (path / "generation-retry-accounting.json").exists()}
        retried = {label: {path.stem for path in
                   (folder / "generation-retries/initial-failures").glob("*.json")}
                   for label, folder in (("A", a_path), ("B", OUT / f"{dataset}-B"))}
        involved = retried["A"] | retried["B"]
        groups[dataset]["technical_retry_pairing"] = {
            "A_retried_questions": sorted(retried["A"]), "B_retried_questions": sorted(retried["B"]),
            "wins": [c["id"] for c, d in zip(cases, differences) if d > 0 and c["id"] in involved],
            "losses": [c["id"] for c, d in zip(cases, differences) if d < 0 and c["id"] in involved],
            "ties": [c["id"] for c, d in zip(cases, differences) if d == 0 and c["id"] in involved]}
    rng = np.random.default_rng(20261016)
    # Multinomial resampling of {-1, 0, +1} is exactly paired bootstrap of correctness deltas.
    totals = np.zeros(100000)
    for differences in deltas:
        counts = np.array([(differences == d).sum() for d in (-1, 0, 1)])
        samples = rng.multinomial(len(differences), counts/len(differences), size=100000)
        totals += samples[:, 2]-samples[:, 0]
    paired_questions = sum(size for _, size in datasets)
    low, high = np.quantile(totals/paired_questions, [0.025, 0.975])
    keep = all(g["B_correct"] >= g["A_correct"] for g in groups.values()) and low > 0
    summary = {"complete": True, "benchmarks": groups,
               "scope": "enterprise_only" if enterprise_only else "both_benchmarks",
               "paired_questions": paired_questions, "promotion_gate_evaluated": not enterprise_only,
               "runtime_infrastructure": {name: read(OUT / name) for name in
                   ("B-runtime-incident.json", "429-diagnosis.json", "fifo-recovery-incident.json", "transport-policy.json")
                   if (OUT / name).exists()},
               "combined_delta": float(sum(d.sum() for d in deltas)/paired_questions),
               "paired_stratified_bootstrap_95": [float(low), float(high)],
               "bootstrap_samples": 100000, "seed": 20261016,
               "decision": "deferred_pending_browsecomp" if enterprise_only else ("keep" if keep else "remove"),
               "limitations": ["Previously used evaluation questions; interval does not capture model/judge resampling.",
                               "Semantic-exclusive reference hits describe this trace, not causal necessity.",
                               "B used shared request queues and startup backpressure after upstream 429 errors; queue time counts toward the unchanged case timeout, so wall-clock efficiency is not directly comparable to A."]}
    if enterprise_only:
        summary["limitations"].append("Enterprise-only interval is descriptive; the original adoption gate still requires both benchmarks.")
    filename = "enterprise-comparison.json" if enterprise_only else "comparison.json"
    (OUT / filename).write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k: summary[k] for k in ("complete", "combined_delta", "paired_stratified_bootstrap_95", "decision")}, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--enterprise-only", action="store_true")
    main(parser.parse_args().enterprise_only)
