"""Freeze a label-blind dev sample and observed-evidence inputs. No network/model calls."""
from collections import Counter
import json
from pathlib import Path
import random
import sqlite3

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "runs/api-validation-20261017"
BASELINE = ROOT / "runs/code-tools-20261017/A"
SEED = 20261017
QUOTAS = {"basic": 8, "semantic": 6, "completeness": 2, "conflicting_info": 2,
          "constrained": 2, "high_level": 2, "info_not_found": 2,
          "intra_document_reasoning": 2, "miscellaneous": 2, "project_related": 2}


def read(path):
    return json.loads(path.read_text())


def freeze(path, value):
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f"Frozen artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    if (OUT / "calls.jsonl").exists():
        raise RuntimeError("Experiment has started; do not rebuild inputs")
    rows = read(ROOT / "data/benchmarks/enterprise/dev.json")
    # Selection uses only IDs and question types, before opening any answer or judgement.
    rng = random.Random(SEED)
    selected = []
    for category, count in sorted(QUOTAS.items()):
        pool = sorted(r["question_id"] for r in rows if r["question_type"] == category)
        selected.extend(rng.sample(pool, count))
    rng.shuffle(selected)
    smoke = [rng.choice(sorted(r["question_id"] for r in rows
                               if r["question_type"] == category and r["question_id"] not in selected))
             for category in ("basic", "semantic")]
    arms = [["low", "medium"], ["medium", "low"]] * 15
    rng.shuffle(arms)
    selection = {"seed": SEED, "population": "enterprise/dev100", "quotas": QUOTAS,
                 "selection": "random.Random(seed), sorted categories and IDs, sample each quota, shuffle",
                 "migration_ids": selected, "smoke_ids": smoke,
                 "verbosity_schedule": [{"id": q, "arms": a} for q, a in zip(selected, arms)],
                 "selection_uses_answers_judgements_or_cost": False,
                 "limitations": "Known development cohort; rare categories oversampled; not representative full-benchmark accuracy."}
    freeze(HERE / "sample.json", selection)
    cases = {r["question_id"]: r for r in rows}
    baseline = []
    evidence_stats = []
    con = sqlite3.connect((ROOT / "data/benchmarks/enterprise/corpus.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    try:
        for qid in selected + smoke:
            case = cases[qid]
            freeze(OUT / "questions" / f"{qid}.json", {"id": qid, "question": case["question"]})
            freeze(OUT / "judge-only" / f"{qid}.json", case)
            if qid in smoke:
                continue
            trace = read(BASELINE / "traces" / f"{qid}.json")
            observed = {}
            for event in trace["events"]:
                content = event.get("content", {})
                if event.get("role") != "tool" or event.get("name") not in ("search", "read") or not isinstance(content, dict):
                    continue
                hits = content.get("hits", []) if event["name"] == "search" else [content]
                for hit in hits:
                    if not hit.get("text") or not hit.get("document_id"):
                        continue
                    did, start, text = hit["document_id"], hit.get("offset", 0), hit["text"]
                    if did not in observed:
                        title, original, source = con.execute(
                            "SELECT title,text,source FROM documents WHERE id=?", (did,)).fetchone()
                        observed[did] = {"title": title, "original": original, "source": source, "spans": []}
                    assert observed[did]["original"][start:start + len(text)] == text, (qid, did, start)
                    observed[did]["spans"].append((start, start + len(text)))
            excerpts = []
            for did, doc in observed.items():
                merged = []
                for start, end in sorted(doc["spans"]):
                    if merged and start <= merged[-1][1]:
                        merged[-1][1] = max(end, merged[-1][1])
                    else:
                        merged.append([start, end])
                excerpts.extend({"document_id": did, "title": doc["title"], "source": doc["source"],
                                 "offset": start, "text": doc["original"][start:end]} for start, end in merged)
            freeze(OUT / "fixed-evidence" / f"{qid}.json", {"id": qid, "question": case["question"], "excerpts": excerpts})
            freeze(OUT / "baseline" / f"{qid}.json", {"id": qid, "result": trace["result"], "judgement": trace["judgement"]})
            facts = trace["judgement"]["facts_supported"]
            baseline.append({"id": qid, "category": case["question_type"], "correct": trace["judgement"]["correct"],
                             "completeness": sum(facts) / len(facts) if facts else 1,
                             "model_calls": len(trace["model_runs"])})
            evidence_stats.append({"id": qid, "documents": len(observed), "spans": len(excerpts),
                                   "characters": sum(len(e["text"]) for e in excerpts)})
    finally:
        con.close()
    correct = sum(row["correct"] for row in baseline)
    summary = {"baseline": str(BASELINE.relative_to(ROOT)), "n": len(selected), "correct": correct,
               "migration_screen_min_correct": max(0, correct - 3),
               "categories": dict(Counter(row["category"] for row in baseline)),
               "correct_by_category": dict(Counter(row["category"] for row in baseline if row["correct"])),
               "cases": baseline, "fixed_evidence": evidence_stats,
               "new_model_calls": 0, "paid_api_requests": 0,
               "generation_input_contains": "question; only observed corpus excerpts for answer-only arm; no old answers, reasoning, gold or judgement",
               "readiness": "design and inputs ready; dedicated key, spend guard and provider preflight required before live execution"}
    freeze(HERE / "preflight.json", summary)
    print(json.dumps({k: summary[k] for k in ("n", "correct", "migration_screen_min_correct", "categories", "new_model_calls")}, ensure_ascii=False))
    print(json.dumps({"smoke_ids": smoke, "evidence_chars_total": sum(x["characters"] for x in evidence_stats),
                      "evidence_chars_max": max(x["characters"] for x in evidence_stats)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
