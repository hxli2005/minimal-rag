"""Offline replay of observed exact-document queries, preserving old evidence."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

from minimal_rag.indexed_rag import IndexedKnowledgeBase


def main(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True  # Reading frozen source must not add cache files there.
    spec = importlib.util.spec_from_file_location("frozen_scoped_search_reference", args.baseline_module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old = module.IndexedKnowledgeBase(args.corpus)
    begin = time.monotonic()
    new = IndexedKnowledgeBase(args.corpus, scope_index=args.scope_index)
    setup_seconds = time.monotonic() - begin
    records, seen = [], set()
    try:
        for path in sorted(args.traces.glob("*/report.json")):
            report = json.loads(path.read_text())
            for event in report["events"]:
                if event.get("role") != "tool" or event.get("name") != "search":
                    continue
                query = json.loads(event["arguments_json"])
                key = json.dumps(query, sort_keys=True)
                scope = query.get("scope")
                if key in seen or not scope or not old.db.execute("SELECT 1 FROM documents WHERE id=?", (scope,)).fetchone():
                    continue
                seen.add(key)
                cached = {**event["content"], "hits": [
                    {k: v for k, v in hit.items() if k not in {"evidence_id", "passage_sha256"}}
                    for hit in event["content"]["hits"]]}
                begin = time.monotonic()
                reference = old.search(**query)
                baseline_seconds = time.monotonic() - begin
                begin = time.monotonic()
                actual = new.search(**query)
                optimized_seconds = time.monotonic() - begin
                record = {"query": query, "baseline_seconds": baseline_seconds,
                          "optimized_seconds": optimized_seconds, "hits": len(actual["hits"]),
                          "matches_cached": actual == cached, "matches_baseline": actual == reference}
                records.append(record)
                (output / f"query-{len(records)}.json").write_text(json.dumps({**record, "result": actual}, ensure_ascii=False, indent=2) + "\n")
                print(json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        old.close()
        new.close()
    summary = {"kind": "offline_known_query_replay", "model_calls": 0, "setup_seconds": setup_seconds,
               "queries": records, "all_identical": bool(records) and all(r["matches_cached"] and r["matches_baseline"] for r in records),
               "baseline_sha256": hashlib.sha256(Path(args.baseline_module).read_bytes()).hexdigest(),
               "candidate_sha256": hashlib.sha256(Path("src/minimal_rag/indexed_rag.py").read_bytes()).hexdigest(),
               "limitations": "Known exact-document queries; baseline precedes candidate and cache state may differ. This is not a service load test or an answer-quality evaluation."}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"queries": len(records), "all_identical": summary["all_identical"]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "scope-index", "traces", "baseline-module", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    main(p.parse_args())
