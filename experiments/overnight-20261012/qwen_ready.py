"""Wait for completed full Qwen index, then CPU recall using saved query vectors.

Never loads an encoder, never touches .building, and never writes the Qwen index.
"""
import os
for setting in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[setting] = "1"

import argparse
import fcntl
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
from minimal_rag.indexed_rag import IndexedKnowledgeBase
from minimal_rag.paths import ROOT
from minimal_rag_experiments.hotpot_benchmark import save
from minimal_rag_experiments.retrieval_scale import fuse_documents
from minimal_rag_experiments.sentence_retrieval import recall_metrics, top_indices, aggregate

SAMPLE = ROOT / "runs/qwen3-chunk-scale-20261012"
FULL = ROOT / "data/benchmarks/enterprise/corpus.qwen3-0.6b.dense"
OUTPUT = ROOT / "runs/overnight-20261012/qwen-full-recall"


def ready():
    required = [FULL / name for name in ("manifest.json", "metadata.sqlite", "vectors.npy")]
    required += [SAMPLE / name for name in ("queries-qwen.npy", "evaluation-cases.json", "qwen.dense/manifest.json")]
    if not all(p.exists() for p in required):
        return False
    manifest = json.loads((FULL / "manifest.json").read_text())
    return manifest.get("complete", False)


def evaluate():
    manifest = json.loads((FULL / "manifest.json").read_text())
    sample_manifest = json.loads((SAMPLE / "qwen.dense/manifest.json").read_text())
    if manifest["embedding"] != sample_manifest["embedding"]:
        raise ValueError("Saved query encoder identity differs from completed full index")
    cases = json.loads((SAMPLE / "evaluation-cases.json").read_text())
    queries = np.load(SAMPLE / "queries-qwen.npy")
    vectors = np.load(FULL / "vectors.npy", mmap_mode="r")
    if queries.shape != (len(cases), vectors.shape[1]) or vectors.shape[0] != manifest["chunks"]:
        raise ValueError("Query order/shape or completed vector dimensions differ")
    aliases = json.loads((ROOT / "data/benchmarks/enterprise/duplicate-id-aliases.json").read_text())
    metadata = sqlite3.connect((FULL / "metadata.sqlite").as_uri() + "?mode=ro", uri=True)
    kb = IndexedKnowledgeBase(ROOT / "data/benchmarks/enterprise/corpus.sqlite")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save(OUTPUT / "config.json", {"model_calls": 0, "gpu": False, "index": str(FULL),
        "index_manifest": manifest, "query_source": str(SAMPLE / "queries-qwen.npy"),
        "case_source": str(SAMPLE / "evaluation-cases.json"), "ids": [c["id"] for c in cases],
        "fusion": "document RRF k=60, 100 candidates/path, top20", "threads": 1,
        "note": "Full-corpus known-dev retrieval recall, not answer accuracy or official score. Saved queries reused from original Qwen task; no concurrent GPU inference."})
    try:
        if metadata.execute("SELECT count(*) FROM passages").fetchone()[0] != manifest["chunks"]:
            raise ValueError("Completed metadata row count differs from manifest")
        for case, query in zip(cases, queries, strict=True):
            target = OUTPUT / f"{case['id']}.json"
            if target.exists():
                continue
            started = time.monotonic()
            scores = np.empty(len(vectors), dtype=np.float32)
            for start in range(0, len(vectors), 32768):
                scores[start:start+32768] = vectors[start:start+32768] @ query
            count = min(1000, len(scores))
            while True:
                positions = top_indices(scores, count)
                by_position = {}
                for start in range(0, len(positions), 500):
                    batch = [int(x) for x in positions[start:start+500]]
                    placeholders = ",".join("?" for _ in batch)
                    by_position.update(metadata.execute(f"SELECT position,document_id FROM passages WHERE position IN ({placeholders})", batch).fetchall())
                dense_ids = list(dict.fromkeys(aliases.get(by_position[int(p)], by_position[int(p)]) for p in positions))[:100]
                if len(dense_ids) >= 100 or count == len(scores):
                    break
                count = min(count * 2, len(scores))
            size = 100
            while True:
                found = kb.search(case["question"], size)
                lexical = list(dict.fromkeys(aliases.get(r["document_id"], r["document_id"]) for r in found["hits"]))[:100]
                if len(lexical) >= 100 or found["next_offset"] is None:
                    break
                size *= 2
            gold = list(dict.fromkeys(aliases.get(x, x) for x in case["gold_ids"]))
            ranks = {"bm25": lexical[:20], "qwen": dense_ids[:20],
                     "hybrid": fuse_documents(lexical, dense_ids, k=60, limit=20)}
            save(target, {"id": case["id"], "category": case["category"], "gold_ids": gold,
                "ranked": ranks, "metrics": {name: recall_metrics(ids, gold) for name, ids in ranks.items()},
                "seconds": time.monotonic() - started})
            print(json.dumps({"full_recall": case["id"], "seconds": time.monotonic()-started}), flush=True)
            records = [json.loads((OUTPUT / f"{c['id']}.json").read_text()) for c in cases if (OUTPUT / f"{c['id']}.json").exists()]
            save(OUTPUT / "summary.json", {"requested": len(cases), "completed": len(records),
                "complete": len(records) == len(cases), "model_calls": 0, "gpu": False,
                "groups": {name: aggregate([{**r, "metrics": r["metrics"][name]} for r in records]) for name in ranks}})
    finally:
        metadata.close()
        kb.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    consumer_lock = (OUTPUT / "consumer.lock").open("w")
    try:
        fcntl.flock(consumer_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A CPU recall consumer is already waiting or running.")
    while not ready():
        save(OUTPUT / "status.json", {"stage": "waiting_for_complete_index", "model_calls": 0,
            "gpu": False, "pid": os.getpid(), "checked_unix": time.time(), "full_index": str(FULL)})
        if not args.watch:
            print("Full Qwen index or frozen query vectors not ready; no encoding started.")
            raise SystemExit(0)
        time.sleep(60)
    evaluate()
    save(OUTPUT / "status.json", {"stage": "complete", "model_calls": 0, "gpu": False, "finished_unix": time.time()})
