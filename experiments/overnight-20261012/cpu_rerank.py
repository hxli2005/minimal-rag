"""Post-gate offline diagnostic over existing sparse/Qwen candidates. CPU only."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")

import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder
from minimal_rag_experiments.hybrid_rag import load_model_config
from minimal_rag_experiments.sentence_retrieval import aggregate, recall_metrics
from offline import expression
from worker import ROOT, OUTPUT, read, save

torch.set_num_threads(1)
SAMPLE = ROOT / "runs/qwen3-chunk-scale-20261012"
DESTINATION = OUTPUT / "qwen-posthoc-rerank"
DESTINATION.mkdir(parents=True, exist_ok=True)
records = read(SAMPLE / "records.json")
bm25 = {r["id"]: r for r in records["bm25"]}
qwen = {r["id"]: r for r in records["qwen"]}
aliases = read(ROOT / "data/benchmarks/enterprise/duplicate-id-aliases.json")
config = load_model_config(ROOT / "models/hybrid-rag/models-minilm.json")
model = CrossEncoder(config["reranker"]["path"], device="cpu", max_length=512,
    activation_fn=torch.nn.Identity(), trust_remote_code=False, local_files_only=True)
assert all(p.device.type == "cpu" for p in model.model.parameters())
save(DESTINATION / "config.json", {"model": config["reranker"], "device": "cpu", "threads": 1,
    "dtype": str(next(model.model.parameters()).dtype), "max_length": 512, "model_calls": 0,
    "candidate_rule": "Best BM25 chunk per top100 canonical docs plus best Qwen chunk per top100 canonical docs; union unique chunk IDs, at most200 passages. Rank raw logits, then canonical document dedup, top20.",
    "limits": "Post-hoc known dev diagnostic; original gate unchanged. No new encodings, GPU or hosted model calls; not full-corpus results."})
source = sqlite3.connect((SAMPLE / "corpus.sqlite").as_uri() + "?mode=ro", uri=True)
source.row_factory = sqlite3.Row
metadata = sqlite3.connect((SAMPLE / "qwen.dense/metadata.sqlite").as_uri() + "?mode=ro", uri=True)
def passage(chunk):
    row = source.execute("SELECT rowid AS chunk_id,document_id,title,text FROM chunks WHERE rowid=?", (chunk,)).fetchone()
    return dict(row)

try:
    for case in records["bm25"]:
        target = DESTINATION / f"{case['id']}.json"
        if target.exists():
            continue
        began = time.monotonic()
        candidates, seen = {}, set()
        cursor = source.execute("SELECT rowid AS chunk_id,document_id,title,text FROM chunks WHERE chunks MATCH ? AND rank MATCH 'bm25(0,0,2,1)' ORDER BY rank", (expression(case["question"]),))
        for row in cursor:
            canonical = aliases.get(row["document_id"], row["document_id"])
            if canonical not in seen:
                seen.add(canonical); candidates[row["chunk_id"]] = dict(row)
            if len(seen) == 100:
                break
        cursor.close()
        seen = set()
        for position in qwen[case["id"]]["top_positions"]:
            chunk, raw_doc = metadata.execute("SELECT chunk_id,document_id FROM passages WHERE position=?", (position,)).fetchone()
            canonical = aliases.get(raw_doc, raw_doc)
            if canonical not in seen:
                seen.add(canonical); candidates[chunk] = passage(chunk)
            if len(seen) == 100:
                break
        rows = list(candidates.values())
        scores = np.asarray(model.predict([(case["question"], r["title"]+"\n"+r["text"]) for r in rows],
            batch_size=8, show_progress_bar=False, convert_to_numpy=True)).reshape(-1)
        assert len(scores)==len(rows) and np.isfinite(scores).all()
        order = sorted(range(len(rows)), key=lambda i:(-float(scores[i]), rows[i]["chunk_id"]))
        ranked = list(dict.fromkeys(aliases.get(rows[i]["document_id"], rows[i]["document_id"]) for i in order))[:20]
        save(target, {"id": case["id"], "category": case["category"], "gold_ids": case["gold_ids"],
            "ranked_document_ids": ranked, "metrics": recall_metrics(ranked, case["gold_ids"]),
            "candidate_passages": len(rows), "seconds": time.monotonic()-began,
            "source_positions": [{"chunk_id": rows[i]["chunk_id"], "raw_document_id": rows[i]["document_id"], "score": float(scores[i])} for i in order[:30]]})
        completed = [read(DESTINATION / f"{c['id']}.json") for c in records["bm25"] if (DESTINATION / f"{c['id']}.json").exists()]
        save(DESTINATION / "summary.json", {"requested":100,"completed":len(completed),"complete":len(completed)==100,
            "groups":aggregate(completed),"model_calls":0,"gpu":False,"post_hoc":True})
        print(json.dumps({"completed":len(completed),"id":case["id"],"seconds":round(time.monotonic()-began,2)}),flush=True)
finally:
    source.close();metadata.close()
