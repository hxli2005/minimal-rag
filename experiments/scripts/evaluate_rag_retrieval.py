"""Development-only retrieval diagnostics; no LLM answers or oracle document scope.

Compares BM25, dense, RRF and reranking for the same unmodified question.
This is retrieval recall, not end-to-end question answering accuracy.
"""
import argparse
import json
import inspect
import statistics
import tempfile
import time
from pathlib import Path

from minimal_rag_experiments.hybrid_rag import DEFAULT_MODELS, DenseIndex, HybridKnowledgeBase, LocalModels, file_sha256
from minimal_rag.indexed_rag import IndexedKnowledgeBase
from minimal_rag_experiments.rerank_fusion import FusionKnowledgeBase
from minimal_rag_experiments.rag_optimization_eval import BASE, normalized_case


def document_ids(chunks, rows, aliases):
    return list(dict.fromkeys(aliases.get(rows[k]["document_id"], rows[k]["document_id"]) for k in chunks))


def metrics(ranked, gold):
    if not gold:
        return {}
    return {**{f"recall@{k}": len(set(ranked[:k]) & gold) / len(gold) for k in (1, 5, 10, 20, 100)},
            "all_gold@5": float(gold <= set(ranked[:5])),
            "mrr": next((1 / (i + 1) for i, doc in enumerate(ranked) if doc in gold), 0)}


def main(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    source = BASE / args.dataset / "dev.json"
    cases = [normalized_case(args.dataset, row) for row in json.loads(source.read_text())]
    if args.category:
        cases = [case for case in cases if case["category"] == args.category]
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        raise ValueError("no cases match the requested cohort")
    aliases_path = BASE / args.dataset / "duplicate-id-aliases.json"
    aliases = json.loads(aliases_path.read_text()) if aliases_path.exists() else {}
    models = LocalModels(args.models, device=args.device)
    corpus = BASE / args.dataset / "corpus.sqlite"
    dense_path = Path(args.dense_index) if args.dense_index else corpus.with_suffix(".dense")
    dense = DenseIndex(dense_path, corpus, models) if args.dataset != "hotpotqa" else None
    config = {"dataset": args.dataset, "split": "dev", "ids": [c["id"] for c in cases],
              "sample_sha256": file_sha256(source), "models": models.config,
              "category": args.category, "model_calls": 0,
              "runtime": {"device": models.device,
                          "embedding_dtype": str(next(models.embedding.parameters()).dtype),
                          "reranker_dtype": str(next(models.reranker.model.parameters()).dtype)},
              "candidates": args.candidates, "rerank": args.rerank,
              "lexical_prior": args.lexical_prior,
              "note": "Natural question only. References are used after ranking, never for scope or candidate generation.",
              "script_sha256": file_sha256(__file__)}
    config["hybrid_sha256"] = file_sha256(inspect.getfile(HybridKnowledgeBase))
    config["reranker_scores"] = "raw logits, without fp16 sigmoid saturation"
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    (output / "hybrid-source.py.txt").write_text(Path(inspect.getfile(HybridKnowledgeBase)).read_text())
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    from minimal_rag.paths import PACKAGE_DIR
    from minimal_rag_experiments.paths import EXPERIMENT_PACKAGE_DIR
    paths = [Path(__file__), *(PACKAGE_DIR / name for name in
             ("indexed_rag.py", "scope_index.py")),
             EXPERIMENT_PACKAGE_DIR / "rerank_fusion.py", EXPERIMENT_PACKAGE_DIR / "hybrid_rag.py"]
    manifest = {}
    for path in paths:
        (snapshot / path.name).write_bytes(path.read_bytes())
        manifest[path.name] = file_sha256(path)
    (snapshot / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records = []
    try:
        for case in cases:
            with tempfile.TemporaryDirectory(prefix="rag-retrieval-eval-") as directory:
                local_dense = None
                if args.dataset == "hotpotqa":
                    path = Path(directory) / "corpus.sqlite"
                    IndexedKnowledgeBase.build(path, [{"id": f"D{i}", "title": title, "text": "".join(parts), "source": f"hotpotqa://{case['id']}/D{i}"}
                        for i, (title, parts) in enumerate(case["raw"]["context"])])
                    local_dense = DenseIndex.in_memory(path, models)
                else:
                    path = corpus
                cls = FusionKnowledgeBase if args.lexical_prior else HybridKnowledgeBase
                extra = {"lexical_prior": args.lexical_prior} if args.lexical_prior else {}
                kb = cls(path, models, local_dense or dense, candidates=args.candidates, rerank=args.rerank, **extra)
                try:
                    begin = time.monotonic()
                    kb.rank_candidates(case["question"], "", False)
                    elapsed = time.monotonic() - begin
                    stages = kb.last_diagnostics
                    rows = kb.passages(list(set().union(*stages.values())))
                    ranks = {name: document_ids(ids, rows, aliases) for name, ids in stages.items()}
                    record = {"id": case["id"], "category": case["category"], "question": case["question"],
                              "seconds": elapsed, "gold_count": len(case["gold_ids"]),
                              "metrics": {name: metrics(ids, set(case["gold_ids"])) for name, ids in ranks.items()},
                              "ranked_document_ids": ranks, "candidate_chunks": rows, "stage_chunk_ids": stages}
                    records.append(record)
                    (output / f"{case['id']}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
                finally:
                    kb.close()
                    if local_dense:
                        local_dense.close()
                summary = {"completed": len(records), "requested": len(cases), "dataset": args.dataset,
                           "category": args.category, "model_calls": 0,
                           "complete": len(records) == len(cases),
                           "mean_seconds": statistics.mean(r["seconds"] for r in records), "stages": {}}
                for stage in ("lexical", "dense", "fused", "reranked"):
                    valid = [r["metrics"][stage] for r in records if r["metrics"][stage]]
                    summary["stages"][stage] = {k: statistics.mean(r[k] for r in valid) for k in valid[0]} if valid else {}
                (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                print(json.dumps({"completed": len(records), "id": case["id"], "seconds": round(elapsed, 2),
                                  "bm25_recall5": record["metrics"]["lexical"].get("recall@5"),
                                  "hybrid_rerank_recall5": record["metrics"]["reranked"].get("recall@5")}), flush=True)
    finally:
        if dense:
            dense.close()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, help="Frozen JSON experiment configuration; one entry for new experiments")
    p.add_argument("--dataset", choices=["financebench", "enterprise", "hotpotqa"])
    p.add_argument("--models", default=str(DEFAULT_MODELS))
    p.add_argument("--dense-index")
    p.add_argument("--limit", type=int)
    p.add_argument("--category", help="Exact development-case category, selected before ranking")
    p.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    p.add_argument("--candidates", type=int, default=100)
    p.add_argument("--rerank", type=int, default=100)
    p.add_argument("--lexical-prior", type=float, default=0)
    p.add_argument("--output")
    a = p.parse_args()
    if min(a.candidates, a.rerank) < 1 or (a.limit is not None and a.limit < 1):
        p.error("counts must be positive")
    if a.config:
        from minimal_rag_experiments.retrieval_scale import run_config
        run_config(a.config)
    else:
        if not a.dataset or not a.output:
            p.error("provide --config, or both --dataset and --output")
        main(a)
