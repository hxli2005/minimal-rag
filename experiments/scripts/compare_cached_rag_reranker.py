"""Re-rank development-only cached passages without changing candidate retrieval.

Questions and original passages are the only inference inputs. Labels are read
only to score the resulting ranks. No hosted model is called by this script.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from minimal_rag_experiments.hybrid_rag import file_sha256, rrf
from minimal_rag_experiments.cascade_rag import load_models
from minimal_rag_experiments.rag_optimization_eval import BASE, normalized_case


def metrics(order, gold):
    if not gold:
        return {}
    return {**{f"recall@{k}": len(set(order[:k]) & gold) / len(gold) for k in (1, 5, 10, 20, 100)},
            "all_gold@5": float(gold <= set(order[:5])),
            "mrr": next((1 / (i + 1) for i, doc in enumerate(order) if doc in gold), 0.0)}


def main(args):
    output, cache = Path(args.output), Path(args.cache)
    if output.exists():
        raise FileExistsError(output)
    cache_config = json.loads((cache / "config.json").read_text())
    if cache_config["split"] != "dev":
        raise ValueError("only cached development candidates may be used for selection")
    dataset = cache_config["dataset"]
    cases = [normalized_case(dataset, r) for r in json.loads((BASE / dataset / "dev.json").read_text())]
    wanted = set(cache_config["ids"])
    cases = [c for c in cases if c["id"] in wanted]
    if args.limit:
        cases = cases[:args.limit]
    aliases_path = BASE / dataset / "duplicate-id-aliases.json"
    aliases = json.loads(aliases_path.read_text()) if aliases_path.exists() else {}
    models = load_models(args.models)
    output.mkdir(parents=True)
    config = {"dataset": dataset, "split": "dev", "models": models.config,
              "cache_config_sha256": file_sha256(cache / "config.json"), "script_sha256": file_sha256(__file__),
              "ids": [c["id"] for c in cases], "rerank": args.rerank,
              "inference_inputs": "Question plus title and original passage text only; no labels or gold scope."}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "script.py.txt").write_text(Path(__file__).read_text())
    results = []
    for case in cases:
        record = json.loads((cache / f"{case['id']}.json").read_text())
        stages = record["stage_chunk_ids"]
        fused, fusion_scores = rrf([stages["lexical"], stages["dense"]])
        selected = fused[:args.rerank]
        passages = {int(k): v for k, v in record["candidate_chunks"].items()}
        begin = time.monotonic()
        scores = models.rank(case["question"], [passages[k]["title"] + "\n" + passages[k]["text"] for k in selected])
        order = sorted(zip(selected, map(float, scores)), key=lambda x: (-x[1], -fusion_scores[x[0]], x[0]))
        elapsed = time.monotonic() - begin
        docs = list(dict.fromkeys(aliases.get(passages[k]["document_id"], passages[k]["document_id"]) for k, _ in order))
        result = {"id": case["id"], "category": case["category"], "seconds": elapsed,
                  "scores": order, "ranked_document_ids": docs,
                  "metrics": metrics(docs, set(case["gold_ids"])), "previous_metrics": record["metrics"]["reranked"]}
        results.append(result)
        (output / f"{case['id']}.json").write_text(json.dumps(result, indent=2) + "\n")
        summary = {"completed": len(results), "requested": len(cases), "mean_seconds": statistics.mean(r["seconds"] for r in results)}
        for name in ("metrics", "previous_metrics"):
            valid = [r[name] for r in results if r[name]]
            summary[name] = {key: statistics.mean(r[key] for r in valid) for key in valid[0]} if valid else {}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({"completed": len(results), "id": case["id"], "seconds": round(elapsed, 2),
                          "recall5": result["metrics"].get("recall@5"), "previous_recall5": result["previous_metrics"].get("recall@5")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rerank", type=int, default=200)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.rerank < 1 or args.limit is not None and args.limit < 1:
        parser.error("counts must be positive")
    main(args)
