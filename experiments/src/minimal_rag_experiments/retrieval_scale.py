"""Configuration-driven chunk scaling, with a frozen paired promotion gate.

Imported by the common retrieval evaluation entry; deliberately has no CLI.
References affect only the explicitly labelled diagnostic corpus and scoring.
Queries and encoding never receive references. No generation or reranking.
"""
from __future__ import annotations

import gc
import hashlib
import heapq
import importlib.util
import json
import math
import os
import shutil
import statistics
import time
from pathlib import Path

import numpy as np

from minimal_rag_experiments.hybrid_rag import DenseIndex, file_sha256, read_only
from minimal_rag.indexed_rag import IndexedKnowledgeBase
from minimal_rag.paths import ROOT
from minimal_rag_experiments.sentence_retrieval import (
    Encoder, aggregate, distinct_documents, load_cases, log, recall_metrics, save, top_indices,
)


def sample_ids(ids, fraction, seed, required=()):
    """Uniform hash background, then add references; do not silently replace random draws."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    ids = set(ids)
    if not set(required) <= ids:
        raise ValueError("reference missing from corpus")
    count = math.ceil(len(ids) * fraction)
    background = set(heapq.nsmallest(count, ids,
        key=lambda doc: (hashlib.sha256((seed + ":" + doc).encode()).hexdigest(), doc)))
    return sorted(background | set(required)), sorted(set(required) - background), count


def fuse_documents(lexical, dense, *, k, limit):
    """Equal-weight document RRF; one rank per canonical document, fixed output budget."""
    scores = {}
    for ranking in (lexical, dense):
        for rank, doc in enumerate(dict.fromkeys(ranking), 1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda doc: (-scores[doc], doc))[:limit]


def promotion(records, gate):
    metric = gate["metric"]
    pairs = list(zip(records[gate["candidate"]], records[gate["baseline"]], strict=True))
    if any(a["id"] != b["id"] or a["gold_ids"] != b["gold_ids"] or a["category"] != b["category"]
           or (a["metrics"] is None) != (b["metrics"] is None) for a, b in pairs):
        raise ValueError("paired cohort differs")
    comparisons = {}
    for category, minimum in ((gate["category"], gate["minimum_gain"]), ("all", gate["overall_minimum_gain"])):
        selected = [(a, b) for a, b in pairs if category == "all" or a["category"] == category]
        deltas = [a["metrics"][metric] - b["metrics"][metric] for a, b in selected if a["metrics"] is not None]
        if not deltas:
            raise ValueError("no scored pairs")
        gain = statistics.mean(deltas)
        wins, losses = sum(d > 0 for d in deltas), sum(d < 0 for d in deltas)
        passed = gain + 1e-12 >= minimum
        if category != "all" and gate["require_more_wins_than_losses"]:
            passed = passed and wins > losses
        comparisons[category] = {"mean_gain": gain, "wins": wins, "losses": losses,
                                "scored": len(deltas), "questions": len(selected),
                                "candidate_recall": statistics.mean(a["metrics"][metric] for a, _ in selected if a["metrics"] is not None),
                                "baseline_recall": statistics.mean(b["metrics"][metric] for _, b in selected if b["metrics"] is not None),
                                "ties": len(deltas) - wins - losses, "passed": passed}
    return {"passed": all(row["passed"] for row in comparisons.values()), "comparisons": comparisons,
            "rule": gate, "note": "Known-development engineering gate, not a significance test or SOTA claim."}


class ChunkEncoder:
    """Use the existing pinned encoder with adaptive batches, behind DenseIndex's protocol."""
    def __init__(self, root, device):
        self.encoder = Encoder("qwen", root, device)
        self.dimension, self.device = self.encoder.dimension, device
        self.identity = {"embedding": {"id": "Qwen/Qwen3-Embedding-0.6B",
                        "revision": self.encoder.identity["revision"]},
                         "query_prompt": self.encoder.prompt, "max_length": self.encoder.max_length,
                         "precision": "float16"}
        self.truncated = 0

    def encode(self, texts, *, query=False, batch_size=64):
        texts = [self.encoder.prompt + text for text in texts] if query else texts
        vectors, lengths = self.encoder.encode_all(texts)
        self.truncated += int(np.sum(lengths > self.encoder.max_length))
        return vectors


def builder(root):
    path = root / "experiments/scripts/build_dense_rag_index.py"
    spec = importlib.util.spec_from_file_location("scale_dense_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freeze_encoding_inputs(path, corpus, models, chunking):
    """Resume identity covers vector inputs; application/evaluation code is irrelevant."""
    source = read_only(corpus)
    try:
        chunks = source.execute("SELECT count(*) FROM chunks").fetchone()[0]
    finally:
        source.close()
    identity = {"corpus_sha256": file_sha256(corpus), "chunks": chunks,
                "embedding": models.identity, "dimension": models.dimension,
                "chunking": chunking, "input_format": "title + newline + original chunk text",
                "vector_dtype": "float32", "normalization": "float32 L2"}
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError("encoding inputs changed: model, corpus, chunking or input format")
    else:
        save(path, identity)
    return identity


def capture_evaluation_sources(root):
    """Keep loaded source bytes in memory; persist hashes only when evaluation starts."""
    return {name: (root / name).read_bytes() for name in (
        "experiments/scripts/evaluate_rag_retrieval.py",
        "experiments/src/minimal_rag_experiments/retrieval_scale.py",
        "experiments/src/minimal_rag_experiments/sentence_retrieval.py",
        "src/minimal_rag/indexed_rag.py", "experiments/src/minimal_rag_experiments/hybrid_rag.py", "src/minimal_rag/loop.py")}


def record_evaluation_start(root, directory, config, sources):
    """Evaluation provenance/consistency never gates an encoding checkpoint."""
    identity = {"config": {name: config[name] for name in ("dataset", "evaluation_category", "fusion", "gate")},
                "dev_sha256": file_sha256(root / "data/benchmarks/enterprise/dev.json"),
                "aliases_sha256": file_sha256(root / "data/benchmarks/enterprise/duplicate-id-aliases.json"),
                "sources": {name: hashlib.sha256(blob).hexdigest() for name, blob in sources.items()}}
    path = directory / "evaluation-plan.json"
    if path.exists():
        if json.loads(path.read_text())["identity"] != identity:
            raise ValueError("evaluation already started with different code/config/cohort; use a new evaluation output")
    else:
        save(path, {"identity": identity, "recorded_at_unix": time.time(),
                    "source_capture": "Loaded at process start; hashes recorded at evaluation start."})
        snapshot = directory / "evaluation-source-snapshot"
        for name, blob in sources.items():
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
    return identity


def prepare_sample(root, directory, config, cases, aliases):
    corpus = root / "data/benchmarks/enterprise/corpus.sqlite"
    legacy = corpus.with_suffix(".dense")
    source_hash = file_sha256(corpus)
    previous = json.loads((legacy / "manifest.json").read_text())
    if not previous["complete"] or source_hash != previous["corpus_sha256"]:
        raise ValueError("legacy index/corpus mismatch")
    meta = read_only(legacy / "metadata.sqlite")
    required = set().union(*(set(c["gold_ids"]) for c in cases)) if config["preserve_references"] else set()
    # Include alias copies of required documents, matching full-corpus scoring.
    required |= {doc for doc, canonical in aliases.items() if canonical in required}
    all_ids = [r[0] for r in meta.execute("SELECT id FROM documents")]
    ids, additions, random_count = sample_ids(all_ids, config["fraction"], config["seed"], required)
    save(directory / "sample-document-ids.json", ids)
    source = read_only(corpus)
    sample = directory / "corpus.sqlite"
    stats = IndexedKnowledgeBase.build(sample, (
        dict(source.execute("SELECT id,title,text,source FROM documents WHERE id=?", (doc,)).fetchone())
        for doc in ids))
    source.close()
    # BGE baseline is a byte-preserving restriction of the existing full index.
    # Save source positions AND document mapping, rather than assume sample chunk IDs match.
    positions, documents = [], []
    for doc in ids:
        rows = meta.execute("SELECT position FROM passages WHERE document_id=? ORDER BY position", (doc,)).fetchall()
        positions.extend(r[0] for r in rows)
        documents.extend([doc] * len(rows))
    meta.close()
    if len(positions) != stats["chunks"]:
        raise ValueError("sample chunking differs from the legacy corpus")
    np.save(directory / "legacy-positions.npy", np.array(positions, dtype=np.int64))
    save(directory / "legacy-documents.json", documents)
    original = np.load(legacy / "vectors.npy", mmap_mode="r")
    vectors = np.lib.format.open_memmap(directory / "bge-vectors.npy", mode="w+", dtype=np.float32,
                                       shape=(len(positions), original.shape[1]))
    for start in range(0, len(positions), 8192):
        vectors[start:start + 8192] = original[positions[start:start + 8192]]
    vectors.flush()
    del vectors, original
    manifest = {**stats, "complete": True, "source_documents": len(all_ids),
                "source_corpus_sha256": source_hash, "sample_corpus_sha256": file_sha256(sample),
                "selection": "uniform hash 10% background plus explicit development references",
                "random_documents": random_count, "reference_additions": additions,
                "actual_document_fraction": len(ids) / len(all_ids),
                "cases_sha256": file_sha256(directory / "cases.json"),
                "bge_vectors_sha256": file_sha256(directory / "bge-vectors.npy"),
                "bge_documents_sha256": file_sha256(directory / "legacy-documents.json"),
                "legacy_manifest": previous, "hosted_model_calls": 0}
    save(directory / "sample-manifest.json", manifest)
    log(stage="sample_prepared", documents=len(ids), chunks=stats["chunks"], reference_additions=len(additions))
    return manifest


def evaluate_sample(root, directory, manifest, cases, aliases, models, fusion):
    dense = DenseIndex(directory / "qwen.dense", directory / "corpus.sqlite", models)
    kb = IndexedKnowledgeBase(directory / "corpus.sqlite")
    qwen_docs = np.array([r[0] for r in dense.meta.execute("SELECT document_id FROM passages ORDER BY position")])
    if file_sha256(directory / "bge-vectors.npy") != manifest["bge_vectors_sha256"]:
        raise ValueError("BGE subset vectors changed")
    if file_sha256(directory / "legacy-documents.json") != manifest["bge_documents_sha256"]:
        raise ValueError("BGE subset document mapping changed")
    bge_vectors = np.load(directory / "bge-vectors.npy", mmap_mode="r")
    bge_docs = np.array(json.loads((directory / "legacy-documents.json").read_text()))
    qwen_queries = models.encode([c["question"] for c in cases], query=True)
    np.save(directory / "queries-qwen.npy", qwen_queries)
    bge = Encoder("bge", root, models.device)
    queries, _ = bge.encode_all([bge.prompt + c["question"] for c in cases])
    np.save(directory / "queries-bge.npy", queries)
    del bge
    candidate_count = fusion["candidates_per_retriever"]
    output_count = fusion["output_documents"]
    if fusion["method"] != "rrf" or fusion["k"] < 1 or candidate_count < 20 or output_count != 20:
        raise ValueError("expected RRF with at least 20 candidates and exactly 20 output documents")
    available = {aliases.get(doc, doc) for doc in json.loads((directory / "sample-document-ids.json").read_text())}
    records = {name: [] for name in ("bm25", "bge", "qwen", "hybrid")}
    try:
        for i, case in enumerate(cases):
            ranks = {}
            for name in ("bm25", "bge", "qwen"):
                begin = time.monotonic()
                if name == "bm25":
                    count = candidate_count
                    while True:
                        found = kb.search(case["question"], limit=count)
                        ranked = distinct_documents([r["document_id"] for r in found["hits"]], aliases)[:candidate_count]
                        if len(ranked) >= candidate_count or found["next_offset"] is None:
                            break
                        count *= 2
                    top = []
                else:
                    matrix, query, docs = (dense.vectors, qwen_queries[i], qwen_docs) if name == "qwen" else (bge_vectors, queries[i], bge_docs)
                    scores = matrix @ query
                    count = min(candidate_count * 5, len(scores))
                    while True:
                        top = top_indices(scores, count)
                        ranked = distinct_documents(docs[top], aliases)[:candidate_count]
                        if len(ranked) >= candidate_count or count == len(scores):
                            break
                        count = min(count * 2, len(scores))
                    top = [int(p) for p in top]
                ranks[name] = ranked
                records[name].append({**case, "ranked_document_ids": ranked[:output_count],
                                      "candidate_document_ids": ranked, "top_positions": top,
                                      "gold_ids_absent_from_sample": sorted(set(case["gold_ids"]) - available),
                                      "metrics": recall_metrics(ranked[:output_count], case["gold_ids"]),
                                      "search_seconds": time.monotonic() - begin})
            begin = time.monotonic()
            fused = fuse_documents(ranks["bm25"], ranks["qwen"], k=fusion["k"], limit=output_count)
            records["hybrid"].append({**case, "ranked_document_ids": fused,
                                       "gold_ids_absent_from_sample": sorted(set(case["gold_ids"]) - available),
                                       "metrics": recall_metrics(fused, case["gold_ids"]),
                                       "search_seconds": sum(records[name][-1]["search_seconds"] for name in ("bm25", "qwen")) + time.monotonic() - begin})
            save(directory / "records.json", records)
            log(stage="scored", completed=i + 1, total=len(cases))
    finally:
        kb.close(); dense.close()
    return records


def run_config(config_path):
    root = ROOT
    config = json.loads(Path(config_path).read_text())
    if config["experiment"] != "dense_scale" or config["dataset"] != "enterprise":
        raise ValueError("unsupported configuration")
    directory = root / config["sample_output"]
    directory.mkdir(parents=True, exist_ok=True)
    status = directory / "status.json"
    current_stage = "starting"
    def state(stage, **extra):
        nonlocal current_stage
        current_stage = stage
        save(status, {"stage": stage, "pid": os.getpid(), "updated_unix": time.time(), "hosted_model_calls": 0, **extra})
        log(stage=stage, **extra)

    def finish_full(plan, models, build):
        corpus, full = root / plan["corpus"], root / plan["output"]
        # This path deliberately does not read development questions, evaluation
        # sources or current gate settings. Resume the saved promotion decision.
        freeze_encoding_inputs(directory / "full-encoding-inputs.json", corpus, models, plan["chunking"])
        state("building_full", gate=plan["gate"])
        if not full.exists():
            build(corpus, full, models, resume=True)
        check = DenseIndex(full, corpus, models)
        check.close()
        state("complete", full_index=str(full), gate=plan["gate"])

    try:
        models = ChunkEncoder(root, config["device"])
        build = builder(root).build
        full_plan_path = directory / "full-build-plan.json"
        if full_plan_path.exists():
            # Full jobs can restart after edits to core, evaluator, gate or dev data.
            finish_full(json.loads(full_plan_path.read_text()), models, build)
            return
        sources = capture_evaluation_sources(root)
        sample_manifest = directory / "sample-manifest.json"
        state("preparing_sample")
        if sample_manifest.exists():
            manifest = json.loads(sample_manifest.read_text())
        else:
            all_cases, aliases = load_cases(root)
            sample_cases = [c for c in all_cases if c["category"] == config["category"]]
            save(directory / "cases.json", sample_cases)
            save(directory / "sampling-plan.json", {name: config[name] for name in
                 ("dataset", "category", "fraction", "seed", "preserve_references")})
            manifest = prepare_sample(root, directory, config, sample_cases, aliases)
        if file_sha256(directory / "corpus.sqlite") != manifest["sample_corpus_sha256"]:
            raise ValueError("sample corpus changed")
        chunking = {name: manifest[name] for name in ("chunk_chars", "overlap")}
        freeze_encoding_inputs(directory / "sample-encoding-inputs.json", directory / "corpus.sqlite", models, chunking)
        state("building_sample", documents=manifest["documents"], chunks=manifest["chunks"])
        if not (directory / "qwen.dense").exists():
            build(directory / "corpus.sqlite", directory / "qwen.dense", models, resume=True)
        state("evaluating_sample")
        record_evaluation_start(root, directory, config, sources)
        all_cases, aliases = load_cases(root)
        cases = all_cases if config["evaluation_category"] == "all" else [c for c in all_cases if c["category"] == config["evaluation_category"]]
        save(directory / "evaluation-cases.json", cases)
        records = evaluate_sample(root, directory, manifest, cases, aliases, models, config["fusion"])
        gate = promotion(records, config["gate"])
        summary = {"complete": True, "kind": "gold_containing_10_percent_background_diagnostic",
                   "groups": {name: aggregate(rows) for name, rows in records.items()},
                   "gate": gate, "sample": manifest, "questions": len(cases),
                   "fusion": config["fusion"], "evaluation_plan_sha256": file_sha256(directory / "evaluation-plan.json"),
                   "reference_coverage": {category: {
                       "questions": len([r for r in records["bm25"] if category == "all" or r["category"] == category]),
                       "gold_occurrences_absent": sum(len(r["gold_ids_absent_from_sample"]) for r in records["bm25"] if category == "all" or r["category"] == category),
                       "note": "Absent references remain in recall denominators for every arm; no-gold cases are unscored."}
                       for category in ("all", "semantic")},
                   "input_format": "title + newline + original chunk text for both dense encoders",
                   "qwen_right_truncated_this_session": models.truncated, "hosted_model_calls": 0,
                   "limits": "Known dev100 including semantic dev25; only semantic references injected. Overall guard is within the fixed sampled corpus. Document recall only; no answer accuracy or SOTA evidence."}
        save(directory / "summary.json", summary)
        from minimal_rag_experiments.evidence_summary import publish
        publish(root / "evidence/qwen3-chunk-scale-20261012.json", summary)
        if not gate["passed"] or not config["build_full_on_pass"]:
            state("complete_without_full_build", gate=gate)
            return
        gc.collect()
        if models.device == "mps":
            models.encoder.torch.mps.empty_cache()
        corpus = root / "data/benchmarks/enterprise/corpus.sqlite"
        if file_sha256(corpus) != manifest["source_corpus_sha256"]:
            raise ValueError("full source corpus changed")
        index_info = json.loads((corpus.parent / "index.json").read_text())
        full_chunking = {name: index_info[name] for name in ("chunk_chars", "overlap")}
        # Persist the encoding contract BEFORE publishing the promotion request.
        freeze_encoding_inputs(directory / "full-encoding-inputs.json", corpus, models, full_chunking)
        plan = {"corpus": "data/benchmarks/enterprise/corpus.sqlite", "output": config["full_output"],
                "chunking": full_chunking, "gate": gate,
                "evaluation_plan_sha256": summary["evaluation_plan_sha256"],
                "summary_sha256": file_sha256(directory / "summary.json")}
        save(full_plan_path, plan)
        finish_full(plan, models, build)
    except BaseException as error:
        state("failed", failed_stage=current_stage, error=type(error).__name__, detail=str(error))
        raise
