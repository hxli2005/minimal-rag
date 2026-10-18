"""Bounded, local-only encoder/granularity diagnostic over a fixed candidate corpus.

The candidate pool deliberately contains development gold documents. It is NOT
full-corpus retrieval evaluation. No generation, judging or hosted embeddings.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import inspect
import json
import os
import re
import resource
import shutil
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from minimal_rag_experiments.hybrid_rag import file_sha256, load_model_config, read_only
from minimal_rag.indexed_rag import IndexedKnowledgeBase
from minimal_rag_experiments.sentences import SPLITTER_VERSION, split_sentences


BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
QWEN_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
QWEN_PROMPT = ("Instruct: Given a question, retrieve relevant passages from enterprise documents "
               "that help answer it.\nQuery: ")
BGE_PROMPT = "Represent this sentence for searching relevant passages: "
KINDS = ("chunk", "sentence")


def save(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def log(**fields):
    print(json.dumps(fields, ensure_ascii=False), flush=True)


def memory_usage():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"process_peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024}


def top_indices(scores, count):
    """Exact top-k with deterministic row-index tie breaking, including cutoff ties."""
    scores = np.asarray(scores)
    if scores.ndim != 1 or not np.isfinite(scores).all() or count < 1:
        raise ValueError("invalid ranking input")
    count = min(count, len(scores))
    if not count:
        return np.array([], dtype=np.int64)
    threshold = np.partition(scores, len(scores) - count)[len(scores) - count]
    above = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)[:count - len(above)]
    selected = np.concatenate((above, tied))
    return selected[np.lexsort((selected, -scores[selected]))]


def distinct_documents(unit_documents, aliases=None):
    aliases = aliases or {}
    return list(dict.fromkeys(aliases.get(str(doc), str(doc)) for doc in unit_documents))


def recall_metrics(ranked, gold):
    gold = set(gold)
    if not gold:
        return None
    return {**{f"recall@{k}": len(set(ranked[:k]) & gold) / len(gold) for k in (5, 10, 20)},
            "all_gold@20": float(gold <= set(ranked[:20]))}


def aggregate(records):
    groups = {}
    for category in ["all", *sorted({r["category"] for r in records})]:
        selected = [r for r in records if category == "all" or r["category"] == category]
        valid = [r["metrics"] for r in selected if r["metrics"] is not None]
        groups[category] = {"questions": len(selected), "scored": len(valid),
                            "no_gold": len(selected) - len(valid),
                            **({key: statistics.mean(v[key] for v in valid) for key in valid[0]} if valid else {})}
    return groups


def lexical_query(query):
    terms = list(dict.fromkeys(a or b for a, b in re.findall(r'"([^"\n]+)"|([\w]+)', query)))
    return " OR ".join('"' + word.replace('"', '""') + '"' for word in terms)


class Encoder:
    """Only local SentenceTransformer inference; no reranker or generation client."""
    def __init__(self, name, root, device="mps"):
        import torch
        from sentence_transformers import SentenceTransformer
        self.torch, self.name, self.device = torch, name, device
        torch.set_num_threads(4)
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; do not silently fall back to a slow CPU run")
        if name == "bge":
            cfg = load_model_config(Path(root) / "models/hybrid-rag/models-minilm.json")["embedding"]
            assert cfg["revision"] == BGE_REVISION
            self.max_length, self.prompt, self.batch_size, self.padded_tokens = 512, BGE_PROMPT, 64, 32768
        elif name == "qwen":
            raw = json.loads((Path(root) / "models/sentence-rag/embedding-qwen3-0.6b.json").read_text())
            if not raw["complete"] or raw["revision"] != QWEN_REVISION:
                raise ValueError("incomplete or unexpected Qwen revision")
            cfg = {"id": raw["model"], "revision": raw["revision"], "path": raw["path"]}
            self.max_length, self.prompt, self.batch_size, self.padded_tokens = 8192, QWEN_PROMPT, 32, 8192
        else:
            raise ValueError(name)
        self.identity = {"id": cfg["id"], "revision": cfg["revision"], "max_length": self.max_length,
                         "query_prompt": self.prompt, "document_prompt": "", "document_input": "original unit body only",
                         "model_dtype": "float16", "vector_dtype": "float32", "normalize": "float32 L2",
                         "overflow": "explicitly audited native-window right truncation in this offline diagnostic",
                         "batch_size_max": self.batch_size, "padded_tokens_max": self.padded_tokens,
                         "device": device, "encoder_source_sha256": hashlib.sha256(inspect.getsource(Encoder).encode()).hexdigest()}
        started = time.monotonic()
        self.model = SentenceTransformer(cfg["path"], device=device, local_files_only=True,
            trust_remote_code=False, model_kwargs={"dtype": torch.float16, "attn_implementation": "sdpa"})
        self.model.max_seq_length = self.max_length
        if next(self.model.parameters()).dtype != torch.float16:
            raise ValueError("model did not load in the declared precision")
        self.model.tokenizer.truncation_side = "right"
        if name == "qwen":
            self.model.tokenizer.padding_side = "left"
        self.dimension = self.model.get_embedding_dimension()
        self.identity["dimension"] = self.dimension
        self.load_seconds = time.monotonic() - started
        self.model.eval()
        log(stage="model_loaded", name=name, seconds=self.load_seconds, dimension=self.dimension,
            dtype=str(next(self.model.parameters()).dtype), **memory_usage())

    def lengths(self, texts):
        result = []
        for begin in range(0, len(texts), 1024):
            ids = self.model.tokenizer(texts[begin:begin + 1024], truncation=False, padding=False,
                return_attention_mask=False, return_token_type_ids=False)["input_ids"]
            result.extend(map(len, ids))
        return np.asarray(result, dtype=np.int32)

    def batches(self, order, lengths, start=0):
        cursor = start
        while cursor < len(order):
            end = cursor + 1
            longest = min(int(lengths[order[cursor]]), self.max_length)
            while end < len(order) and end - cursor < self.batch_size:
                prospective = max(longest, min(int(lengths[order[end]]), self.max_length))
                if prospective * (end - cursor + 1) > self.padded_tokens:
                    break
                longest, end = prospective, end + 1
            yield order[cursor:end]
            cursor = end

    def encode(self, texts):
        with self.torch.inference_mode():
            vectors = self.model.encode(texts, prompt="", batch_size=len(texts), show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=False)
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if vectors.shape != (len(texts), self.dimension) or not np.isfinite(vectors).all() or np.any(norms < 1e-12):
            raise ValueError("invalid embedding")
        if self.device == "mps" and self.torch.mps.driver_allocated_memory() > 16 * 1024**3:
            raise MemoryError("MPS driver allocation exceeds the 16 GiB local experiment guard")
        return vectors / norms

    def encode_all(self, texts):
        lengths = self.lengths(texts)
        order = np.argsort(lengths, kind="stable")
        vectors = np.empty((len(texts), self.dimension), dtype=np.float32)
        for indices in self.batches(order, lengths):
            vectors[indices] = self.encode([texts[int(i)] for i in indices])
        return vectors, lengths


def load_cases(root):
    path = Path(root) / "data/benchmarks/enterprise/dev.json"
    aliases = json.loads((path.parent / "duplicate-id-aliases.json").read_text())
    return [{"id": row["question_id"], "category": row["question_type"], "question": row["question"],
             "gold_ids": sorted({aliases.get(i, i) for i in row["expected_doc_ids"]})}
            for row in json.loads(path.read_text())], aliases


def prepare(root, output, device):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    corpus = root / "data/benchmarks/enterprise/corpus.sqlite"
    dense_dir = corpus.with_suffix(".dense")
    log(stage="verify_original_corpus")
    corpus_hash = file_sha256(corpus)
    original = json.loads((dense_dir / "manifest.json").read_text())
    if corpus_hash != original["corpus_sha256"] or not original["complete"]:
        raise ValueError("original dense index/corpus identity mismatch")
    cases, aliases = load_cases(root)
    source, meta = read_only(corpus), read_only(dense_dir / "metadata.sqlite")
    kb = IndexedKnowledgeBase(corpus)
    reasons = defaultdict(set)
    for case in cases:
        for doc in case["gold_ids"]:
            reasons[doc].add("gold")
            for alias, canonical in aliases.items():
                if canonical == doc:
                    reasons[alias].add("gold_alias")
    encoder = Encoder("bge", root, device)
    assert encoder.dimension == original["dimension"]
    vectors = np.load(dense_dir / "vectors.npy", mmap_mode="r")
    queries, _ = encoder.encode_all([encoder.prompt + case["question"] for case in cases])
    selections = []
    for i, (case, query) in enumerate(zip(cases, queries)):
        begin = time.monotonic()
        lexical = [h["document_id"] for h in kb.search(case["question"], limit=20)["hits"]]
        scores = vectors @ query
        count = 512
        while True:
            positions = top_indices(scores, count)
            docs = {r["position"]: r["document_id"] for r in meta.execute(
                "SELECT position,document_id FROM passages WHERE position IN (" + ",".join("?" for _ in positions) + ")",
                [int(p) for p in positions])}
            semantic = distinct_documents([docs[int(p)] for p in positions])[:20]
            if len(semantic) == 20 or count >= len(scores):
                break
            count *= 2
        for method, ids in (("bm25", lexical), ("legacy_dense", semantic)):
            for doc in ids:
                reasons[doc].add(method)
        selections.append({"id": case["id"], "bm25": lexical, "legacy_dense": semantic,
                           "seconds": time.monotonic() - begin, "dense_units_examined": len(positions)})
        save(output / "pool-selection.json", selections)
        if (i + 1) % 10 == 0:
            log(stage="pool_selection", completed=i + 1, total=len(cases), documents=len(reasons))
    # The encoder never receives the gold list. It is only used by this explicit
    # candidate-corpus selector and by scoring after all ranks have been produced.
    negatives = heapq.nsmallest(1000, (r[0] for r in source.execute("SELECT id FROM documents") if r[0] not in reasons),
        key=lambda value: (hashlib.sha256(("20261012:" + value).encode()).hexdigest(), value))
    for doc in negatives:
        reasons[doc].add("background_hash")
    if len(reasons) > 6000:
        raise ValueError("candidate corpus exceeds the explicit 6000-document safety cap")
    documents = []
    for doc in sorted(reasons):
        row = source.execute("SELECT * FROM documents WHERE id=?", (doc,)).fetchone()
        if row is None:
            raise ValueError(f"candidate document missing: {doc}")
        documents.append(dict(row))
    chunks_by_document = defaultdict(list)
    for row in meta.execute("SELECT chunk_id,document_id FROM passages ORDER BY position"):
        if row["document_id"] in reasons:
            chunks_by_document[row["document_id"]].append(row["chunk_id"])
    database = output / "units.sqlite"
    db = sqlite3.connect(database)
    db.execute("CREATE TABLE documents(id TEXT PRIMARY KEY,title TEXT,text TEXT,source TEXT)")
    db.executemany("INSERT INTO documents VALUES (:id,:title,:text,:source)", documents)
    stats = {}
    for kind in KINDS:
        db.execute(f"CREATE TABLE {kind}_units(id INTEGER PRIMARY KEY,document_id TEXT,offset INTEGER,end_offset INTEGER,text TEXT,unit_type TEXT,source_chunk_id INTEGER)")
        db.execute(f"CREATE VIRTUAL TABLE {kind}_fts USING fts5(text, tokenize='unicode61')")
        lengths, types, total = [], Counter(), 0
        for doc in documents:
            if kind == "chunk":
                spans = []
                for chunk in chunks_by_document[doc["id"]]:
                    row = source.execute("SELECT start,text FROM chunks WHERE rowid=?", (chunk,)).fetchone()
                    spans.append((int(row["start"]), int(row["start"]) + len(row["text"]), row["text"], "chunk", chunk))
            else:
                spans = [(u.offset, u.end_offset, u.text, u.unit_type, None) for u in split_sentences(doc["text"])]
            for begin, end, body, unit_type, chunk_id in spans:
                if doc["text"][begin:end] != body:
                    raise ValueError("source span mismatch")
                total += 1
                db.execute(f"INSERT INTO {kind}_units VALUES (?,?,?,?,?,?,?)", (total, doc["id"], begin, end, body, unit_type, chunk_id))
                db.execute(f"INSERT INTO {kind}_fts(rowid,text) VALUES (?,?)", (total, body))
                lengths.append(len(body)); types[unit_type] += 1
        db.execute(f"INSERT INTO {kind}_fts({kind}_fts) VALUES ('optimize')")
        stats[kind] = {"units": total, "characters": sum(lengths), "max_chars": max(lengths),
                       "p50_chars": float(np.median(lengths)), "p95_chars": float(np.percentile(lengths, 95)),
                       "unit_types": dict(types)}
    db.commit(); db.close(); kb.close(); source.close(); meta.close()
    selection = [{"id": doc["id"], "reasons": sorted(reasons[doc["id"]]),
                  "text_sha256": hashlib.sha256(doc["text"].encode()).hexdigest()} for doc in documents]
    save(output / "documents.json", selection)
    save(output / "cases.json", cases)
    save(output / "aliases.json", aliases)
    snapshot = output / "source-snapshot"; snapshot.mkdir()
    from minimal_rag_experiments import sentences
    sources = [Path(__file__), Path(sentences.__file__), root / "docs/experiment-protocol.md"]
    for path in sources:
        shutil.copyfile(path, snapshot / path.name)
    manifest = {"kind": "gold_containing_candidate_corpus_diagnostic", "complete": True,
                "corpus_sha256": corpus_hash, "dev_sha256": file_sha256(root / "data/benchmarks/enterprise/dev.json"),
                "cases_sha256": file_sha256(output / "cases.json"), "units_sha256": file_sha256(database),
                "aliases_sha256": file_sha256(output / "aliases.json"), "documents_sha256": file_sha256(output / "documents.json"),
                "documents": len(documents), "gold_documents": sum("gold" in r["reasons"] for r in selection),
                "random_background_documents": 1000, "questions": len(cases), "semantic_questions": sum(c["category"] == "semantic" for c in cases),
                "no_gold_questions": sum(not c["gold_ids"] for c in cases), "unit_stats": stats,
                "splitter_version": SPLITTER_VERSION, "source_hashes": {p.name: file_sha256(p) for p in sources},
                "hosted_model_call_cap": 0, "hosted_model_calls": 0, "no_full_index_build": True,
                "encoding_overflow_policy": "Native encoder windows; explicitly count right-truncated units. Offline diagnostic only, not full sentence-index acceptance.",
                "comparison": "BGE-small / Qwen3-Embedding-0.6B x chunk / sentence, body-only, native dimensions, exact float32 ranking, no reranker"}
    save(output / "manifest.json", manifest)
    log(stage="prepared", **manifest)


def load_run(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if not manifest["complete"] or file_sha256(directory / "units.sqlite") != manifest["units_sha256"]:
        raise ValueError("candidate corpus incomplete or changed")
    if file_sha256(directory / "cases.json") != manifest["cases_sha256"]:
        raise ValueError("question set changed")
    for name in ("aliases", "documents"):
        if file_sha256(directory / f"{name}.json") != manifest[f"{name}_sha256"]:
            raise ValueError(f"{name} changed")
    return directory, manifest


def lengths_report(lengths, limit):
    return {"units": len(lengths), "original_tokens": int(lengths.sum()),
            "encoded_tokens": int(np.minimum(lengths, limit).sum()),
            "right_truncated_units": int(np.sum(lengths > limit)), "max_original_tokens": int(lengths.max()),
            "p50_tokens": float(np.median(lengths)), "p95_tokens": float(np.percentile(lengths, 95))}


def probe(root, directory, model, device):
    directory, manifest = load_run(directory)
    destination = directory / f"probe-{model}.json"
    if destination.exists():
        raise FileExistsError(destination)
    db = read_only(directory / "units.sqlite")
    sampled = []
    for kind in KINDS:
        rows = db.execute(f"SELECT id,text FROM {kind}_units ORDER BY length(text),id").fetchall()
        if len(rows) < 512:
            raise ValueError("probe requires at least 512 units of each kind")
        # Equal-width quantiles across the empirical length distribution.
        for position in np.linspace(0, len(rows) - 1, 512).astype(int):
            sampled.append({"kind": kind, "id": rows[position]["id"], "text": rows[position]["text"]})
    db.close()
    encoder = Encoder(model, root, device)
    timing = []; report = {}
    for kind in KINDS:
        texts = [r["text"] for r in sampled if r["kind"] == kind]
        started = time.monotonic(); vectors, lengths = encoder.encode_all(texts)
        seconds = time.monotonic() - started; timing.append(seconds)
        report[kind] = {**lengths_report(lengths, encoder.max_length), "seconds": seconds,
                        "units_per_second": len(texts) / seconds, "encoded_tokens_per_second": int(np.minimum(lengths, encoder.max_length).sum()) / seconds,
                        "vector_norm_max_error": float(np.max(abs(np.linalg.norm(vectors, axis=1) - 1)))}
        log(stage="probe", model=model, kind=kind, **report[kind])
    record = {"model": encoder.identity, "model_load_seconds": encoder.load_seconds, "results": report,
              "sample": [{k: r[k] for k in ("kind", "id")} for r in sampled],
              "sample_sha256": digest([(r["kind"], r["id"]) for r in sampled]),
              "units_sha256": manifest["units_sha256"], "hosted_model_calls": 0, **memory_usage()}
    save(destination, record)


def encode_run(root, directory, model, kinds, device, max_seconds):
    directory, manifest = load_run(directory)
    if not (directory / f"probe-{model}.json").exists():
        raise ValueError("run the cost probe before encoding the candidate corpus")
    encoder = Encoder(model, root, device)
    db = read_only(directory / "units.sqlite")
    deadline = time.monotonic() + max_seconds
    for kind in kinds:
        rows = db.execute(f"SELECT id,text FROM {kind}_units ORDER BY id").fetchall()
        texts = [r["text"] for r in rows]
        identity = {"encoder": encoder.identity, "granularity": kind, "units": len(rows),
                    "units_sha256": manifest["units_sha256"], "cases_sha256": manifest["cases_sha256"]}
        final = directory / f"{model}-{kind}"
        working = directory / f"{model}-{kind}.building"
        if final.exists():
            previous = json.loads((final / "manifest.json").read_text())
            if previous["identity"] != identity or not previous["complete"]:
                raise ValueError("completed embedding identity differs")
            log(stage="already_encoded", model=model, kind=kind)
            continue
        working.mkdir(exist_ok=True)
        config = working / "config.json"
        if config.exists() and json.loads(config.read_text()) != identity:
            raise ValueError("cannot resume a changed encoder, unit corpus or question set")
        if not config.exists():
            save(config, identity)
        length_path = working / "lengths.npy"
        tokenization_started = time.monotonic()
        if length_path.exists():
            lengths = np.load(length_path)
        else:
            lengths = encoder.lengths(texts); np.save(length_path, lengths)
        tokenization_seconds_this_session = time.monotonic() - tokenization_started
        if lengths.shape != (len(texts),):
            raise ValueError("invalid token lengths")
        order = np.argsort(lengths, kind="stable")
        progress_path = working / "progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"completed": 0, "encoding_seconds": 0.0}
        done = progress["completed"]
        path = working / "vectors.npy"
        if done and not path.exists():
            raise ValueError("progress exists without vectors")
        vectors = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=np.float32, shape=(len(texts), encoder.dimension))
        if vectors.shape != (len(texts), encoder.dimension) or vectors.dtype != np.float32 or not 0 <= done <= len(texts):
            raise ValueError("invalid partial vectors/progress")
        elapsed = progress["encoding_seconds"]; last_log = 0
        for indices in encoder.batches(order, lengths, start=done):
            if time.monotonic() >= deadline:
                raise TimeoutError("local encoding time cap reached; checkpoint preserved")
            begin = time.monotonic()
            block = encoder.encode([texts[int(i)] for i in indices])
            vectors[indices] = block; vectors.flush()
            elapsed += time.monotonic() - begin; done += len(indices)
            progress = {"completed": done, "total": len(texts), "encoding_seconds": elapsed,
                        "units_per_second": done / max(elapsed, 1e-9), **memory_usage()}
            save(progress_path, progress)
            if time.monotonic() - last_log >= 30 or done == len(texts):
                log(stage="encoding", model=model, kind=kind, **progress); last_log = time.monotonic()
        for begin in range(0, len(vectors), 8192):
            block = np.asarray(vectors[begin:begin + 8192])
            if not np.isfinite(block).all() or not np.allclose(np.linalg.norm(block, axis=1), 1, atol=1e-5):
                raise ValueError("embedding validation failed")
        del vectors
        record = {"identity": identity, "complete": True, "encoding": progress,
                  "model_load_seconds": encoder.load_seconds, "tokenization_seconds_this_session": tokenization_seconds_this_session,
                  "token_lengths": lengths_report(lengths, encoder.max_length),
                  "vectors_sha256": file_sha256(path), "vector_bytes": path.stat().st_size,
                  "hosted_model_calls": 0}
        save(working / "manifest.json", record)
        working.replace(final)
        log(stage="encoded", model=model, kind=kind, **record["token_lengths"])
    db.close()
    query_path = directory / f"queries-{model}.npy"
    query_manifest = directory / f"queries-{model}.json"
    query_identity = {"encoder": encoder.identity, "cases_sha256": manifest["cases_sha256"]}
    if query_path.exists():
        saved = json.loads(query_manifest.read_text())
        if saved["identity"] != query_identity or saved["sha256"] != file_sha256(query_path):
            raise ValueError("query embeddings changed")
    else:
        cases = json.loads((directory / "cases.json").read_text())
        begin = time.monotonic()
        query_vectors, lengths = encoder.encode_all([encoder.prompt + c["question"] for c in cases])
        np.save(query_path, query_vectors)
        save(query_manifest, {"identity": query_identity, "sha256": file_sha256(query_path),
                              "seconds": time.monotonic() - begin, "lengths": lengths_report(lengths, encoder.max_length)})


def evaluate(directory):
    directory, manifest = load_run(directory)
    destination = directory / "results.json"
    if destination.exists():
        raise FileExistsError(destination)
    cases = json.loads((directory / "cases.json").read_text())
    aliases = json.loads((directory / "aliases.json").read_text())
    db = read_only(directory / "units.sqlite")
    results, configurations = {}, {}
    for kind in KINDS:
        rows = db.execute(f"SELECT id,document_id FROM {kind}_units ORDER BY id").fetchall()
        docs = np.array([r["document_id"] for r in rows])
        for model in ("bm25", "bge", "qwen"):
            name = f"{model}-{kind}"
            if model != "bm25":
                arm = directory / name
                cfg = json.loads((arm / "manifest.json").read_text())
                if not cfg["complete"] or cfg["identity"]["units_sha256"] != manifest["units_sha256"]:
                    raise ValueError("embedding not ready for evaluation")
                if file_sha256(arm / "vectors.npy") != cfg["vectors_sha256"]:
                    raise ValueError("embedding bytes changed")
                query_cfg = json.loads((directory / f"queries-{model}.json").read_text())
                if query_cfg["identity"]["encoder"] != cfg["identity"]["encoder"] or query_cfg["identity"]["cases_sha256"] != manifest["cases_sha256"]:
                    raise ValueError("query/encoder identity mismatch")
                if file_sha256(directory / f"queries-{model}.npy") != query_cfg["sha256"]:
                    raise ValueError("query embeddings changed")
                vectors = np.load(arm / "vectors.npy", mmap_mode="r")
                queries = np.load(directory / f"queries-{model}.npy")
                configurations[name] = cfg
            records = []
            for i, case in enumerate(cases):
                begin = time.monotonic()
                if model == "bm25":
                    query = lexical_query(case["question"])
                    ranked = list(db.execute(f"SELECT rowid,bm25({kind}_fts) AS score FROM {kind}_fts WHERE {kind}_fts MATCH ? ORDER BY score,rowid LIMIT 100", (query,))) if query else []
                    indices = np.array([r["rowid"] - 1 for r in ranked], dtype=np.int64)
                    values = [-r["score"] for r in ranked]
                else:
                    scores = vectors @ queries[i]
                    indices = top_indices(scores, 100); values = [float(scores[p]) for p in indices]
                ranked_docs = distinct_documents(docs[indices], aliases)[:20]
                records.append({"id": case["id"], "category": case["category"],
                                "gold_ids": case["gold_ids"], "ranked_document_ids": ranked_docs,
                                "top_unit_ids": [int(p) + 1 for p in indices], "top_unit_scores": values,
                                "metrics": recall_metrics(ranked_docs, case["gold_ids"]),
                                "search_seconds": time.monotonic() - begin})
            results[name] = records
            log(stage="evaluated", arm=name, all=aggregate(records)["all"], semantic=aggregate(records)["semantic"])
    db.close()
    summary = {"kind": manifest["kind"], "complete": True, "questions": len(cases),
               "documents": manifest["documents"], "unit_stats": manifest["unit_stats"],
               "hosted_model_calls": 0, "arms": {name: aggregate(records) for name, records in results.items()},
               "paired": {}, "complementarity": {}, "configurations": configurations,
               "limitations": ["Known dev100 with gold-containing candidate corpus, not full Enterprise retrieval.",
                               "Native windows may right-truncate inputs; counts are recorded per model/granularity.",
                               "No reranking, generation, answer judging, ANN or full-corpus index build.",
                               "Gold document recall does not measure whether the answer-bearing sentence was found."]}
    comparisons = [(f"qwen-{kind}", f"bge-{kind}") for kind in KINDS] + [(f"{model}-sentence", f"{model}-chunk") for model in ("bge", "qwen")]
    for candidate, baseline in comparisons:
        summary["paired"][f"{candidate}_minus_{baseline}"] = {}
        for category in ("all", "semantic"):
            pairs = [(a, b) for a, b in zip(results[candidate], results[baseline]) if a["metrics"] is not None and (category == "all" or a["category"] == category)]
            assert all(a["id"] == b["id"] for a, b in pairs)
            changes = [a["metrics"]["recall@20"] - b["metrics"]["recall@20"] for a, b in pairs]
            summary["paired"][f"{candidate}_minus_{baseline}"][category] = {"n": len(changes), "mean_delta_recall@20": statistics.mean(changes),
                "wins": sum(d > 0 for d in changes), "losses": sum(d < 0 for d in changes), "ties": sum(d == 0 for d in changes),
                "changes": [{"id": a["id"], "delta": d} for (a, _), d in zip(pairs, changes)]}
    for kind in KINDS:
        for model in ("bge", "qwen"):
            for category in ("all", "semantic"):
                rows = []
                for dense, lexical in zip(results[f"{model}-{kind}"], results[f"bm25-{kind}"]):
                    if not dense["gold_ids"] or (category != "all" and dense["category"] != category):
                        continue
                    gold, d, k = set(dense["gold_ids"]), set(dense["ranked_document_ids"]), set(lexical["ranked_document_ids"])
                    rows.append({"id": dense["id"], "new_gold": len((d - k) & gold), "lost_gold": len((k - d) & gold),
                                 "union_recall": len((d | k) & gold) / len(gold)})
                summary["complementarity"][f"{model}-{kind}/{category}"] = {"n": len(rows), "new_gold_occurrences": sum(r["new_gold"] for r in rows),
                    "lost_gold_occurrences": sum(r["lost_gold"] for r in rows), "mean_union_recall@20_each": statistics.mean(r["union_recall"] for r in rows), "cases": rows}
    save(destination, results); save(directory / "summary.json", summary)
    log(stage="complete", directory=str(directory), hosted_model_calls=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "probe", "encode", "evaluate"))
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", choices=("bge", "qwen"))
    p.add_argument("--granularity", choices=(*KINDS, "both"), default="both")
    p.add_argument("--device", choices=("mps", "cpu"), default="mps")
    p.add_argument("--max-seconds", type=float, default=21600)
    args = p.parse_args()
    if args.max_seconds <= 0:
        p.error("max-seconds must be positive")
    if args.command in ("probe", "encode") and not args.model:
        p.error("--model is required")
    if args.command == "prepare":
        prepare(args.root, args.output, args.device)
    elif args.command == "probe":
        probe(args.root, args.output, args.model, args.device)
    elif args.command == "encode":
        encode_run(args.root, args.output, args.model, KINDS if args.granularity == "both" else (args.granularity,), args.device, args.max_seconds)
    else:
        evaluate(args.output)


if __name__ == "__main__":
    main()
