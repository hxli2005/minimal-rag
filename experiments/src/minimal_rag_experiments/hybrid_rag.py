"""BM25 + exact dense retrieval + RRF + cross-encoder, behind the same search tool.

Models and indexes are local and pinned. No query planning model, generated
summaries, benchmark labels, or framework scheduler are involved in search.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
import threading
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

from minimal_rag.loop import Tool
from minimal_rag.indexed_rag import IndexedKnowledgeBase, IndexedSearchArgs, indexed_tools

from minimal_rag.paths import ROOT
DEFAULT_MODELS = ROOT / "models/hybrid-rag/models-minilm.json"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_only(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, check_same_thread=False)
    db.row_factory = sqlite3.Row
    return db


def load_model_config(manifest):
    """Resolve asset paths relative to their manifest so local caches can move."""
    manifest = Path(manifest).resolve()
    config = json.loads(manifest.read_text())
    for role in ("embedding", "reranker", "prefilter"):
        if role in config:
            path = Path(config[role]["path"]).expanduser()
            config[role]["path"] = str(path if path.is_absolute() else (manifest.parent / path).resolve())
    return config


class LocalModels:
    """One shared GPU model pair per process; inference is serialized, not Agent decisions."""
    def __init__(self, manifest=DEFAULT_MODELS, device="auto"):
        import torch
        from sentence_transformers import CrossEncoder, SentenceTransformer
        self.config = load_model_config(manifest)
        self.device = ("mps" if torch.backends.mps.is_available() else "cpu") if device == "auto" else device
        self.lock = threading.Lock()
        self.embedding = SentenceTransformer(self.config["embedding"]["path"], device=self.config.get("embedding_device", self.device),
                                             trust_remote_code=False, local_files_only=True)
        if self.config.get("max_length") is not None:
            self.embedding.max_seq_length = self.config["max_length"]
        self.reranker = CrossEncoder(self.config["reranker"]["path"], device=self.device,
                                    max_length=self.config.get("rerank_max_length", self.config.get("max_length") or 512),
                                    activation_fn=torch.nn.Identity(),
                                    trust_remote_code=False, local_files_only=True)
        if self.device == "mps" and self.config.get("precision", "float32") == "float16":
            self.embedding.half()
            self.reranker.model.half()
        self.dimension = self.embedding.get_embedding_dimension()
        self.identity = {k: self.config[k] for k in ("embedding", "query_prompt", "max_length", "precision")}
        self.identity["embedding"] = {k: v for k, v in self.identity["embedding"].items() if k != "path"}

    def encode(self, texts, *, query=False, batch_size=64):
        prompt = self.config["query_prompt"] if query else ""
        with self.lock:
            vectors = self.embedding.encode(texts, prompt=prompt, batch_size=batch_size,
                                            normalize_embeddings=True, show_progress_bar=False)
        vectors = np.asarray(vectors, dtype=np.float32)
        if not np.isfinite(vectors).all():
            raise ValueError("non-finite embedding")
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

    def rank(self, query, texts):
        with self.lock:
            scores = self.reranker.predict([(query, text) for text in texts], batch_size=16,
                                           show_progress_bar=False, convert_to_numpy=True)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if len(scores) != len(texts) or not np.isfinite(scores).all():
            raise ValueError("invalid reranker scores")
        return scores


def rrf(rankings, k=60):
    """Fuse ranks rather than incomparable BM25 and cosine score scales."""
    scores = Counter()
    for ranking in rankings:
        for rank, key in enumerate(dict.fromkeys(ranking), 1):
            scores[key] += 1.0 / (k + rank)
    return sorted(scores, key=lambda key: (-scores[key], key)), scores


class DenseIndex:
    def __init__(self, directory, corpus, models, *, verify=True):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest["embedding"] != models.identity:
            raise ValueError("embedding model differs from the dense index")
        if verify and file_sha256(corpus) != self.manifest["corpus_sha256"]:
            raise ValueError("dense index belongs to a different corpus")
        self.vectors = np.load(self.directory / "vectors.npy", mmap_mode="r")
        self.meta = read_only(self.directory / "metadata.sqlite")
        self.meta_lock = threading.Lock()
        if list(self.vectors.shape) != [self.manifest["chunks"], models.dimension]:
            raise ValueError("dense index shape mismatch")

    @classmethod
    def in_memory(cls, corpus, models):
        """Small per-question supplied corpora, e.g. HotpotQA distractor passages."""
        source = read_only(corpus)
        rows = source.execute("SELECT rowid AS chunk_id,document_id,title,text FROM chunks ORDER BY rowid").fetchall()
        obj = cls.__new__(cls)
        obj.vectors = models.encode([r["title"] + "\n" + r["text"] for r in rows])
        obj.meta = sqlite3.connect(":memory:", check_same_thread=False)
        obj.meta_lock = threading.Lock()
        obj.meta.row_factory = sqlite3.Row
        obj.meta.execute("CREATE TABLE documents(id TEXT PRIMARY KEY,title TEXT,source TEXT)")
        obj.meta.execute("CREATE TABLE passages(position INTEGER PRIMARY KEY,chunk_id INTEGER,document_id TEXT)")
        obj.meta.executemany("INSERT INTO documents VALUES (?,?,?)", source.execute("SELECT id,title,source FROM documents"))
        obj.meta.executemany("INSERT INTO passages VALUES (?,?,?)", [(i, r["chunk_id"], r["document_id"]) for i, r in enumerate(rows)])
        obj.meta.commit()
        obj.manifest = {"chunks": len(rows), "embedding": models.identity, "in_memory": True}
        source.close()
        return obj

    def scope_positions(self, scope):
        if not scope:
            return None
        # The index is shared by question workers. Protect execution AND cursor
        # consumption; check_same_thread=False alone does not protect statement reuse.
        with self.meta_lock:
            exact = self.meta.execute("SELECT 1 FROM documents WHERE id=?", (scope,)).fetchone()
            if exact:
                rows = self.meta.execute("SELECT position FROM passages WHERE document_id=? ORDER BY position", (scope,))
            else:
                pattern = "%" + scope.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
                rows = self.meta.execute("SELECT p.position FROM passages p JOIN documents d ON d.id=p.document_id "
                    "WHERE d.id LIKE ? ESCAPE '\\' OR d.title LIKE ? ESCAPE '\\' OR d.source LIKE ? ESCAPE '\\' ORDER BY p.position",
                    [pattern] * 3)
            try:
                return np.fromiter((r[0] for r in rows), dtype=np.int64)
            finally:
                rows.close()

    def search(self, vector, count, scope=""):
        positions = self.scope_positions(scope)
        if positions is None:
            # Exact dot products avoid ANN recall errors. This is a memory-mapped matrix, not a copy.
            scores = self.vectors @ vector
        else:
            scores = np.empty(len(positions), dtype=np.float32)
            for begin in range(0, len(positions), 8192):
                part = positions[begin:begin + 8192]
                scores[begin:begin + len(part)] = self.vectors[part] @ vector
        if not len(scores):
            return []
        count = min(count, len(scores))
        best = np.argpartition(-scores, count - 1)[:count]
        best = sorted(best, key=lambda i: (-float(scores[i]), int(i)))
        chosen = best if positions is None else [int(positions[i]) for i in best]
        placeholders = ",".join("?" for _ in chosen)
        with self.meta_lock:
            cursor = self.meta.execute(
                f"SELECT position,chunk_id FROM passages WHERE position IN ({placeholders})", [int(i) for i in chosen])
            try:
                mapping = {r["position"]: r["chunk_id"] for r in cursor}
            finally:
                cursor.close()
        return [mapping[int(i)] for i in chosen]

    def close(self):
        with self.meta_lock:
            self.meta.close()


class HybridKnowledgeBase(IndexedKnowledgeBase):
    def __init__(self, path, models, dense, excerpt_chars=4000, *, candidates=100, rerank=200, mode="hybrid", scope_index=None):
        super().__init__(path, excerpt_chars, check_same_thread=False, scope_index=scope_index)
        if candidates < 1 or rerank < 0 or mode not in {"hybrid", "dense", "bm25_rerank"}:
            raise ValueError("invalid retrieval configuration")
        self.models, self.dense = models, dense
        self.candidates, self.rerank_count, self.mode = candidates, rerank, mode
        self.cache, self.search_lock = OrderedDict(), threading.Lock()
        self.last_diagnostics = {}

    def lexical(self, query, count, scope, all_terms):
        terms = list(dict.fromkeys(a or b for a, b in re.findall(r'"([^"\n]+)"|([\w]+)', query)))
        match = (" AND " if all_terms else " OR ").join('"' + t.replace('"', '""') + '"' for t in terms)
        if not match:
            return []
        params = [match]
        sql = "SELECT c.rowid AS chunk_id,c.document_id FROM chunks c WHERE chunks MATCH ? AND rank MATCH 'bm25(0,0,2,1)'"
        if scope:
            if self.db.execute("SELECT 1 FROM documents WHERE id=?", (scope,)).fetchone():
                sql += " AND c.document_id=?"
                params.append(scope)
                bounds = self.document_bounds(scope)
                if bounds is not None:
                    sql += " AND c.rowid>=? AND c.rowid<=?"
                    params += list(bounds)
            else:
                pattern = "%" + scope.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
                sql += (" AND c.document_id IN (SELECT id FROM documents WHERE id LIKE ? ESCAPE '\\' "
                        "OR title LIKE ? ESCAPE '\\' OR source LIKE ? ESCAPE '\\')")
                params += [pattern] * 3
        sql += " ORDER BY rank"
        # Limit repeated chunks from one document without truncating the corpus before retrieval.
        counts, result = Counter(), []
        cursor = self.db.execute(sql, params)
        try:
            for row in cursor:
                cap = count if scope == row["document_id"] else 2
                if counts[row["document_id"]] >= cap:
                    continue
                counts[row["document_id"]] += 1
                result.append(row["chunk_id"])
                if len(result) == count:
                    break
        finally:
            cursor.close()
        return result

    def passages(self, ids):
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        return {r["chunk_id"]: dict(r) for r in self.db.execute(
            "SELECT c.rowid AS chunk_id,c.document_id,c.start,c.text,d.title,d.source,length(d.text) AS document_chars "
            "FROM chunks c JOIN documents d ON d.id=c.document_id "
            f"WHERE c.rowid IN ({placeholders})", ids)}

    def rank_candidates(self, query, scope, all_terms):
        lexical = self.lexical(query, self.candidates, scope, all_terms) if self.mode != "dense" or all_terms else []
        semantic = []
        if not all_terms and self.mode != "bm25_rerank":
            vector = self.models.encode([query], query=True)[0]
            raw = self.dense.search(vector, self.candidates * 5, scope)
            metadata, counts = self.passages(raw), Counter()
            for chunk in raw:
                doc = metadata[chunk]["document_id"]
                if counts[doc] >= (self.candidates if scope == doc else 2):
                    continue
                semantic.append(chunk)
                counts[doc] += 1
                if len(semantic) == self.candidates:
                    break
        fused, fusion_scores = rrf([lexical, semantic])
        selected = fused[:self.rerank_count] if self.rerank_count else fused
        passages = self.passages(selected)
        if self.rerank_count and selected:
            scores = self.models.rank(query, [passages[k]["title"] + "\n" + passages[k]["text"] for k in selected])
            ordered = sorted(zip(selected, scores), key=lambda item: (-float(item[1]), -fusion_scores[item[0]], item[0]))
        else:
            ordered = [(k, fusion_scores[k]) for k in selected]
        self.last_diagnostics = {"lexical": lexical, "dense": semantic, "fused": fused,
                                 "reranked": [k for k, _ in ordered]}
        return [(passages[k], float(score)) for k, score in ordered]

    def search(self, query, limit=5, offset=0, scope="", all_terms=False):
        if not query.strip():
            exact = self.db.execute("SELECT * FROM documents WHERE id=?", (scope,)).fetchone() if scope else None
            if exact:
                text = exact["text"][:self.excerpt_chars]
                hit = {"document_id": exact["id"], "title": exact["title"], "offset": 0,
                       "text": text, "document_chars": len(exact["text"]), "source": exact["source"],
                       "truncated": len(text) < len(exact["text"]), "score": 0.0}
                return {"hits": [] if offset else [hit], "next_offset": None}
            return super().search(query, limit, offset, scope, all_terms)
        key = (query, scope, all_terms)
        with self.search_lock:
            if key not in self.cache:
                self.cache[key] = self.rank_candidates(query, scope, all_terms)
                if len(self.cache) > 32:
                    self.cache.popitem(last=False)
            ranked = self.cache[key]
        hits, seen = [], set()
        exact = bool(scope and self.db.execute("SELECT 1 FROM documents WHERE id=?", (scope,)).fetchone())
        for row, score in ranked:
            before = max(0, self.excerpt_chars - len(row["text"])) // 2
            start = max(0, min(int(row["start"]) - before, row["document_chars"] - self.excerpt_chars))
            identity = (row["document_id"], start) if exact else row["document_id"]
            if identity in seen:
                continue
            seen.add(identity)
            if len(seen) <= offset:
                continue
            original = self.db.execute("SELECT text FROM documents WHERE id=?", (row["document_id"],)).fetchone()[0]
            text = original[start:start + self.excerpt_chars]
            hits.append({"document_id": row["document_id"], "title": row["title"], "offset": start,
                         "text": text, "document_chars": len(original), "source": row["source"],
                         "truncated": start > 0 or start + len(text) < len(original), "score": score})
            if len(hits) > limit:
                break
        return {"hits": hits[:limit], "next_offset": offset + limit if len(hits) > limit else None}

    async def asearch(self, **arguments):
        pending = asyncio.create_task(asyncio.to_thread(self.search, **arguments))
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError:
            # A Python worker cannot be killed safely; finish before its SQLite connection is closed.
            await asyncio.gather(pending, return_exceptions=True)
            raise


def hybrid_tools(kb):
    tools = indexed_tools(kb)
    retrieval = {
        "hybrid": "Search by meaning and keywords, then rerank relevance. Natural-language questions work; exact names and IDs also help. ",
        "dense": "Search by semantic similarity, then optionally rerank relevance. Use natural-language questions. ",
        "bm25_rerank": "Search by BM25 keywords, then rerank relevance. Use informative keywords, names and IDs. ",
    }[kb.mode]
    tools["search"] = Tool(
        retrieval +
        "scope restricts both retrieval paths to an ID, title or source substring; an exact document ID searches only that document. "
        "all_terms=true switches to strict lexical matching of every keyword/quoted phrase. "
        "Results contain original context, offsets and document lengths; use read for more. "
        "Global results contain distinct documents. limit/offset page through a bounded ranked pool; refine the query to explore further. "
        "An empty query lists documents.", IndexedSearchArgs, kb.asearch)
    return tools
