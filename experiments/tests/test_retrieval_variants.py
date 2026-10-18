"""Rank ordering, full scope filtering and exact source offsets; no fake quality scores."""
import json
import asyncio
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from minimal_rag_experiments.hybrid_rag import DenseIndex, HybridKnowledgeBase, file_sha256, hybrid_tools, rrf
from minimal_rag.indexed_rag import IndexedKnowledgeBase
from minimal_rag_experiments.rerank_fusion import FusionKnowledgeBase
from minimal_rag_experiments.preview_rag import PreviewKnowledgeBase, preview_tools


class Models:
    dimension = 2
    identity = {"test_model": "fixed"}

    def __init__(self):
        self.ranked_texts = []

    def encode(self, texts, **kwargs):
        return np.asarray([[1., 0.] for _ in texts], dtype=np.float32)

    def rank(self, query, texts):
        self.ranked_texts = list(texts)
        return np.asarray([10. if "H100" in text else 0. for text in texts])


@pytest.fixture
def collection(tmp_path):
    path = tmp_path / "corpus.sqlite"
    docs = [
        {"id": "keyword", "title": "GPU chatter", "text": "GPU GPU GPU; no rollout schedule here.", "source": "slack://chatter"},
        {"id": "answer", "title": "Dedicated capacity", "text": "H100 machines undergo maintenance Friday at 02:00 UTC.", "source": "drive://infra"},
        {"id": "other", "title": "Other project", "text": "GPU tests are unrelated to this rollout.", "source": "drive://other-answer"},
    ]
    IndexedKnowledgeBase.build(path, docs)
    folder = tmp_path / "dense"
    folder.mkdir()
    models = Models()
    # The semantic candidate has no literal GPU token; it must still reach reranking.
    np.save(folder / "vectors.npy", np.asarray([[.3, .954], [1., 0.], [.1, .995]], dtype=np.float32))
    db = sqlite3.connect(folder / "metadata.sqlite")
    db.execute("CREATE TABLE passages(position INTEGER PRIMARY KEY,chunk_id INTEGER,document_id TEXT)")
    db.execute("CREATE TABLE documents(id TEXT PRIMARY KEY,title TEXT,source TEXT)")
    db.executemany("INSERT INTO passages VALUES (?,?,?)", [(i, i + 1, d["id"]) for i, d in enumerate(docs)])
    db.executemany("INSERT INTO documents VALUES (?,?,?)", [(d["id"], d["title"], d["source"]) for d in docs])
    db.commit(); db.close()
    (folder / "manifest.json").write_text(json.dumps({"embedding": models.identity, "corpus_sha256": file_sha256(path), "chunks": 3}))
    dense = DenseIndex(folder, path, models)
    kb = HybridKnowledgeBase(path, models, dense, candidates=3, rerank=3)
    yield kb, dense, models
    kb.close(); dense.close()


def test_keyword_prior_does_not_remove_the_best_semantic_match(collection):
    base, dense, models = collection
    kb = FusionKnowledgeBase(base.path, models, dense, candidates=3, rerank=3, lexical_prior=.5)
    try:
        assert kb.search("GPU rollout", limit=1)["hits"][0]["document_id"] == "answer"
        assert kb.search("GPU", scope="other")["hits"][0]["document_id"] == "other"
    finally:
        kb.close()

def test_additional_source_pointers_respect_scope_and_do_not_repeat_full_hits(collection):
    base, dense, models = collection
    kb = PreviewKnowledgeBase(base.path, models, dense, candidates=3, rerank=3, preview_count=2)
    try:
        result = kb.search("GPU", limit=1)
        assert result["hits"] == base.search("GPU", limit=1)["hits"]
        assert len(result["other_matches"]) == 2
        assert not {r["document_id"] for r in result["hits"]} & {r["document_id"] for r in result["other_matches"]}
        assert all("text" not in r for r in result["other_matches"])
        assert kb.search("GPU", scope="other")["other_matches"] == []
        assert kb.search("GPU", scope="%", limit=1)["other_matches"] == []
        assert set(preview_tools(kb)) == {"search", "read"}
    finally:
        kb.close()
