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


def test_semantic_only_hit_is_reranked_before_final_truncation(collection):
    kb, _, models = collection
    result = kb.search("GPU rollout", limit=1)
    assert result["hits"][0]["document_id"] == "answer"
    assert len(models.ranked_texts) == 3  # Not just the lexical top 1.
    assert any("H100" in text for text in models.ranked_texts)


def test_scope_is_applied_before_vector_topk(collection):
    kb, dense, _ = collection
    # Globally the answer ranks first; restricting to "other" must still retrieve "other".
    assert dense.search(np.array([1., 0.], dtype=np.float32), 1, "other") == [3]
    assert [r["document_id"] for r in kb.search("GPU", scope="other")["hits"]] == ["other"]
    assert kb.search("GPU", scope="%")["hits"] == []


def test_strict_keywords_cannot_be_bypassed_by_dense_similarity(collection):
    kb, _, _ = collection
    assert kb.search("GPU", all_terms=True)["hits"][0]["document_id"] != "answer"
    assert kb.search("GPU Friday", all_terms=True)["hits"] == []


def test_pagination_is_stable_and_uses_original_text(collection):
    kb, _, _ = collection
    first = kb.search("GPU", limit=1)
    second = kb.search("GPU", limit=2, offset=first["next_offset"])
    ids = [r["document_id"] for r in first["hits"] + second["hits"]]
    assert len(set(ids)) == 3
    assert second["next_offset"] is None
    for hit in first["hits"] + second["hits"]:
        assert hit["text"] == kb.read(hit["document_id"], offset=hit["offset"])["text"]


def test_only_two_agent_tools_are_exposed(collection):
    assert set(hybrid_tools(collection[0])) == {"search", "read"}


def test_empty_query_respects_an_exact_document_scope(collection):
    kb = collection[0]
    assert [x["document_id"] for x in kb.search("", scope="answer")["hits"]] == ["answer"]
    assert kb.search("", scope="answer", offset=1)["hits"] == []






def test_index_model_and_corpus_mismatch_fail_closed(collection, tmp_path):
    _, dense, models = collection
    wrong = tmp_path / "different.sqlite"
    wrong.write_text("different corpus")
    with pytest.raises(ValueError, match="different corpus"):
        DenseIndex(dense.directory, wrong, models)


def test_rrf_uses_ranks_and_does_not_reward_duplicate_ids():
    order, scores = rrf([[1, 1, 2], [2, 3]])
    assert order[0] == 2
    assert scores[1] == pytest.approx(1 / 61)


@pytest.mark.parametrize("storage", ["disk", "memory"])
def test_shared_dense_index_matches_serial_results_under_concurrent_search(collection, storage):
    kb, existing, models = collection
    dense = existing if storage == "disk" else DenseIndex.in_memory(kb.path, models)
    queries = [(np.array(vector, dtype=np.float32), count, scope)
               for vector in ([1., 0.], [0., 1.], [-1., .3])
               for count, scope in ((3, ""), (2, ""), (1, "answer"), (2, "drive"), (1, "missing"))]
    expected = [dense.search(*query) for query in queries]

    def search_many(worker):
        for i in range(600):
            index = (i + worker) % len(queries)
            assert dense.search(*queries[index]) == expected[index]

    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(search_many, range(3)))
    finally:
        if storage == "memory":
            dense.close()


@pytest.mark.asyncio
async def test_cancellation_waits_for_read_only_worker_before_caller_can_close(collection, monkeypatch):
    kb = collection[0]
    entered, release = threading.Event(), threading.Event()
    def slow(**kwargs):
        entered.set()
        release.wait(2)
        return {"hits": [], "next_offset": None}
    monkeypatch.setattr(kb, "search", slow)
    task = asyncio.create_task(kb.asearch(query="GPU"))
    await asyncio.to_thread(entered.wait, 1)
    task.cancel()
    await asyncio.sleep(.01)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

from minimal_rag_experiments.hybrid_rag import load_model_config

def test_model_manifests_are_relocatable(tmp_path, monkeypatch):
    folder = tmp_path / "models"
    folder.mkdir()
    manifest = folder / "models.json"
    manifest.write_text(json.dumps({"embedding": {"path": "weights/embed"},
                                    "reranker": {"path": "weights/rank"},
                                    "prefilter": {"path": "weights/prefilter"}}))
    monkeypatch.chdir(tmp_path.parent)
    config = load_model_config(manifest)
    assert config["embedding"]["path"] == str(folder / "weights/embed")
    assert config["prefilter"]["path"] == str(folder / "weights/prefilter")
