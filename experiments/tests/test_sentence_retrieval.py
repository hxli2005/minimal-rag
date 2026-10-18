import json
import sqlite3

import numpy as np
import pytest

from minimal_rag_experiments import sentence_retrieval as study


def test_exact_top_k_including_cutoff_ties():
    scores = np.array([.7, .9, .7, .7, -.1], dtype=np.float32)
    assert study.top_indices(scores, 3).tolist() == [1, 0, 2]
    assert study.top_indices(scores, 10).tolist() == [1, 0, 2, 3, 4]
    assert study.top_indices(np.array([]), 5).tolist() == []
    with pytest.raises(ValueError):
        study.top_indices(np.array([np.nan]), 1)


def test_document_aliases_and_no_gold_are_not_false_zero_scores():
    docs = study.distinct_documents(["a", "a-copy", "b"], {"a-copy": "a"})
    assert docs == ["a", "b"]
    assert study.recall_metrics(docs, ["a", "c"])["recall@20"] == .5
    assert study.recall_metrics(docs, []) is None
    rows = [{"category": "semantic", "metrics": None},
            {"category": "semantic", "metrics": study.recall_metrics(docs, ["a"])}]
    assert study.aggregate(rows)["semantic"] == {"questions": 2, "scored": 1, "no_gold": 1,
        "recall@5": 1., "recall@10": 1., "recall@20": 1., "all_gold@20": 1.}


def make_run(tmp_path):
    db = sqlite3.connect(tmp_path / "units.sqlite")
    for kind in study.KINDS:
        db.execute(f"CREATE TABLE {kind}_units(id INTEGER PRIMARY KEY,document_id TEXT,text TEXT)")
        db.executemany(f"INSERT INTO {kind}_units VALUES (?,?,?)", [(1, "a", "alpha retry"), (2, "b", "beta timeout"), (3, "c", "gamma data")])
        db.execute(f"CREATE VIRTUAL TABLE {kind}_fts USING fts5(text, tokenize='unicode61')")
        db.executemany(f"INSERT INTO {kind}_fts(rowid,text) VALUES (?,?)", [(1,"alpha retry"),(2,"beta timeout"),(3,"gamma data")])
    db.commit(); db.close()
    cases = [{"id": "q1", "category": "semantic", "question": "alpha", "gold_ids": ["a"]},
             {"id": "q2", "category": "basic", "question": "beta", "gold_ids": ["b"]},
             {"id": "q3", "category": "info_not_found", "question": "unknown", "gold_ids": []}]
    for name, data in (("cases",cases),("aliases",{}),("documents",["a","b","c"])):
        study.save(tmp_path / f"{name}.json", data)
    manifest = {"complete": True, "kind": "gold_containing_candidate_corpus_diagnostic", "documents": 3, "unit_stats": {},
                "units_sha256": study.file_sha256(tmp_path / "units.sqlite"),
                **{f"{name}_sha256": study.file_sha256(tmp_path / f"{name}.json") for name in ("cases", "aliases", "documents")}}
    study.save(tmp_path / "manifest.json", manifest)
    for model in ("bge", "qwen"):
        encoder = {"model": model}
        path = tmp_path / f"queries-{model}.npy"
        np.save(path, np.eye(3, dtype=np.float32))
        study.save(tmp_path / f"queries-{model}.json", {"identity": {"encoder": encoder, "cases_sha256": manifest["cases_sha256"]}, "sha256": study.file_sha256(path)})
        for kind in study.KINDS:
            arm = tmp_path / f"{model}-{kind}"; arm.mkdir()
            np.save(arm / "vectors.npy", np.eye(3, dtype=np.float32))
            study.save(arm / "manifest.json", {"complete": True, "identity": {"encoder": encoder, "units_sha256": manifest["units_sha256"]},
                "vectors_sha256": study.file_sha256(arm / "vectors.npy")})
    return manifest


def test_end_to_end_scoring_without_loading_models(tmp_path):
    make_run(tmp_path)
    study.evaluate(tmp_path)
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["hosted_model_calls"] == 0
    assert len(result["arms"]) == 6
    assert result["arms"]["qwen-sentence"]["all"]["scored"] == 2
    assert result["arms"]["qwen-sentence"]["all"]["no_gold"] == 1
    assert result["paired"]["qwen-sentence_minus_bge-sentence"]["semantic"]["ties"] == 1
    with pytest.raises(FileExistsError):
        study.evaluate(tmp_path)


@pytest.mark.parametrize("target", ["aliases.json", "cases.json", "documents.json"])
def test_changed_scoring_inputs_are_rejected(tmp_path, target):
    make_run(tmp_path)
    (tmp_path / target).write_text("[]")
    with pytest.raises(ValueError):
        study.load_run(tmp_path)


def test_query_corruption_is_rejected_before_scoring(tmp_path):
    make_run(tmp_path)
    np.save(tmp_path / "queries-bge.npy", np.zeros((3, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="query embeddings changed"):
        study.evaluate(tmp_path)
    assert not (tmp_path / "summary.json").exists()


def test_interrupted_encoding_resumes_only_with_same_identity(tmp_path, monkeypatch):
    make_run(tmp_path)
    # Only remove fixtures created by this test, never original experiment assets.
    import shutil
    shutil.rmtree(tmp_path / "bge-chunk")
    (tmp_path / "queries-bge.npy").unlink()
    (tmp_path / "queries-bge.json").unlink()
    study.save(tmp_path / "probe-bge.json", {"test_fixture": True})
    original_encoder = study.Encoder

    class FakeEncoder:
        fail = True
        version = 1
        def __init__(self, *_):
            self.identity = {"version": self.version}
            self.dimension, self.max_length = 2, 128
            self.prompt, self.load_seconds = "", 0.
            self.batch_size, self.padded_tokens, self.calls = 1, 128, 0
        def lengths(self, texts):
            return np.array([len(t) for t in texts], dtype=np.int32)
        batches = original_encoder.batches
        def encode(self, texts):
            self.calls += 1
            if self.fail and self.calls == 2:
                raise RuntimeError("simulated interruption")
            vectors = np.array([[len(t), 1.] for t in texts], dtype=np.float32)
            return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        def encode_all(self, texts):
            return self.encode(texts), self.lengths(texts)

    monkeypatch.setattr(study, "Encoder", FakeEncoder)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        study.encode_run(tmp_path, tmp_path, "bge", ("chunk",), "cpu", 60)
    progress = json.loads((tmp_path / "bge-chunk.building/progress.json").read_text())
    assert progress["completed"] == 1
    assert not (tmp_path / "bge-chunk").exists()
    FakeEncoder.version = 2
    with pytest.raises(ValueError, match="cannot resume"):
        study.encode_run(tmp_path, tmp_path, "bge", ("chunk",), "cpu", 60)
    FakeEncoder.version, FakeEncoder.fail = 1, False
    study.encode_run(tmp_path, tmp_path, "bge", ("chunk",), "cpu", 60)
    manifest = json.loads((tmp_path / "bge-chunk/manifest.json").read_text())
    assert manifest["complete"] and manifest["encoding"]["completed"] == 3
    vectors = np.load(tmp_path / "bge-chunk/vectors.npy")
    expected = FakeEncoder().encode(["alpha retry", "beta timeout", "gamma data"])
    assert np.allclose(vectors, expected)
    assert not (tmp_path / "bge-chunk.building").exists()
