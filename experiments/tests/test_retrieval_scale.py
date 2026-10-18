import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from minimal_rag.indexed_rag import IndexedKnowledgeBase
from minimal_rag_experiments import retrieval_scale as scale

from minimal_rag_experiments.retrieval_scale import fuse_documents, promotion, sample_ids


def test_sampling_is_order_independent_and_reference_additions_are_explicit():
    ids = [f"d{i}" for i in range(100)]
    random, additions, count = sample_ids(ids, .1, "fixed")
    assert len(random) == count == 10 and additions == []
    extra = next(doc for doc in ids if doc not in random)
    sampled, additions, count = sample_ids(reversed(ids), .1, "fixed", [extra])
    assert sampled == sorted(random + [extra])
    assert additions == [extra] and count == 10
    with pytest.raises(ValueError, match="missing"):
        sample_ids(ids, .1, "fixed", ["absent"])


def records(values, categories=None):
    categories = categories or ["semantic"] * len(values)
    return [{"id": f"q{i}", "category": categories[i], "gold_ids": ["d"], "metrics": {"recall@20": value}}
            for i, value in enumerate(values)]


GATE = {"metric": "recall@20", "minimum_gain": .1,
        "candidate": "hybrid", "baseline": "bm25", "category": "semantic",
        "require_more_wins_than_losses": True, "overall_minimum_gain": 0.0}


def test_gate_uses_hybrid_and_ignores_qwen_and_bge():
    arms = {"hybrid": records([1, 1, 1, 0]), "bge": records([0, 0, 0, 0]),
            "qwen": records([0, 0, 0, 0]), "bm25": records([1, 0, 1, 0])}
    assert promotion(arms, GATE)["passed"]
    arms["bm25"][0]["id"] = "different-question"
    with pytest.raises(ValueError, match="cohort"):
        promotion(arms, GATE)


def test_equal_wins_and_losses_cannot_trigger_full_build():
    arms = {"hybrid": records([1, 0]), "bm25": records([0, .2])}
    assert not promotion(arms, GATE)["passed"]


def test_overall_regression_blocks_full_build_even_when_semantic_passes():
    categories = ["semantic", "basic", "basic"]
    arms = {"hybrid": records([1, 0, 0], categories), "bm25": records([0, 1, 1], categories)}
    result = promotion(arms, GATE)
    assert result["comparisons"]["semantic"]["passed"]
    assert not result["comparisons"]["all"]["passed"]
    assert not result["passed"]


def test_semantic_threshold_and_overall_tie_pass():
    categories = ["semantic", "basic"]
    arms = {"hybrid": records([.1, 0], categories), "bm25": records([0, .1], categories)}
    result = promotion(arms, GATE)
    assert result["passed"] and result["comparisons"]["all"]["mean_gain"] == 0
    arms["hybrid"][0]["metrics"]["recall@20"] = .09
    assert not promotion(arms, GATE)["passed"]


def test_document_rrf_deduplicates_and_keeps_equal_output_budget():
    assert fuse_documents(["a", "a", "b"], ["b", "a"], k=60, limit=20) == ["a", "b"]
    lexical = [f"lex{i:03d}" for i in range(100)]
    dense = [f"dense{i:03d}" for i in range(100)]
    fused = fuse_documents(lexical, dense, k=60, limit=20)
    assert len(fused) == 20 and len(set(fused)) == 20
    assert len(set(fused) & set(lexical)) == 10
    assert len(set(fused) & set(dense)) == 10


class TinyEncoder:
    dimension, device, identity = 2, "test", {"revision": "fixed-model"}

    def encode(self, texts, **_):
        return np.array([[1., 0.]] * len(texts), dtype=np.float32)


def test_encoding_identity_ignores_core_edits_but_rejects_changed_inputs(tmp_path):
    corpus = tmp_path / "corpus.sqlite"
    IndexedKnowledgeBase.build(corpus, [{"id": "a", "title": "A", "text": "original", "source": "test://a"}])
    path = tmp_path / "encoding-inputs.json"
    chunking = {"chunk_chars": 1600, "overlap": 200}
    original = scale.freeze_encoding_inputs(path, corpus, TinyEncoder(), chunking)
    for name in ("loop.py", "indexed_rag.py", "hybrid_rag.py"):
        (tmp_path / name).write_text("changed core implementation")
    assert scale.freeze_encoding_inputs(path, corpus, TinyEncoder(), chunking) == original
    assert "sources" not in original
    changed = TinyEncoder()
    changed.identity = {"revision": "different-model"}
    with pytest.raises(ValueError, match="encoding inputs changed"):
        scale.freeze_encoding_inputs(path, corpus, changed, chunking)
    with pytest.raises(ValueError, match="encoding inputs changed"):
        scale.freeze_encoding_inputs(path, corpus, TinyEncoder(), {**chunking, "overlap": 100})
    saved = json.loads(path.read_text()); saved["input_format"] = "body only"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="encoding inputs changed"):
        scale.freeze_encoding_inputs(path, corpus, TinyEncoder(), chunking)
    path.write_text(json.dumps(original))
    # A corpus mutation is a real encoding input change, even if counts match.
    import sqlite3
    with sqlite3.connect(corpus) as db:
        db.execute("UPDATE documents SET text='changed' WHERE id='a'")
    with pytest.raises(ValueError, match="encoding inputs changed"):
        scale.freeze_encoding_inputs(path, corpus, TinyEncoder(), chunking)


def test_full_resume_skips_evaluation_and_uses_saved_promotion(tmp_path, monkeypatch):
    corpus = tmp_path / "data/benchmarks/enterprise/corpus.sqlite"
    IndexedKnowledgeBase.build(corpus, [{"id": "a", "title": "A", "text": "original", "source": "test://a"}])
    directory = tmp_path / "run"; directory.mkdir()
    chunking = {"chunk_chars": 1600, "overlap": 200}
    scale.freeze_encoding_inputs(directory / "full-encoding-inputs.json", corpus, TinyEncoder(), chunking)
    (directory / "full-build-plan.json").write_text(json.dumps({
        "corpus": "data/benchmarks/enterprise/corpus.sqlite", "output": "full.dense",
        "chunking": chunking, "gate": {"passed": True}}))
    config = {"experiment": "dense_scale", "dataset": "enterprise", "device": "test", "sample_output": "run"}
    config_path = tmp_path / "config.json"; config_path.write_text(json.dumps(config))
    # No dev set, evaluator snapshot or current gate exists: none is required.
    def forbidden(*_, **__):
        raise AssertionError("Full encoding must not read evaluation inputs or sources")
    monkeypatch.setattr(scale, "ROOT", tmp_path)
    monkeypatch.setattr(scale, "ChunkEncoder", lambda *_: TinyEncoder())
    monkeypatch.setattr(scale, "load_cases", forbidden)
    monkeypatch.setattr(scale, "capture_evaluation_sources", forbidden)
    monkeypatch.setattr(scale, "record_evaluation_start", forbidden)
    script = Path(__file__).resolve().parents[2] / "experiments/scripts/build_dense_rag_index.py"
    spec = importlib.util.spec_from_file_location("full_resume_builder", script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(scale, "builder", lambda *_: module)
    scale.run_config(config_path)
    assert json.loads((directory / "status.json").read_text())["stage"] == "complete"
    assert (tmp_path / "full.dense/manifest.json").exists()


def test_evaluation_hashes_are_recorded_only_when_evaluation_starts(tmp_path):
    directory = tmp_path / "run"; directory.mkdir()
    data = tmp_path / "data/benchmarks/enterprise"; data.mkdir(parents=True)
    (data / "dev.json").write_text("[]")
    (data / "duplicate-id-aliases.json").write_text("{}")
    config = {"dataset": "enterprise", "evaluation_category": "all", "fusion": {}, "gate": {}}
    assert not (directory / "evaluation-plan.json").exists()
    scale.record_evaluation_start(tmp_path, directory, config, {"loop.py": b"loaded code"})
    record = json.loads((directory / "evaluation-plan.json").read_text())
    assert "loop.py" in record["identity"]["sources"]
    assert (directory / "evaluation-source-snapshot/loop.py").read_bytes() == b"loaded code"
