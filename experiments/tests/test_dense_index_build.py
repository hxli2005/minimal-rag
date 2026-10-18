"""Long index builds must resume without publishing a partial or mismatched corpus."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from minimal_rag_experiments.hybrid_rag import DenseIndex
from minimal_rag.indexed_rag import IndexedKnowledgeBase

SCRIPT = Path(__file__).resolve().parents[2] / "experiments/scripts/build_dense_rag_index.py"
spec = importlib.util.spec_from_file_location("rag_dense_builder_test", SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class Encoder:
    dimension, device = 2, "test"
    identity = {"revision": "test-one"}

    def __init__(self, fail_second=False):
        self.calls, self.fail_second = 0, fail_second

    def encode(self, texts, **kwargs):
        self.calls += 1
        if self.fail_second and self.calls == 2:
            raise RuntimeError("simulated interruption")
        return np.array([[1., 0.]] * len(texts), dtype=np.float32)


def test_interrupted_build_resumes_and_binds_every_vector_to_its_source(tmp_path):
    corpus, target = tmp_path / "corpus.sqlite", tmp_path / "dense"
    IndexedKnowledgeBase.build(corpus, ({"id": f"d{i}", "title": f"title {i}", "text": "One short original passage.", "source": f"test://{i}"} for i in range(2050)))
    with pytest.raises(RuntimeError, match="simulated interruption"):
        builder.build(corpus, target, Encoder(fail_second=True))
    partial = tmp_path / "dense.building"
    assert not target.exists()
    assert not (partial / "manifest.json").exists()
    assert json.loads((partial / "progress.json").read_text())["completed"] == 2048
    changed = Encoder()
    changed.identity = {"revision": "test-two"}
    with pytest.raises(ValueError, match="configuration differs"):
        builder.build(corpus, target, changed, resume=True)
    model = Encoder()
    builder.build(corpus, target, model, resume=True)
    assert model.calls == 1  # Only the two remaining passages are encoded.
    index = DenseIndex(target, corpus, model)
    try:
        rows = index.meta.execute("SELECT position,chunk_id,document_id FROM passages ORDER BY position").fetchall()
        assert len(rows) == 2050
        assert tuple(rows[-1]) == (2049, 2050, "d2049")
        assert index.vectors.shape == (2050, 2)
        assert np.all(index.vectors[:, 0] == 1.)
        assert not partial.exists()
    finally:
        index.close()
    with pytest.raises(FileExistsError):
        builder.build(corpus, target, model)
