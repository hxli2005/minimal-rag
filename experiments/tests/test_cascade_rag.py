import threading
import numpy as np
import pytest
from minimal_rag_experiments.cascade_rag import CascadeModels


class Scores:
    def __init__(self, values):
        self.values, self.seen = values, []

    def predict(self, pairs, **kwargs):
        self.seen.append(pairs)
        return [self.values[text] for _, text in pairs]


def fixture():
    models = CascadeModels.__new__(CascadeModels)
    models.lock, models.prefilter_count = threading.Lock(), 2
    models.prefilter = Scores({"A": 10, "B": 9, "C": 1, "D": 0})
    models.reranker = Scores({"A": -2, "B": 3, "C": -1, "D": 7})
    return models


def test_cascade_reorders_prefiltered_candidates_and_preserves_the_tail():
    models = fixture()
    scores = models.rank("question", ["A", "B", "C", "D"])
    assert np.isfinite(scores).all()
    assert np.argsort(-scores).tolist() == [1, 0, 2, 3]
    assert models.reranker.seen == [[("question", "A"), ("question", "B")]]


def test_small_pool_goes_directly_to_precise_reranker():
    models = fixture()
    assert models.rank("question", ["C", "D"]).tolist() == [-1, 7]
    assert not models.prefilter.seen
    assert models.rank("question", []).size == 0


def test_cascade_rejects_nonfinite_model_output():
    models = fixture()
    models.prefilter.values["C"] = float("nan")
    with pytest.raises(ValueError, match="invalid cascade"):
        models.rank("question", ["A", "B", "C"])
