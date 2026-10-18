"""Optional cheap prefilter followed by precise local reranking; no Agent routing."""
import json
from pathlib import Path

import numpy as np

from minimal_rag_experiments.hybrid_rag import LocalModels


class CascadeModels(LocalModels):
    def __init__(self, manifest, device="auto"):
        super().__init__(manifest, device)
        import torch
        from sentence_transformers import CrossEncoder
        self.prefilter_count = self.config["prefilter_count"]
        if not isinstance(self.prefilter_count, int) or self.prefilter_count < 1:
            raise ValueError("prefilter_count must be positive")
        self.prefilter = CrossEncoder(self.config["prefilter"]["path"], device=self.device,
            max_length=self.config.get("prefilter_max_length", 512), activation_fn=torch.nn.Identity(),
            trust_remote_code=False, local_files_only=True)
        if self.device == "mps" and self.config.get("precision") == "float16":
            self.prefilter.model.half()

    @staticmethod
    def _scores(model, query, texts):
        scores = np.asarray(model.predict([(query, text) for text in texts], batch_size=16,
            show_progress_bar=False, convert_to_numpy=True), dtype=np.float32).reshape(-1)
        if len(scores) != len(texts) or not np.isfinite(scores).all():
            raise ValueError("invalid cascade reranker scores")
        return scores

    def rank(self, query, texts):
        if not texts:
            return np.empty(0, dtype=np.float32)
        with self.lock:
            if len(texts) <= self.prefilter_count:
                return self._scores(self.reranker, query, texts)
            coarse = self._scores(self.prefilter, query, texts)
            selected = sorted(range(len(texts)), key=lambda i: (-float(coarse[i]), i))[:self.prefilter_count]
            fine = self._scores(self.reranker, query, [texts[i] for i in selected])
        # Ranking keys, not probabilities: retain the unselected tail in coarse
        # order below all precisely scored candidates, so pagination still works.
        scores = float(np.min(fine)) - 2.0 + coarse / (1.0 + np.abs(coarse))
        scores[selected] = fine
        return scores


def load_models(manifest, device="auto"):
    config = json.loads(Path(manifest).read_text())
    return (CascadeModels if config.get("prefilter") else LocalModels)(manifest, device)
