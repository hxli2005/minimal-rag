"""Preserve some exact-keyword rank signal after a learned reranker.

A single global weight, chosen on development data; no entity lists, question
IDs, domain rules, or extra model calls. The Agent loop is unchanged.
"""
from math import isfinite

from minimal_rag_experiments.hybrid_rag import HybridKnowledgeBase


class FusionKnowledgeBase(HybridKnowledgeBase):
    def __init__(self, *args, lexical_prior=.5, **kwargs):
        super().__init__(*args, **kwargs)
        if not isfinite(lexical_prior) or lexical_prior < 0:
            raise ValueError("lexical prior must be nonnegative")
        self.lexical_prior = lexical_prior

    def rank_candidates(self, query, scope, all_terms):
        ranked = super().rank_candidates(query, scope, all_terms)
        if not ranked or not self.lexical_prior:
            return ranked
        # Exact-document searches still rank its passages by the cross-encoder.
        if scope and self.db.execute("SELECT 1 FROM documents WHERE id=?", (scope,)).fetchone():
            return ranked
        lexical_ids = self.last_diagnostics["lexical"]
        metadata = self.passages(lexical_ids)
        lexical_docs = list(dict.fromkeys(metadata[k]["document_id"] for k in lexical_ids))
        learned_docs = list(dict.fromkeys(row["document_id"] for row, _ in ranked))
        lexical_rank = {doc: i + 1 for i, doc in enumerate(lexical_docs)}
        scores = {doc: 1 / (61 + i) + (self.lexical_prior / (60 + lexical_rank[doc]) if doc in lexical_rank else 0)
                  for i, doc in enumerate(learned_docs)}
        # Keep the learned model's first result even if it has no lexical match.
        scores[learned_docs[0]] = max(scores.values()) + 1e-9
        # Stable sorting retains the best learned passage within each document.
        ordered = sorted(ranked, key=lambda item: -scores[item[0]["document_id"]])
        self.last_diagnostics["cross_encoder"] = self.last_diagnostics["reranked"]
        self.last_diagnostics["reranked"] = [row["chunk_id"] for row, _ in ordered]
        return [(row, scores[row["document_id"]]) for row, _ in ordered]
