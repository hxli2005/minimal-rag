"""Expose compact pointers to more ranked sources inside the existing search tool."""
from minimal_rag_experiments.hybrid_rag import HybridKnowledgeBase, hybrid_tools


class PreviewKnowledgeBase(HybridKnowledgeBase):
    def __init__(self, *args, preview_count=15, **kwargs):
        super().__init__(*args, **kwargs)
        if preview_count < 0:
            raise ValueError("preview_count must be nonnegative")
        self.preview_count = preview_count

    def search(self, query, limit=5, offset=0, scope="", all_terms=False):
        result = super().search(query, limit, offset, scope, all_terms)
        if not query.strip() or not self.preview_count:
            return result
        with self.search_lock:
            ranked = list(self.cache.get((query, scope, all_terms), []))
        seen = {r["document_id"] for r in result["hits"]}
        alternatives = []
        for row, _ in ranked:
            if row["document_id"] in seen:
                continue
            seen.add(row["document_id"])
            alternatives.append({"document_id": row["document_id"], "title": row["title"],
                                 "source": row["source"], "offset": int(row["start"])})
            if len(alternatives) == self.preview_count:
                break
        result["other_matches"] = alternatives
        return result


def preview_tools(kb):
    tools = hybrid_tools(kb)
    tools["search"].description += (
        " other_matches lists additional ranked source titles and passage offsets, without evidence text. "
        "Use read to verify promising sources, especially when the full hits do not match the precise question.")
    return tools
