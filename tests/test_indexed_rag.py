"""Evidence offsets, complete corpus ingestion, scope isolation and pagination."""
from pathlib import Path

import pytest

from minimal_rag.indexed_rag import IndexedKnowledgeBase


@pytest.fixture
def kb(tmp_path):
    docs = [
        {"id": "a", "title": "Alpha report 2023", "text": "Preface. " * 300 + "\nNet revenue was $35 million.\n" + "Footnote. " * 100, "source": "public://a"},
        {"id": "b", "title": "Alpha report 2024", "text": "Net revenue was $17 million.", "source": "public://b"},
        {"id": "c", "title": "Product XY-02", "text": "Apples are delivered on Tuesday.", "source": "public://c"},
    ]
    path = tmp_path / "corpus.sqlite"
    IndexedKnowledgeBase.build(path, iter(docs))
    result = IndexedKnowledgeBase(path, excerpt_chars=1600)
    yield result
    result.close()


def test_search_can_find_evidence_far_from_document_start(kb):
    hit = kb.search('"Net revenue"', scope="2023")["hits"][0]
    assert hit["document_id"] == "a"
    assert "$35 million" in hit["text"]
    assert hit["offset"] > 0
    original = kb.read("a", limit=16000)["text"]
    assert original[hit["offset"]:hit["offset"] + len(hit["text"])] == hit["text"]
    assert hit["truncated"] is True
    assert hit["document_chars"] == len(original)


def test_read_supports_following_chunk_to_original_context(kb):
    hit = kb.search("revenue", scope="2023")["hits"][0]
    read = kb.read(hit["document_id"], offset=hit["offset"], limit=16000)
    assert read["text"].startswith(hit["text"])
    assert read["next_offset"] is None


def test_scopes_and_sql_metacharacters_are_literal(kb):
    assert {h["document_id"] for h in kb.search("revenue", scope="2024")["hits"]} == {"b"}
    assert kb.search("revenue", scope="%")["hits"] == []
    assert kb.search("revenue", scope="' OR 1=1 --")["hits"] == []


def test_unknown_and_empty_query_and_pageable_directory(kb):
    assert kb.search("zqwnotaword")["hits"] == []
    assert "error" in kb.read("unknown")
    one = kb.search("", limit=2)
    two = kb.search("", limit=2, offset=one["next_offset"])
    assert [h["document_id"] for h in one["hits"] + two["hits"]] == ["a", "b", "c"]


def test_stemming_and_punctuation_do_not_raise_match_syntax_errors(kb):
    assert kb.search("apple")["hits"][0]["document_id"] == "c"
    assert kb.search("XY-02? (Tuesday)")["hits"][0]["document_id"] == "c"


def test_every_original_character_is_covered_by_chunks(kb):
    for row in kb.db.execute("SELECT id,text FROM documents"):
        intervals = kb.db.execute("SELECT start,length(text) FROM chunks WHERE document_id=? ORDER BY CAST(start AS INTEGER)", (row["id"],)).fetchall()
        end = 0
        for start, length in intervals:
            assert start <= end
            end = max(end, start + length)
        assert end == len(row["text"])


def test_existing_index_is_never_overwritten(kb):
    with pytest.raises(FileExistsError):
        IndexedKnowledgeBase.build(kb.path, [])


def test_broad_search_diversifies_documents_but_exact_scope_returns_passages(tmp_path):
    path = tmp_path / "diverse.sqlite"
    IndexedKnowledgeBase.build(path, [
        {"id": "a", "title": "needle", "text": "needle " * 900, "source": "public://a"},
        {"id": "b", "title": "other", "text": "needle in another document", "source": "public://b"},
    ])
    kb = IndexedKnowledgeBase(path)
    try:
        assert {r["document_id"] for r in kb.search("needle", limit=2)["hits"]} == {"a", "b"}
        passages = kb.search("needle", scope="a", limit=2)["hits"]
        assert len(passages) == 2 and passages[0]["offset"] != passages[1]["offset"]
    finally:
        kb.close()


def test_model_can_require_all_terms(kb):
    assert kb.search("revenue Tuesday", all_terms=True)["hits"] == []
    assert kb.search("revenue Tuesday", all_terms=False)["hits"]


def test_failed_build_does_not_publish_a_partial_corpus(tmp_path):
    path = tmp_path / "broken.sqlite"
    duplicate = {"id": "a", "title": "a", "text": "example", "source": "public://a"}
    with pytest.raises(Exception):
        IndexedKnowledgeBase.build(path, [duplicate, duplicate])
    assert not path.exists()


def test_context_expansion_preserves_offsets_and_reports_entire_short_documents(kb):
    kb.excerpt_chars = 4000
    row = kb.search('"Net revenue"', scope="2023")["hits"][0]
    assert row["offset"] == 0
    assert row["text"] == kb.read("a", limit=16000)["text"]
    assert row["truncated"] is False
