"""Fast exact-document search keeps full-corpus ranks, source text and isolation."""
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from minimal_rag.indexed_rag import IndexedKnowledgeBase
from minimal_rag.scope_index import ScopeIndex


@pytest.fixture
def corpora(tmp_path):
    current, legacy = tmp_path / "current.sqlite", tmp_path / "legacy.sqlite"
    IndexedKnowledgeBase.build(current, [
        {"id": "a", "title": "Rollout", "text": ("rollback readiness threshold needle " * 100) + "\nOther conditions apply.", "source": "test://a"},
        {"id": "ab", "title": "Another tenant", "text": "needle rollback for another tenant", "source": "test://ab"},
        {"id": "wild%_", "title": "Literal scope", "text": "readiness needle", "source": "test://wild"},
        {"id": "empty", "title": "needle", "text": "", "source": "test://empty"},
    ], chunk_chars=200, overlap=30)
    shutil.copyfile(current, legacy)
    db = sqlite3.connect(legacy)
    db.execute("DROP TABLE document_ranges")
    db.commit()
    db.close()
    directory = tmp_path / "scope"
    directory.mkdir()
    db = sqlite3.connect(directory / "metadata.sqlite")
    db.execute("CREATE TABLE passages(chunk_id INTEGER PRIMARY KEY,document_id TEXT)")
    source = sqlite3.connect(legacy)
    db.executemany("INSERT INTO passages VALUES (?,?)", source.execute("SELECT rowid,document_id FROM chunks"))
    db.execute("CREATE INDEX passages_document ON passages(document_id)")
    db.commit()
    source.close()
    db.close()
    (directory / "manifest.json").write_text(json.dumps({"complete": True, "corpus_sha256": hashlib.sha256(legacy.read_bytes()).hexdigest()}))
    return current, legacy, directory


def test_embedded_and_external_ranges_preserve_full_ranked_results_and_pagination(corpora):
    current, legacy, directory = corpora
    readers = [IndexedKnowledgeBase(current, 400), IndexedKnowledgeBase(legacy, 400),
               IndexedKnowledgeBase(legacy, 400, scope_index=directory)]
    try:
        for query in ('rollback readiness threshold', '"Other conditions"', 'needle', ''):
            for scope in ("a", "wild%_", "empty", "unknown", "tenant", ""):
                for offset in (0, 1, 3):
                    results = [kb.search(query, scope=scope, limit=2, offset=offset) for kb in readers]
                    assert results[0] == results[1] == results[2]
        for kb in readers:
            assert {hit["document_id"] for hit in kb.search("needle", scope="a")["hits"]} == {"a"}
            assert {hit["document_id"] for hit in kb.search("", scope="a")["hits"]} == {"a"}
    finally:
        for kb in readers:
            kb.close()


def test_external_range_never_replaces_exact_source_predicate(corpora):
    _, legacy, directory = corpora
    # Even a broad range cannot leak a different document into an exact scope.
    meta = sqlite3.connect(directory / "metadata.sqlite")
    meta.execute("UPDATE passages SET document_id='a'")
    meta.commit()
    meta.close()
    kb = IndexedKnowledgeBase(legacy, scope_index=directory)
    try:
        assert {hit["document_id"] for hit in kb.search("needle", scope="a")["hits"]} == {"a"}
    finally:
        kb.close()


def test_mismatched_and_incomplete_scope_indexes_are_rejected(corpora):
    current, legacy, directory = corpora
    with pytest.raises(ValueError, match="different corpus"):
        ScopeIndex(directory, current)
    manifest = directory / "manifest.json"
    values = json.loads(manifest.read_text())
    manifest.write_text(json.dumps({**values, "complete": False}))
    with pytest.raises(ValueError, match="incomplete"):
        ScopeIndex(directory, legacy)


def test_digest_cache_invalidates_when_source_changes(corpora):
    _, legacy, directory = corpora
    ScopeIndex(directory, legacy).close()
    db = sqlite3.connect(legacy)
    db.execute("UPDATE documents SET title='Changed' WHERE id='a'")
    db.commit()
    db.close()
    with pytest.raises(ValueError, match="different corpus"):
        ScopeIndex(directory, legacy)


def test_shared_scope_metadata_cursor_access_is_serialized(corpora):
    _, legacy, directory = corpora
    index = ScopeIndex(directory, legacy)
    expected = index.bounds("a")
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: index.bounds("a"), range(200)))
        assert all(value == expected for value in results)
    finally:
        index.close()


def test_index_filename_is_literal_even_with_uri_metacharacters(tmp_path):
    path = tmp_path / "知识库 ?%# .sqlite"
    IndexedKnowledgeBase.build(path, [{"id": "correct", "title": "Expected corpus", "text": "needle", "source": "test://expected"}])
    kb = IndexedKnowledgeBase(path)
    try:
        assert kb.read("correct")["source"] == "test://expected"
        assert kb.search("needle", scope="correct")["hits"][0]["document_id"] == "correct"
    finally:
        kb.close()
