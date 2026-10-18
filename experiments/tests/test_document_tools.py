from minimal_rag_experiments.document_tools import IndexedKnowledgeBase, indexed_tools


def test_explicit_metadata_and_unknown_comment_dates(tmp_path):
    path = tmp_path / "corpus.sqlite"
    IndexedKnowledgeBase.build(path, [
        {"id": "a", "title": "A", "source": "a", "text": "Last updated: 2026-10-01\nVersion: v2\nStatus: superseded\nSuperseded by: b\n\n## Rules\nOld rule"},
        {"id": "b", "title": "B", "source": "b", "text": "Comments:\n2026-10-02 — Alice: changed runtime v9", "metadata": {"date": "2026-10-03", "version": "v3", "superseded": False}},
        {"id": "c", "title": "C", "source": "c", "text": "2026-10-04 — Bob: updated deployment v4"},
    ])
    kb = IndexedKnowledgeBase(path, document_tools=True)
    assert kb.read("a")["metadata"]["superseded"] is True
    assert kb.read("b")["metadata"]["date"] == "2026-10-03"
    assert kb.read("c")["metadata"] == {"date": None, "version": None, "superseded": None}
    assert kb.search("Rules")["hits"][0]["metadata"]["version"] == "v2"
    kb.close()


def test_toc_section_children_and_pagination(tmp_path):
    text = "## Procedure\nStart\n\nInterpretation:\n- pass\n\n### Save\nWrite JSON\n\n## Next\nDone\n"
    path = tmp_path / "corpus.sqlite"
    IndexedKnowledgeBase.build(path, [{"id": "a", "title": "A", "source": "a", "text": text}])
    kb = IndexedKnowledgeBase(path, document_tools=True)
    toc = kb.read("a", view="toc")["sections"]
    assert [s["title"] for s in toc] == ["Procedure", "Save", "Next"]
    result = kb.read("a", section=toc[0]["id"], limit=16000)
    assert "Write JSON" in result["text"] and "## Next" not in result["text"]
    first = kb.read("a", section=toc[0]["id"], limit=12)
    second = kb.read("a", section=toc[0]["id"], offset=first["next_offset"], limit=12)
    assert first["text"] + second["text"] == text[:24]
    assert kb.read("a", section="absent")["error"] == "section_not_found"
    kb.close()


def test_plain_headings_and_base_wire_unchanged(tmp_path):
    text = "Rollback criteria (hard stop)\nImmediate rollback if any occurs:\n- failure\n\nRollback actions\n1) disable\n"
    path = tmp_path / "corpus.sqlite"
    IndexedKnowledgeBase.build(path, [{"id": "a", "title": "A", "source": "a", "text": text}])
    kb = IndexedKnowledgeBase(path)
    assert "metadata" not in kb.read("a")
    assert "view" not in indexed_tools(kb)["read"].arguments.model_fields
    kb.close()
    kb = IndexedKnowledgeBase(path, document_tools=True)
    toc = kb.read("a", view="toc")["sections"]
    assert toc[0]["title"] == "Rollback criteria (hard stop)"
    assert "failure" in kb.read("a", section=toc[0]["id"])["text"]
    kb.close()
