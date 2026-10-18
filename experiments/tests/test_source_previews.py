"""Source discovery must preserve pagination and never certify unread text."""
import asyncio

import pytest

from minimal_rag_experiments.evidence import EvidenceLedger
from minimal_rag.indexed_rag import IndexedKnowledgeBase, indexed_tools
from minimal_rag_experiments.source_previews import source_preview_tools


@pytest.fixture
def kb(tmp_path):
    path = tmp_path / "docs.sqlite"
    IndexedKnowledgeBase.build(path, [{"id": f"d{i:02d}", "title": f"Incident {i}",
                                      "source": f"test://{i}", "text": f"timeout diagnosis {i}"}
                                     for i in range(25)])
    instance = IndexedKnowledgeBase(path)
    yield instance
    instance.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("async_backend", [False, True])
async def test_previews_keep_original_page_and_only_read_passages_become_evidence(kb, async_backend):
    tools = indexed_tools(kb)
    original, calls = tools["search"].execute, []
    def execute(**args):
        calls.append(args)
        result = original(**args)
        if not async_backend:
            return result
        async def later():
            await asyncio.sleep(0)
            return result
        return later()
    tools["search"].execute = execute
    ledger = EvidenceLedger(kb.read)
    wrapped = ledger.wrap(source_preview_tools(tools, 15))
    observed = await wrapped["search"].execute(query="timeout", limit=5, offset=5)
    expected = kb.search("timeout", limit=5, offset=5)
    assert len(calls) == 1 and calls[0]["limit"] == 20
    assert [{k: v for k, v in row.items() if k not in {"evidence_id", "passage_sha256"}}
            for row in observed["hits"]] == expected["hits"]
    assert observed["next_offset"] == expected["next_offset"] == 10
    assert len(observed["other_matches"]) == 15
    pointer = observed["other_matches"][0]
    assert set(pointer) == {"document_id", "title", "source", "offset"}
    assert len(ledger.export()) == 5
    assert ledger.check([pointer["document_id"]])
    receipt = await wrapped["read"].execute(document_id=pointer["document_id"], offset=pointer["offset"])
    assert ledger.check([receipt["evidence_id"]]) == []
    assert len(ledger.export()) == 6


@pytest.mark.asyncio
async def test_last_page_scoped_search_directory_and_disabled_previews(kb):
    tools = indexed_tools(kb)
    preview = source_preview_tools(tools, 15)["search"]
    last = await preview.execute(query="timeout", limit=5, offset=23)
    assert len(last["hits"]) == 2 and last["next_offset"] is None
    assert last["other_matches"] == []
    for args in ({"query": "timeout", "scope": "d01"}, {"query": ""}):
        assert await preview.execute(**args) == kb.search(**args)
    assert source_preview_tools(tools, 0)["search"] is tools["search"]
    with pytest.raises(ValueError):
        source_preview_tools(tools, 16)


@pytest.mark.asyncio
async def test_preview_propagates_backend_errors_and_cancellation(kb):
    tools = indexed_tools(kb)
    for exception in (RuntimeError, asyncio.CancelledError):
        async def fail(**args):
            raise exception("unavailable")
        tools["search"].execute = fail
        wrapped = source_preview_tools(tools, 15)
        with pytest.raises(exception):
            await wrapped["search"].execute(query="timeout")
