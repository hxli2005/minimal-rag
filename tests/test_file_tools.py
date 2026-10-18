"""Source text, real ripgrep semantics, pagination, and corpus boundaries."""
import json
import sqlite3
from urllib.parse import quote

import pytest

from minimal_rag.file_tools import CorpusFiles
from minimal_rag.indexed_rag import IndexedKnowledgeBase
from minimal_rag.loop import run_agent


@pytest.fixture
def corpus(tmp_path):
    documents = [
        {"id": "a", "title": "Runbook", "text": "Heading\nAlpha-42 starts\nnext line\nALPHA-42 again\nEnd\n", "source": "test://a"},
        {"id": "b:p1", "title": "Other", "text": "Alpha-42\r\nalpha-99\x00 visible after NUL\r\n", "source": "test://b"},
        {"id": "c", "title": "Empty", "text": "", "source": "test://c"},
    ]
    root = tmp_path / "files"
    (root / "text/team").mkdir(parents=True)
    with sqlite3.connect(root / "catalog.sqlite") as db:
        db.execute("CREATE TABLE files(id TEXT PRIMARY KEY,path TEXT UNIQUE)")
        for doc in documents:
            path = f"team/{quote(doc['id'], safe='')}--{doc['title']}.txt"
            (root / "text" / path).write_bytes(doc["text"].encode())
            db.execute("INSERT INTO files VALUES (?,?)", (doc["id"], path))
    index = tmp_path / "corpus.sqlite"
    IndexedKnowledgeBase.build(index, documents)
    kb, files = IndexedKnowledgeBase(index), CorpusFiles(root)
    yield kb, files
    kb.close(); files.close()


@pytest.mark.asyncio
async def test_grep_context_case_multiline_and_real_errors(corpus):
    _, files = corpus
    result = await files.grep("Alpha-42", output_mode="content", ignore_case=True, before=1, after=1,
                              path="/corpus/team/a--Runbook.txt")
    assert "a--Runbook.txt:2:Alpha-42 starts" in result["content"]
    assert "a--Runbook.txt-1-Heading" in result["content"]
    assert files.evidence_ids("Grep", result) == ["a"]
    assert (await files.grep("starts.*next", multiline=True))["numFiles"] == 1
    assert (await files.grep("starts.*next"))["numFiles"] == 0
    assert (await files.grep("visible after NUL"))["numFiles"] == 1
    result = await files.grep("visible after NUL", output_mode="content")
    assert files.evidence_ids("Grep", result) == ["b:p1"]
    with pytest.raises(ValueError, match="regex"):
        await files.grep("[")


@pytest.mark.asyncio
async def test_grep_pagination_and_count_total(corpus):
    _, files = corpus
    first = await files.grep("Alpha", head_limit=1)
    second = await files.grep("Alpha", head_limit=1, offset=1)
    assert first["totalFiles"] == 2 and first["truncated"]
    assert first["filenames"] != second["filenames"] and not second["truncated"]
    counts = await files.grep("alpha", output_mode="count", ignore_case=True, head_limit=1)
    assert counts["numMatches"] == 4
    assert files.evidence_ids("Grep", counts) == []
    assert (await files.grep("Alpha", offset=20))["notice"].startswith("No entries")


@pytest.mark.asyncio
async def test_grep_corpus_boundary_and_glob_filter(corpus):
    _, files = corpus
    result = await files.grep("Alpha", glob="**/*Runbook.txt")
    assert result["filenames"] == ["/corpus/team/a--Runbook.txt"]
    with pytest.raises(ValueError, match="inside"):
        await files.grep("Alpha", path="/etc/passwd")


@pytest.mark.asyncio
async def test_original_parameter_names_work_through_unchanged_agent_loop(corpus):
    kb, files = corpus
    tools = files.tools(kb)
    assert set(tools) == {"search", "read", "Grep"}
    replies = iter([
        {"text": "", "tool_calls": [{"name": "Grep", "arguments_json": json.dumps({
            "pattern": "ALPHA-42", "-i": True, "-C": 1, "-n": True, "output_mode": "content"})}]},
        {"text": "found", "tool_calls": [], "sources": ["a"]},
    ])
    async def complete(messages, specs):
        assert "-i" in next(t for t in specs if t["name"] == "Grep")["parameters"]["properties"]
        return json.dumps(next(replies))
    events = []
    result = await run_agent(complete, tools, "Find the identifier.", emit=events.append)
    assert result["status"] == "finished"
    observation = next(e["content"] for e in events if e["role"] == "tool")
    assert observation["numFiles"] == 2 and "error" not in observation
    assert tools["search"].execute(query="Alpha")["hits"][0]["file_path"].startswith("/corpus/")
