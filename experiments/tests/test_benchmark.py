"""Current entry keeps tool autonomy, separates labels, and resumes without resampling."""
from argparse import Namespace
import json
from types import SimpleNamespace

import pytest

from minimal_rag.indexed_rag import IndexedKnowledgeBase
from minimal_rag_experiments import benchmark


@pytest.mark.parametrize("dataset", benchmark.DATASETS)
def test_all_benchmarks_use_the_same_search_read_corpus(tmp_path, monkeypatch, dataset):
    monkeypatch.setattr(benchmark, "BASE", tmp_path)
    case = {"id": "q", "raw": {"context": [["Manual", ["Use the retry setting."]]], "answer": "SECRET"}}
    if dataset != "hotpotqa":
        folder = tmp_path / ("browsecomp-plus" if dataset == "browsecomp" else dataset)
        IndexedKnowledgeBase.build(folder / "corpus.sqlite", [{"id": "D0", "title": "Manual", "text": "Use the retry setting.", "source": "test://D0"}])
    with benchmark.knowledge_base(dataset, case, tmp_path) as kb:
        assert type(kb) is IndexedKnowledgeBase
        assert kb.search("retry")["hits"][0]["document_id"] == "D0"
        assert kb.read("D0")["text"] == "Use the retry setting."
        assert "SECRET" not in json.dumps(kb.search(""))


def test_browsecomp_export_counts_search_but_not_read_only_documents():
    report = {"id": "q", "events": [
        {"role": "tool", "name": "search", "content": {"hits": [{"document_id": "a"}]}},
        {"role": "tool", "name": "read", "content": {"document_id": "b"}}],
        "result": {"status": "finished", "answer": "result"}}
    exported = benchmark.official_export(report)
    assert exported["retrieved_docids"] == ["a"]
    assert exported["tool_call_counts"] == {"search": 1, "read": 1}


@pytest.mark.asyncio
async def test_current_entry_keeps_agent_search_and_read_and_judges_only_after_answer(tmp_path, monkeypatch):
    import openai_codex
    monkeypatch.setattr(benchmark, "BASE", tmp_path)
    case = {"id": "q", "question": "Which setting?", "category": "simple", "gold": "SECRET", "facts": ["SECRET"], "gold_ids": ["D0"]}
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps(case))
    monkeypatch.setattr(benchmark, "load_cases", lambda *args: (sample, [case]))
    IndexedKnowledgeBase.build(tmp_path / "enterprise/corpus.sqlite", [{"id": "D0", "title": "Manual", "text": "Use retry.", "source": "test://D0"}])

    class Client:
        def __init__(self, *args): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def account(self, **kwargs): return SimpleNamespace(account=SimpleNamespace(type="chatgpt"))

    (tmp_path / "enterprise/index.json").write_text('{"documents": 1, "chunks": 1}')
    calls = []
    class Model:
        def __init__(self, *args, **kwargs): self.runs = []; self.step = 0
        async def __call__(self, messages, tools):
            assert "SECRET" not in json.dumps(messages)
            assert {t["name"] for t in tools} == {"search", "read"}
            calls.append(self.step)
            replies = [
                {"text": "", "tool_calls": [{"name": "search", "arguments_json": '{"query":"retry"}'}]},
                {"text": "", "tool_calls": [{"name": "read", "arguments_json": '{"document_id":"D0"}'}]},
                {"text": "Use retry.", "tool_calls": [], "sources": ["D0"]}]
            reply = replies[self.step]; self.step += 1
            self.runs.append({"status": "completed"})
            return json.dumps(reply)

    async def judge(factory, payload, count, attempts, persist):
        assert payload["reference_answer"] == "SECRET"
        assert payload["candidate_answer"] == "Use retry."
        return {"correct": True, "facts_supported": [True], "reason": "fixture"}, []

    monkeypatch.setattr(openai_codex, "AsyncCodex", Client)
    monkeypatch.setattr(benchmark.rag, "CodexModel", Model)
    monkeypatch.setattr(benchmark, "assess", judge)
    args = Namespace(dataset="enterprise", split="dev", output=str(tmp_path / "run"), limit=None,
                     model="fixture", concurrency=1, judge_concurrency=1, resume=False)
    assert await benchmark.main(args) == 0
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert summary["complete"] and summary["correct_over_requested"] == 1
    assert summary["metrics"]["tool_calls"] == 2
    args.resume = True
    assert await benchmark.main(args) == 0
    assert calls == [0, 1, 2]
