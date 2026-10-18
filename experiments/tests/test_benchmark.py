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
    openai_codex = pytest.importorskip("openai_codex")
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


@pytest.mark.asyncio
@pytest.mark.parametrize("skip_judge", [False, True])
async def test_api_benchmark_and_resume_need_no_codex_and_keep_labels_out_of_generation(tmp_path, monkeypatch, skip_judge):
    import builtins
    httpx = pytest.importorskip("httpx")
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")
    original_import = builtins.__import__
    def without_codex(name, *args, **kwargs):
        if name.startswith("openai_codex"):
            raise ImportError("Codex not installed")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_codex)
    monkeypatch.setattr(benchmark, "BASE", tmp_path)
    case = {"id": "q", "question": "Which setting?", "category": "simple", "gold": "SECRET",
            "facts": ["SECRET"], "gold_ids": ["D0"]}
    source = tmp_path / "sample.json"; source.write_text(json.dumps(case))
    monkeypatch.setattr(benchmark, "load_cases", lambda *args: (source, [case]))
    IndexedKnowledgeBase.build(tmp_path / "enterprise/corpus.sqlite",
        [{"id": "D0", "title": "Manual", "text": "Use retry.", "source": "test://D0"}])
    (tmp_path / "enterprise/index.json").write_text('{"documents":1}')
    bodies = []
    def serve(request):
        body = json.loads(request.content); bodies.append(body)
        if "blind" in body["instructions"]:
            assert body["text"]["verbosity"] == "medium"
            assert not skip_judge and len(body["input"]) == 1
            assert "SECRET" in json.dumps(body) and "Use retry." in json.dumps(body)
            text = json.dumps({"correct": True, "facts_supported": [True], "reason": "fixture"})
            output = [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}]
            assert "tools" not in body and "format" not in body["text"]
        else:
            assert body["text"]["verbosity"] == "low"
            assert "SECRET" not in json.dumps(body)
            assert "outer text field" not in json.dumps(body) and "arguments_json" not in json.dumps(body)
            if len(bodies) == 1:
                assert {t["name"] for t in body["tools"]} == {"search", "read"}
                output = [{"type": "function_call", "name": "search", "call_id": "call_search",
                           "arguments": '{"query":"retry"}'}]
            else:
                assert body["input"][-1]["type"] == "function_call_output"
                assert body["input"][-1]["call_id"] == "call_search"
                output = [{"type": "message", "role": "assistant", "content": [{"type": "output_text",
                           "text": json.dumps({"answer": "Use retry.", "sources": ["D0"]})}]}]
        return httpx.Response(200, json={"id": f"resp_{len(bodies)}", "status": "completed",
            "usage": {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
            "output": output})
    client_type = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client_type(transport=httpx.MockTransport(serve), **kw))
    args = Namespace(dataset="enterprise", split="dev", output=str(tmp_path / "run"), limit=None,
                     model="fixture-api", backend="api", concurrency=1, judge_concurrency=1,
                     resume=False, skip_judge=skip_judge)
    assert await benchmark.main(args) == 0
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert summary["complete"] and summary["config"]["settings"]["max_turns"] is None
    assert summary["config"]["verbosity"] == "low" and summary["config"]["judge_verbosity"] == "medium"
    assert summary["metrics"]["input_tokens"] == 200
    if skip_judge:
        assert summary["correct_over_requested"] is None
    assert summary["requests"] == {"generation": 2, "judge": 0 if skip_judge else 1}
    args.resume = True
    assert await benchmark.main(args) == 0
    assert len(bodies) == (2 if skip_judge else 3)
