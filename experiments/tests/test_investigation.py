"""Provenance, refusal, shared budgets and offline delivery; no live model use."""
import asyncio
import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path
from importlib.resources import files

import pytest

from minimal_rag.indexed_rag import SearchArgs
from minimal_rag.loop import Tool
from minimal_rag.indexed_rag import IndexedKnowledgeBase, indexed_tools
from minimal_rag_experiments.evidence import CallBudget, EvidenceError, EvidenceLedger
from minimal_rag_experiments.applications.investigation import SECTIONS, inspect_brief, investigate
from minimal_rag_experiments.applications.investigation_cli import cli
from minimal_rag_experiments.applications.investigation_demo import demo
from minimal_rag_experiments.applications.investigation_report import render_html


def corpus():
    temporary = tempfile.TemporaryDirectory(prefix="investigation-test-")
    path = Path(temporary.name) / "corpus.sqlite"
    IndexedKnowledgeBase.build(path, [{"id": "a", "title": "Incident", "text": "Old deployment caused a latency regression.", "source": "test://a"}])
    kb = IndexedKnowledgeBase(path)
    kb.temporary = temporary
    return kb


def brief(evidence="E1"):
    return "\n".join(f"## {name}\n- " + (
        "待核查：当前部署和监控尚未取得。" if i == 4 else f"工单记录了历史延迟回退。[{evidence}]")
        for i, name in enumerate(SECTIONS))


def reply(text="", calls=(), sources=()):
    return json.dumps({"text": text, "sources": list(sources), "tool_calls": [
        {"name": name, "arguments_json": json.dumps(args)} for name, args in calls]})


def replay(*responses):
    iterator = iter(responses)
    async def complete(messages, tools):
        assert {t["name"] for t in tools} == {"search", "read"}
        return next(iterator)
    return complete


def test_evidence_binds_exact_observed_range_and_rejects_later_change():
    kb = corpus()
    ledger = EvidenceLedger(kb.read)
    row = kb.read("a", offset=4, limit=10)
    observed = ledger.capture(row)
    assert observed["evidence_id"] == ledger.capture(row)["evidence_id"]
    assert ledger.check(["E1"]) == []
    assert ledger.check(["E2"])
    with pytest.raises(EvidenceError):
        ledger.capture({**row, "offset": 5})
    with sqlite3.connect(kb.path) as writer:
        writer.execute("UPDATE documents SET text=? WHERE id=?", ("New unrelated text", "a"))
    assert "changed" in ledger.check(["E1"])[0]


@pytest.mark.asyncio
async def test_partial_failed_tool_response_does_not_publish_unobserved_receipts():
    kb = corpus()
    ledger = EvidenceLedger(kb.read)
    good = kb.search("deployment")["hits"][0]
    bad = {**good, "text": "forged source"}
    tool = Tool("test", SearchArgs, lambda **_: {"hits": [good, bad]})
    with pytest.raises(EvidenceError):
        await ledger.wrap({"search": tool})["search"].execute(query="x")
    assert ledger.export() == []
    assert ledger.check(["E1"])


def test_source_reader_failure_and_revocation_fail_closed():
    kb = corpus()
    ledger = EvidenceLedger(kb.read)
    ledger.capture(kb.read("a"))
    ledger.reader = lambda *_args, **_kwargs: {"error": "not_found_or_not_visible"}
    assert ledger.check(["E1"])
    def unavailable(*a, **kw):
        raise OSError("source unavailable")
    ledger.reader = unavailable
    assert "could not be checked" in ledger.check(["E1"])[0]


def test_passage_provenance_is_explicitly_not_a_semantic_correctness_check():
    kb = corpus()
    ledger = EvidenceLedger(kb.read)
    ledger.capture(kb.read("a"))
    result = inspect_brief(brief().replace("历史延迟回退", "月亮由奶酪构成"), ["E1"], ledger)
    assert result["checks_passed"]
    assert result["semantic_correctness"] == "not_verified"
    assert not inspect_brief(brief("E999"), ["E999"], ledger)["checks_passed"]
    assert not inspect_brief(brief(), [], ledger)["checks_passed"]
    assert not inspect_brief(brief().replace("[E1]", ""), [], ledger)["checks_passed"]


def test_oversized_reference_is_rejected_without_crashing_validation():
    kb = corpus()
    ledger = EvidenceLedger(kb.read)
    invalid = "E" + "9" * 5000
    result = inspect_brief(brief(invalid), [invalid], ledger)
    assert not result["checks_passed"]
    assert result["semantic_correctness"] == "not_verified"


@pytest.mark.asyncio
async def test_actual_search_observation_can_be_cited_without_prescribed_read_step():
    kb = corpus()
    result = await investigate(replay(reply(calls=[("search", {"query": "deployment"})]),
                                     reply(brief(), sources=["E1"])),
                               indexed_tools(kb), kb.read, "What happened?", budget=CallBudget(2))
    assert result["status"] == "draft_for_review"
    assert result["budget"]["used"] == 2
    assert result["validation"]["semantic_correctness"] == "not_verified"
    assert result["evidence"][0]["text"] == kb.read("a")["text"]


@pytest.mark.asyncio
async def test_failed_calls_consume_shared_budget_and_new_run_cannot_reset_it():
    kb, budget, calls = corpus(), CallBudget(1), []
    async def fail(messages, tools):
        calls.append(1)
        raise RuntimeError("transport failed")
    first = await investigate(fail, indexed_tools(kb), kb.read, "one", budget=budget)
    second = await investigate(fail, indexed_tools(kb), kb.read, "two", budget=budget)
    assert first["result"]["status"] == "runtime_error"
    assert second["result"]["status"] == "budget_exhausted"
    assert len(calls) == 1 and second["outer_calls_this_run"] == 0


@pytest.mark.asyncio
async def test_shared_budget_is_reserved_before_concurrent_dispatch():
    kb, budget, calls = corpus(), CallBudget(1), []
    async def complete(messages, tools):
        calls.append(1)
        await asyncio.sleep(.01)
        return reply()
    reports = await asyncio.gather(*(investigate(complete, indexed_tools(kb), kb.read,
                                                f"request {i}", budget=budget) for i in range(4)))
    assert len(calls) == 1
    assert sum(r["result"]["status"] == "budget_exhausted" for r in reports) == 3


@pytest.mark.asyncio
async def test_deadline_stops_dispatch_and_cancellation_propagates():
    kb, calls = corpus(), []
    async def slow(messages, tools):
        calls.append(1)
        await asyncio.sleep(10)
    result = await investigate(slow, indexed_tools(kb), kb.read, "query", budget=CallBudget(3), timeout=.01)
    assert result["result"]["status"] == "timeout" and len(calls) == 1
    class CancelledModel:
        def __init__(self):
            self.runs = []
        async def __call__(self, messages, tools):
            self.runs.append({"status": "interrupted", "usage": None})
            raise asyncio.CancelledError()
    checkpoints = []
    with pytest.raises(asyncio.CancelledError):
        await investigate(CancelledModel(), indexed_tools(kb), kb.read, "query", budget=CallBudget(3),
                          checkpoint=lambda state: checkpoints.append(json.loads(json.dumps(state))))
    assert checkpoints[0]["budget"]["used"] == 1
    assert checkpoints[-1]["status"] == "interrupted"
    assert checkpoints[-1]["model_runs"] == [{"status": "interrupted", "usage": None}]


@pytest.mark.asyncio
async def test_source_recheck_that_finishes_late_cannot_publish_a_ready_draft(monkeypatch):
    from types import SimpleNamespace
    import minimal_rag_experiments.applications.investigation as module
    clock = [0.]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    kb = corpus()
    check = EvidenceLedger.check
    def slow_check(self, identifiers):
        result = check(self, identifiers)
        clock[0] = 3.
        return result
    monkeypatch.setattr(EvidenceLedger, "check", slow_check)
    result = await investigate(replay(reply(calls=[("read", {"document_id": "a"})]),
                                     reply(brief(), sources=["E1"])),
                               indexed_tools(kb), kb.read, "query", budget=CallBudget(2), timeout=2)
    assert result["status"] == "incomplete"
    assert result["result"]["status"] == "timeout"
    assert result["validation"]["checks_passed"]


@pytest.mark.asyncio
async def test_expired_cooperative_deadline_cannot_dispatch_another_call():
    import time
    kb, calls = corpus(), []
    async def blocking(messages, tools):
        calls.append(1)
        time.sleep(.02)
        return reply(calls=[("search", {"query": "deployment"})])
    result = await investigate(blocking, indexed_tools(kb), kb.read, "query", budget=CallBudget(3), timeout=.01)
    assert result["result"]["status"] == "timeout" and len(calls) == 1


@pytest.mark.asyncio
async def test_abstention_is_not_reported_as_a_completed_investigation():
    kb = corpus()
    text = "\n".join(f"## {name}\n- 待核查：没有取得相关资料。" for name in SECTIONS)
    result = await investigate(replay(reply(text)), indexed_tools(kb), kb.read, "query", budget=CallBudget(1))
    assert result["status"] == "insufficient_evidence"
    assert result["validation"]["checks_passed"]


@pytest.mark.asyncio
async def test_demo_uses_packaged_sources_and_real_tools_without_live_calls(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = await demo()
    assert result["status"] == "draft_for_review"
    assert result["generation_calls"] == 0
    assert result["mode"] == "offline_authored_replay"
    assert result["outer_calls_this_run"] == 4
    assert len(result["validation"]["cited"]) == 3
    source = files("minimal_rag_experiments").joinpath("fixtures")
    manifest = json.loads(source.joinpath("northwind-manifest.json").read_text())
    assert hashlib.sha256(source.joinpath("northwind.jsonl").read_bytes()).hexdigest() == manifest["fixture_sha256"]
    stopped = await demo(max_calls=2)
    assert stopped["status"] == "incomplete"
    assert stopped["result"]["status"] == "budget_exhausted"
    assert stopped["budget"]["used"] == 2


def test_cli_rejects_uncapped_live_calls_and_existing_output_before_dispatch(tmp_path, monkeypatch):
    import minimal_rag_experiments.applications.investigation_cli as module
    monkeypatch.setattr(module, "live", lambda *a: pytest.fail("must not dispatch"))
    with pytest.raises(SystemExit):
        cli(["--live", "--index", "absent", "--question", "q", "--output", str(tmp_path / "new")])
    with pytest.raises(SystemExit):
        cli(["--live", "--max-calls", "1", "--index", "absent", "--question", "q", "--output", str(tmp_path)])
    assert list(tmp_path.iterdir()) == []


def test_cli_setup_failure_leaves_a_terminal_zero_call_record(tmp_path, monkeypatch):
    import minimal_rag_experiments.applications.investigation_cli as module
    async def failure(*args, **kwargs):
        raise FileNotFoundError("source index unavailable")
    monkeypatch.setattr(module, "live", failure)
    output = tmp_path / "failed"
    with pytest.raises(FileNotFoundError):
        cli(["--live", "--index", "absent", "--question", "q", "--max-calls", "1", "--output", str(output)])
    state = json.loads((output / "progress.json").read_text())
    assert state["execution_status"] == state["status"] == "failed"
    assert state["stage"] == "investigation" and state["budget"]["used"] == 0


def test_cli_render_failure_preserves_complete_answer_for_offline_recovery(tmp_path, monkeypatch):
    import minimal_rag_experiments.applications.investigation_cli as module
    def failure(*args, **kwargs):
        raise OSError("cannot write rendered report")
    monkeypatch.setattr(module, "save_report", failure)
    output = tmp_path / "failed-render"
    with pytest.raises(OSError):
        cli(["--demo", "--output", str(output)])
    state = json.loads((output / "progress.json").read_text())
    assert state["status"] == "failed" and state["stage"] == "delivery"
    assert state["result"]["status"] == "finished"
    assert state["validation"]["checks_passed"] and state["generation_calls"] == 0


def test_interrupted_usage_is_marked_incomplete_without_adding_cumulative_totals_twice():
    from minimal_rag_experiments.applications.investigation_cli import usage_record
    runs = [
        {"thread_id": "one", "usage": {"total": {"input_tokens": 50, "cached_input_tokens": 10, "output_tokens": 5}}},
        {"thread_id": "one", "usage": {"total": {"input_tokens": 100, "cached_input_tokens": 30, "output_tokens": 11}}},
    ]
    assert usage_record(runs)["token_usage_complete"]
    partial = usage_record(runs + [{"thread_id": "one", "status": "interrupted", "usage": None}])
    assert not partial["token_usage_complete"]
    assert partial["attempts_without_complete_usage"] == [3]
    assert partial["token_usage"] == {"input_tokens": 100, "cached_input_tokens": 30, "output_tokens": 11}


@pytest.mark.asyncio
async def test_html_escapes_source_and_model_markup_and_only_links_local_evidence():
    report = await demo()
    payload = '<script>alert("x")</script><img src="https://example.com/track">'
    report["question"] = payload
    report["evidence"][0]["text"] = payload
    report["evidence"][0]["title"] = payload
    report["validation"]["sections"][SECTIONS[0]][0]["text"] = payload + " [E1]"
    html = render_html(report)
    assert "<script>" not in html and "<img" not in html
    assert "&lt;script&gt;" in html and 'href="#source-E1"' in html
    assert "Content-Security-Policy" in html
    assert 'href="https://' not in html
