"""One recorded retry of cancelled request 80, after billing reconciliation; no new sample."""
import asyncio
import copy
from decimal import Decimal
import json
from pathlib import Path
import time

import httpx

import run as pilot
from minimal_rag.api_adapter import APIModel
from minimal_rag.indexed_rag import IndexedKnowledgeBase, indexed_tools
from minimal_rag.loop import run_agent
from minimal_rag.prompts import SYSTEM_PROMPT


async def main():
    root = pilot.OUT
    lock = (root / "runner.lock").open("a")
    pilot.fcntl.flock(lock.fileno(), pilot.fcntl.LOCK_EX | pilot.fcntl.LOCK_NB)
    archive = root / "recovery-001"
    if archive.exists():
        raise pilot.StopRun("Recovery 001 already exists; never retry it automatically")
    ledger = pilot.read(root / "ledger.json")
    failed = ledger[-1]
    receipt = pilot.read(root / "interruption-receipt.json")
    assert failed["number"] == 80 and failed["id"] == "qst_0368"
    assert failed["status"] == "cost_unresolved" and failed["error_type"] == "RemoteProtocolError"
    assert receipt["cancelled"] is True and receipt["finish_reason"] is None
    assert receipt["provider_name"] == "OpenAI"
    assert not (root / "private-http/00080-response.json").exists()
    request = pilot.read(root / "private-http/00080-request.json")
    original = pilot.read(pilot.report_path(1, "qst_0368", "low"))
    assert not original.get("judge_attempts")
    assert original["judgement"]["reason"] == "Generation failed; kept in denominator."
    pending = pilot.read(pilot.report_path(1, "qst_0253", "low"))
    assert pending["result"]["error"] == "StopRun" and not pending["events"]
    assert not any(r["id"] == "qst_0253" for r in ledger)
    headers = {"Authorization": "Bearer " + pilot.KEY.read_text().strip(), "X-OpenRouter-Metadata": "enabled",
               "X-OpenRouter-Title": "Minimal RAG fixed API validation"}
    async with httpx.AsyncClient(base_url="https://openrouter.ai/api/v1/", headers=headers) as client:
        guard = pilot.GuardedClient(client)
        key = await guard.key_state()
        known = sum((pilot.money(r["cost_usd"]) for r in ledger[:-1]), Decimal(0))
        assert abs(pilot.money(key["usage"]) - known - pilot.money(receipt["total_cost"])) < Decimal("0.000001")
        archive.mkdir(mode=0o700)
        pilot.save(archive / "original-ledger.json", ledger)
        pilot.save(archive / "original-failed-case.json", original)
        pilot.save(archive / "never-sent-placeholder.json", pending)
        decision = {"kind": "same_config_pre_complete_output_transport_recovery", "time": time.time(),
                    "recovery_source_sha256": pilot.hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "original_request": 80, "original_generation_id": receipt["id"],
                    "cancelled": True, "delivered_complete_response": False, "valid_answer_resampled": False,
                    "charged_usd": receipt["total_cost"], "key_usage_reconciled_usd": key["usage"],
                    "policy": pilot.PLAN["failures"]["generation_retry"],
                    "retry": "Exactly the saved pending request body, once; keep previous model and tool history.",
                    "judge": "Original automatic failed-generation zero is archived; no valid model judgement existed.",
                    "never_sent_placeholder": "qst_0253 is archived and will be generated for the first time.",
                    "status": "started"}
        pilot.save(archive / "decision.json", decision)
        failed.update(status="settled", cost_usd=receipt["total_cost"], response_id=receipt["id"],
                      receipt=receipt, transport_cancelled=True, cost_source="generation receipt; matched dedicated-key usage delta")
        guard.rows = ledger
        guard.persist(failed)

        # Enforce byte-equivalent JSON values at the actual outgoing request boundary.
        expected = copy.deepcopy(request)
        for k in ("provider", "service_tier", "max_output_tokens", "truncation"):
            expected.pop(k)
        expected["reasoning"].pop("context")
        original_post = guard.post
        first_post = True
        async def checked_post(endpoint, *, json, timeout):
            nonlocal first_post
            if first_post:
                assert json == expected, "Recovery request differs from cancelled request"
                first_post = False
            return await original_post(endpoint, json=json, timeout=timeout)
        guard.post = checked_post
        guard.begin(1, "qst_0368", "low")
        guard.previous_reasoning = [copy.deepcopy(x) for x in request["input"] if x.get("type") == "reasoning"]

        class ResumeModel(APIModel):
            first = True
            async def __call__(self, messages, tools):
                if self.first:
                    self.first = False
                    self._messages = messages
                    self._cursor = len(messages)
                    self._history = copy.deepcopy(request["input"])
                    self._pending = []
                return await super().__call__(messages, tools)

        record = copy.deepcopy(original)
        record.pop("result")
        record.pop("judgement")
        record["recovery"] = {"decision": "recovery-001/decision.json", "original_request": 80}
        model = ResumeModel(guard, pilot.MODEL, 300, "high", SYSTEM_PROMPT, "low")
        model.runs = record["model_runs"]
        path = pilot.report_path(1, "qst_0368", "low")
        pilot.save(path, record)
        def emit(event):
            record["events"].append(event)
            pilot.save(path, record)
        start = time.monotonic()
        try:
            kb = IndexedKnowledgeBase(pilot.ROOT / "data/benchmarks/enterprise/corpus.sqlite", 4000)
            try:
                message = "Answer the question using the supplied corpus and cite supporting documents.\nQuestion: " + record["question"]
                record["result"] = await asyncio.wait_for(run_agent(model, indexed_tools(kb), message, None, emit), max(1, 3600-original["seconds"]))
            finally:
                kb.close()
            if record["result"]["status"] != "finished":
                raise pilot.StopRun("Recovered question did not finish")
            pilot.report_path(1, "qst_0253", "low").rename(archive / "never-sent-placeholder-moved.json")
            decision["status"] = "recovered_original_question_without_restart"
            # Preserve the stop record; the unchanged runner can now resume completed cases.
            if (root / "stop.json").exists():
                (root / "stop.json").rename(archive / "original-stop.json")
        except BaseException as error:
            record["result"] = {"status": "runtime_error", "answer": "", "sources": [],
                                "error": type(error).__name__, "detail": str(error)}
            decision.update(status="recovery_failed", error=str(error))
            raise
        finally:
            record["seconds"] = original["seconds"] + time.monotonic()-start
            pilot.save(path, record)
            pilot.save(archive / "decision.json", decision)
            pilot.overview()
        print(json.dumps({"recovery": decision["status"], "question": "qst_0368", "paid_requests": len(guard.rows),
                          "total_cost_usd": sum(r["cost_usd"] or 0 for r in guard.rows)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
