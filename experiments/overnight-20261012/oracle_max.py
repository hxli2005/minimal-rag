"""One paired effort diagnostic on the same frozen gold-only inputs; no best-of-N."""
import asyncio
import fcntl

import worker
from budget import BudgetModel
from minimal_rag.codex_adapter import rag_codex_config


class EffortModel(BudgetModel):
    def __init__(self, *args, label, **kwargs):
        if label.endswith("/generate"):
            kwargs.update(effort="max", timeout=600)
        super().__init__(*args, label="oracle_max/" + label, **kwargs)


async def run():
    from openai_codex import AsyncCodex
    worker.RESULTS = worker.OUTPUT / "oracle-max-results"
    worker.BudgetModel = EffortModel
    failures = 0
    async with AsyncCodex(rag_codex_config("/private/tmp")) as client:
        account = await client.account(refresh_token=False)
        if getattr(getattr(account.account, "root", account.account), "type", None) != "chatgpt":
            raise RuntimeError("Existing ChatGPT subscription required")
        for case in worker.prepare()["oracle"]:
            if worker.used() + 2 > 500:
                break
            okay = await worker.one(client, case, "oracle", "gold")
            failures = 0 if okay else failures + 1
            if failures >= 3:
                break
    cases = worker.prepare()["oracle"]
    rows = [worker.read(worker.RESULTS / "oracle/gold" / f"{c['id']}.json") for c in cases
            if (worker.RESULTS / "oracle/gold" / f"{c['id']}.json").exists()]
    worker.save(worker.OUTPUT / "oracle-max-summary.json", {"planned": 12,
        "terminal": sum(r.get("terminal", False) for r in rows), "correct": sum(r.get("metric", 0) for r in rows),
        "score_is_final": len(rows)==12 and all(r.get("terminal") for r in rows),
        "effort": "max", "judge_effort": "medium", "model": "gpt-6.1-sol", "calls_reserved_total": worker.used(),
        "cases": [{"id": r["id"], "metric": r.get("metric"), "status": r.get("result", {}).get("status")} for r in rows],
        "limits": ["Gold-only known-failure diagnostic, not a benchmark score.",
            "Same raw gold-document input, prompt, model and one generation as the high-effort run; only generation effort and timeout change.",
            "Both successful and failed grades retained. No reference corrections, no choosing better per-case answer.",
            "Whole-night shared 500 cap includes this experiment and retries; no other model or GPU."]})


if __name__ == "__main__":
    with (worker.OUTPUT / "worker.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another overnight experiment is active")
        asyncio.run(run())
