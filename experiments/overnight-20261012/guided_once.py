"""Stronger fixed-prefix baseline: model chooses one query, then answers."""
import asyncio
import fcntl
import json
import statistics
import tempfile
import time

from budget import BudgetModel
from minimal_rag.codex_adapter import SYSTEM_PROMPT, rag_codex_config, token_usage
from minimal_rag.indexed_rag import IndexedKnowledgeBase, indexed_tools
from minimal_rag.loop import Tool, run_agent
from minimal_rag_experiments.rag_judge import PROMPT, assess
from minimal_rag_experiments.rag_optimization_eval import NATIVE_FORMAT
from worker import BASE, OUTPUT, RESULTS, prepare, read, save, used


async def run():
    from openai_codex import AsyncCodex
    cases = prepare()["datasets"]["enterprise"][:12]
    async with AsyncCodex(rag_codex_config("/private/tmp")) as client:
        account = await client.account(refresh_token=False)
        if getattr(getattr(account.account, "root", account.account), "type", None) != "chatgpt":
            raise RuntimeError("Existing ChatGPT subscription required")
        for case in cases:
            path = RESULTS / "enterprise/guided_once" / f"{case['id']}.json"
            report = read(path) if path.exists() else {"id": case["id"], "question": case["question"],
                "category": case["category"], "events": [], "search_count": 0,
                "constraint": "at most one BM25 query, first five distinct documents, no read; max two generation turns"}
            if report.get("terminal"):
                continue
            if used() + (1 if "result" in report else 3) > 480:
                break
            with tempfile.TemporaryDirectory(prefix="rag-guided-once-") as cwd:
                if "result" not in report:
                    kb = IndexedKnowledgeBase(BASE / "enterprise/corpus.sqlite", 4000, check_same_thread=False)
                    search = indexed_tools(kb)["search"]
                    async def execute(**kwargs):
                        if report["search_count"]:
                            return {"error": "This baseline permits exactly one retrieval query."}
                        report["search_count"] += 1
                        kwargs.update(limit=5, offset=0)
                        report["effective_search"] = kwargs
                        return await asyncio.to_thread(kb.search, **kwargs)
                    tools = {"search": Tool(search.description + " This baseline executes at most one query and always returns the first five documents. No read or further search is available.", search.arguments, execute)}
                    model = BudgetModel(client, cwd, label=f"guided_once/{case['id']}/generate", timeout=180,
                        system_prompt=SYSTEM_PROMPT, effort="high", max_calls=2)
                    report["model_runs"] = model.runs
                    began = time.monotonic()
                    try:
                        report["result"] = await asyncio.wait_for(run_agent(model, tools,
                            NATIVE_FORMAT + case["question"] + "\nChoose one useful search query, then answer from its observation. You have only one retrieval and no read tool.",
                            2, lambda event: (report["events"].append(event), save(path, report))), 600)
                    except Exception as error:
                        report.update(error=str(error), result={"status": "runtime_error", "answer": "", "sources": []})
                    finally:
                        kb.close()
                    report["generation_seconds"] = time.monotonic() - began
                    save(path, report)
                if report["result"]["status"] == "finished":
                    payload = {"question": case["question"], "reference_answer": case["gold"],
                        "reference_facts": case["facts"], "candidate_answer": report["result"]["answer"]}
                    report["judgement"], _ = await assess(lambda: BudgetModel(client, cwd,
                        label=f"guided_once/{case['id']}/judge", timeout=150, effort="medium", system_prompt=PROMPT),
                        payload, len(case["facts"]), report.setdefault("judge_attempts", []), lambda: save(path, report))
                    report["metric"] = float(report["judgement"]["correct"])
                else:
                    report["metric"] = 0.
                report["terminal"] = True
                save(path, report)
                print(json.dumps({"id": case["id"], "correct": report["metric"], "calls": used()}), flush=True)
    rows = [read(RESULTS / "enterprise/guided_once" / f"{case['id']}.json") for case in cases
            if (RESULTS / "enterprise/guided_once" / f"{case['id']}.json").exists()]
    save(OUTPUT / "guided-once-summary.json", {"planned": 12, "terminal": sum(r.get("terminal", False) for r in rows),
        "score_is_final": len(rows) == 12 and all(r.get("terminal") for r in rows),
        "correct": sum(r.get("metric", 0) for r in rows), "generation_calls": sum(len(r.get("model_runs", [])) for r in rows),
        "generation_input_tokens": sum(token_usage(r.get("model_runs", [])).get("input_tokens", 0) for r in rows),
        "ids": [c["id"] for c in cases], "cases": [{"id": r["id"], "metric": r.get("metric"),
            "searches": r["search_count"], "status": r.get("result", {}).get("status")} for r in rows],
        "limitations": ["Fixed first 12 of the original frozen stratified dev cohort, not a new holdout.",
            "Model-generated query, exactly one retrieval of five documents, two generation turns; no read.",
            "Added after full natural-question baseline completed; not selected by candidate scores."]})


if __name__ == "__main__":
    with (OUTPUT / "worker.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another overnight experiment is active; do not overlap.")
        asyncio.run(run())
