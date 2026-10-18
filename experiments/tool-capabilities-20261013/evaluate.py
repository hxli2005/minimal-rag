"""Two24-case tool-capability arms, unchanged system prompt,336 reserved calls."""
import asyncio
import json
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/overnight-20261012"))
import budget
from minimal_rag.codex_adapter import SYSTEM_PROMPT, rag_codex_config, token_usage
from minimal_rag_experiments.document_tools import IndexedKnowledgeBase, indexed_tools
from minimal_rag.loop import Tool, run_agent
from minimal_rag_experiments.rag_judge import PROMPT, assess
from minimal_rag_experiments.rag_optimization_eval import NATIVE_FORMAT

OUTPUT = ROOT / "runs/tool-capabilities-20261013"
budget.LEDGER, budget.LIMIT = OUTPUT / "calls.jsonl", 336
cases = json.loads((ROOT / "experiments/overnight-20261012/cohort.json").read_text())["datasets"]["enterprise"]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def summary():
    groups = {}
    for arm in ("baseline", "document_tools"):
        rows = [json.loads(p.read_text()) for p in (OUTPUT / arm).glob("*.json")]
        terminal = [r for r in rows if r.get("terminal")]
        groups[arm] = {"planned": 24, "completed": len(terminal), "correct": sum(r.get("metric", 0) for r in terminal),
                       "score": sum(r.get("metric", 0) for r in terminal) / 24,
                       "generation_calls": sum(len(r.get("model_runs", [])) for r in rows),
                       "generation_input_tokens": sum(token_usage(r.get("model_runs", [])).get("input_tokens", 0) for r in rows),
                       "cases": [{"id":r["id"],"metric":r.get("metric"),"status":r.get("result",{}).get("status")} for r in terminal]}
    result = {"groups": groups, "calls": len(budget.LEDGER.read_text().splitlines()) if budget.LEDGER.exists() else 0,
              "complete": all(g["completed"] == 24 for g in groups.values())}
    if result["complete"]:
        scores = [{r["id"]:r["metric"] for r in groups[arm]["cases"]} for arm in ("baseline","document_tools")]
        deltas = [scores[1][c["id"]]-scores[0][c["id"]] for c in cases]
        result["paired"] = {"wins":sum(d>0 for d in deltas),"losses":sum(d<0 for d in deltas),"ties":sum(d==0 for d in deltas),"gain":statistics.mean(deltas)}
    save(OUTPUT / "summary.json", result)
    return result


async def one(client, case, arm):
    path = OUTPUT / arm / f"{case['id']}.json"
    report = json.loads(path.read_text()) if path.exists() else {"id":case["id"],"events":[],"model_runs":[],"judge_attempts":[]}
    if report.get("terminal"):
        return
    kb = IndexedKnowledgeBase(ROOT / "data/benchmarks/enterprise/corpus.sqlite", check_same_thread=False,
                              document_tools=arm=="document_tools")
    with tempfile.TemporaryDirectory(prefix="rag-document-tools-") as cwd:
        try:
            if "result" not in report:
                model = budget.BudgetModel(client, cwd, label=f"{arm}/{case['id']}/generate", effort="high", timeout=180,
                                           max_calls=6, system_prompt=SYSTEM_PROMPT)
                report["model_runs"] = model.runs
                tools = {}
                for name, tool in indexed_tools(kb).items():
                    async def execute(_fn=tool.execute, **kwargs):
                        return await asyncio.to_thread(_fn, **kwargs)
                    tools[name] = Tool(tool.description, tool.arguments, execute)
                report["result"] = await asyncio.wait_for(run_agent(model, tools, NATIVE_FORMAT+case["question"],6,
                    lambda e:(report["events"].append(e),save(path,report))),900)
                save(path, report)
            if report["result"]["status"] != "finished":
                report.update(metric=0,terminal=True)
            else:
                grade,_ = await assess(lambda:budget.BudgetModel(client,cwd,label=f"{arm}/{case['id']}/judge",effort="medium",timeout=150,
                    max_calls=1,system_prompt=PROMPT),{"question":case["question"],"reference_answer":case["gold"],
                    "reference_facts":case["facts"],"candidate_answer":report["result"]["answer"]},len(case["facts"]),report["judge_attempts"],lambda:save(path,report))
                report.update(judgement=grade,metric=float(grade["correct"]),terminal=True)
            save(path,report)
        finally:
            kb.close()
            save(path,report)


async def main():
    from openai_codex import AsyncCodex
    save(OUTPUT / "protocol.json", {"model":"gpt-6.1-sol","calls_limit":336,"cases":[c["id"] for c in cases],
        "arms":["baseline","document_tools"],"system_prompt":SYSTEM_PROMPT,"generation_effort":"high","max_generation_calls":6,
        "judge":"sol medium, same frozen references","context_chars":4000,"index":"unchanged full SQLite BM25",
        "ordering":"Interleaved baseline/candidate per case","selection":"Exact prior frozen dev24; known development set, not untouched.",
        "changed":"Tool response metadata and read toc/section capability only; no system prompt change."})
    async with AsyncCodex(rag_codex_config("/private/tmp")) as client:
        account=await client.account(refresh_token=False)
        if getattr(getattr(account.account,"root",account.account),"type",None)!="chatgpt":
            raise RuntimeError("Existing subscription only; no API-key fallback")
        for case in cases:
            for arm in ("baseline","document_tools"):
                await one(client,case,arm)
                current=summary()
                print(json.dumps({"case":case["id"],"arm":arm,"calls":current["calls"],"completed":{a:g["completed"] for a,g in current["groups"].items()}},ensure_ascii=False),flush=True)
    print(json.dumps(summary(),ensure_ascii=False))


if __name__=="__main__":
    asyncio.run(main())
