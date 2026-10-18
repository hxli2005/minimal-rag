"""Bounded provider compatibility probe: at most three calls, no judge or retry."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile

from minimal_rag import codex_adapter as rag
from minimal_rag.loop import run_agent
from minimal_rag.indexed_rag import indexed_tools
from minimal_rag.demo import fixture_index
from minimal_rag_experiments.evidence import CallBudget


async def run(output):
    from openai_codex import AsyncCodex
    output.mkdir(parents=True, exist_ok=False)
    budget = CallBudget(3)
    cases = [
        {"id": "without_evidence", "max_calls": 1,
         "question": "请先搜索 RECOVERY-UNKNOWN 事故，告诉我当前是否已恢复。没有证据请明确说无法判断。"},
        {"id": "with_search", "max_calls": 2,
         "question": "请先检索资料，依据来源回答：MUG-C02 的无理由退货规则是什么？"},
    ]
    plan = {"model": "gpt-6.1-sol", "total_call_cap": 3, "cases": cases,
            "judge_calls": 0, "retries": 0, "resume": False,
            "note": "SDK final-response-schema compatibility probe on fictional teaching data; not a quality benchmark."}
    (output / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    manifest = {}
    for path in (Path(__file__), Path(rag.__file__)):
        (snapshot / path.name).write_bytes(path.read_bytes())
        manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (snapshot / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records = []
    with tempfile.TemporaryDirectory(prefix="rag-budget-probe-") as cwd:
        async with AsyncCodex(rag.rag_codex_config(cwd)) as client:
            wrapper = await client.account(refresh_token=False)
            if getattr(getattr(wrapper.account, "root", wrapper.account), "type", None) != "chatgpt":
                raise RuntimeError("requires existing ChatGPT login")
            for case in cases:
                model = rag.CodexModel(client, cwd, "gpt-6.1-sol", max_calls=case["max_calls"], timeout=120)
                record = {"case": case, "model_runs": model.runs, "events": []}
                def save():
                    record["budget"] = budget.snapshot()
                    (output / f"{case['id']}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
                async def limited(messages, tools):
                    budget.reserve()
                    save()
                    return await model(messages, tools)
                def emit(event):
                    record["events"].append(event)
                    save()
                try:
                    with fixture_index() as kb:
                        record["result"] = await run_agent(limited, indexed_tools(kb),
                                                          case["question"], case["max_calls"], emit)
                except Exception as error:
                    record["result"] = {"status": "runtime_error", "error": type(error).__name__, "detail": str(error)}
                finally:
                    save()
                records.append({"id": case["id"], "result": record["result"], "calls": len(model.runs),
                                "final_schema_exercised": any(r["final_response_required"] for r in model.runs),
                                "tool_calls": sum(e.get("role") == "tool" for e in record["events"])})
    summary = {"plan": plan, "budget": budget.snapshot(), "records": records,
               "all_finished": all(r["result"]["status"] == "finished" for r in records)}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args().output))
