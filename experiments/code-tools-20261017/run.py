"""Paired Enterprise dev100: current tools vs additive file tools, interleaved."""
import argparse
import asyncio
from collections import Counter
import inspect
import json
from pathlib import Path
import tempfile
import time

from minimal_rag import codex_adapter as rag
from frozen_file_tools import CorpusFiles, SDK_VERSION
from minimal_rag.indexed_rag import indexed_tools
from minimal_rag.loop import Tool, run_agent
from minimal_rag_experiments import benchmark
from minimal_rag_experiments.benchmark_metrics import aggregate, score
from minimal_rag_experiments.hotpot_benchmark import save
from minimal_rag_experiments.rag_judge import PROMPT, assess


def summary(config, cases, reports, aliases):
    rows = []
    for case in cases:
        report = reports.get(case["id"], {})
        if "result" not in report:
            continue
        row = score(case, report, "enterprise", None)
        # Count text exposure from the new tools separately from path discovery.
        seen = set()
        for event in report.get("events", []):
            if event.get("role") != "tool":
                continue
            seen.update(event.get("observed_document_ids", []))
            content = event["content"]
            for hit in content.get("hits", [content]):
                if hit.get("document_id") and hit.get("text"):
                    seen.add(hit["document_id"])
        canonical = {aliases.get(doc, doc) for doc in seen}
        gold = set(case["gold_ids"])
        row["retrieved_document_recall"] = len(gold & canonical) / len(gold) if gold else None
        row["all_gold_retrieved"] = gold <= canonical if gold else None
        row["unseen_citations"] = sorted(set(row["sources"]) - seen)
        rows.append(row)
    counts = Counter(event["name"] for report in reports.values() for event in report.get("events", []) if event.get("role") == "tool")
    metrics = aggregate(rows)
    return {"config": config, "requested": len(cases), "completed": len(rows),
            "complete": len(rows) == len(cases) and metrics["graded"] == len(cases),
            "correct": sum(row.get("correct") or 0 for row in rows), "metrics": metrics,
            "tool_calls": dict(counts), "cases": rows,
            "requests": {"generation": sum(len(r.get("model_runs", [])) for r in reports.values()),
                         "judge": sum(len(a.get("runs", [])) for r in reports.values() for a in r.get("judge_attempts", []))}}


async def main(args):
    from openai_codex import AsyncCodex
    output = Path(args.output)
    _, cases = benchmark.load_cases("enterprise", "dev")
    if len(cases) != 100:
        raise ValueError("the frozen Enterprise development cohort must have 100 cases")
    cfg = benchmark.settings("enterprise")
    template = benchmark.format_request("enterprise")
    config = {"dataset": "enterprise", "split": "dev", "model": args.model, "settings": cfg,
              "ids": [c["id"] for c in cases], "format_request": template, "judge_model": args.model,
              "judge_effort": "medium", "judge_prompt": PROMPT, "concurrency": args.concurrency,
              "judge_concurrency": 1, "file_tools": str(Path(args.files).resolve()), "sdk_interface_version": SDK_VERSION,
              "A": "current search/read", "B": "A plus Grep/Glob/Read and search-hit file_path bridge",
              "order": "per question A,B on even indices; B,A on odd indices; shared generation queue",
              "budget_note": "Same model-turn and timeout caps. Tool output defaults/units are treatment; actual tokens are reported, not matched."}
    if (output / "config.json").exists():
        if not args.resume or json.loads((output / "config.json").read_text()) != config:
            raise ValueError("existing output requires --resume with identical configuration")
    else:
        save(output / "config.json", config)
    aliases = json.loads((benchmark.BASE / "enterprise/duplicate-id-aliases.json").read_text())
    reports = {group: {case["id"]: json.loads(path.read_text()) for case in cases
                      if (path := output / group / "traces" / f"{case['id']}.json").exists()} for group in ("A", "B")}
    generation, judging = asyncio.Semaphore(args.concurrency), asyncio.Semaphore(1)
    stopped = asyncio.Event()

    def snapshot(group):
        result = summary(config, cases, reports[group], aliases)
        save(output / group / "summary.json", result)
        return result

    async with AsyncCodex(rag.rag_codex_config("/private/tmp")) as client:
        account = (await client.account(refresh_token=False)).account
        if getattr(getattr(account, "root", account), "type", None) != "chatgpt":
            raise RuntimeError("requires existing ChatGPT login")

        async def one(group, case):
            qid = case["id"]
            path = output / group / "traces" / f"{qid}.json"
            report = reports[group].get(qid)
            async with generation:
                if stopped.is_set():
                    return
                if report is None:
                    report = {"id": qid, "question": case["question"], "native_answer": True, "events": [], "model_runs": []}
                    reports[group][qid] = report
                    files = CorpusFiles(args.files) if group == "B" else None
                    started = time.monotonic()
                    def emit(event):
                        if files and event["role"] == "tool":
                            event["observed_document_ids"] = files.evidence_ids(event["name"], event["content"])
                        report["events"].append(event)
                        save(path, report)
                    save(path, report)
                    try:
                        with tempfile.TemporaryDirectory(prefix="code-tools-dev-") as cwd:
                            with benchmark.knowledge_base("enterprise", case, cwd) as kb:
                                model = rag.CodexModel(client, cwd, args.model, cfg["timeout"], effort=cfg["effort"],
                                                       system_prompt=cfg["prompt"], max_calls=cfg["max_turns"])
                                report["model_runs"] = model.runs
                                tools = files.tools(kb) if files else indexed_tools(kb)
                                def timed(name, tool):
                                    async def execute(**arguments):
                                        start = time.monotonic()
                                        try:
                                            value = tool.execute(**arguments)
                                            return await value if inspect.isawaitable(value) else value
                                        finally:
                                            report.setdefault("tool_timings", []).append({"name": name, "seconds": time.monotonic()-start})
                                    return Tool(tool.description, tool.arguments, execute)
                                tools = {name: timed(name, tool) for name, tool in tools.items()}
                                report["result"] = await asyncio.wait_for(
                                    run_agent(model, tools, template + case["question"], cfg["max_turns"], emit), cfg["case_timeout"])
                    except Exception as error:
                        report["result"] = {"status": "runtime_error", "answer": "", "sources": [],
                                            "error": type(error).__name__, "detail": str(error)}
                        if benchmark.fatal(error):
                            stopped.set()
                    finally:
                        report["seconds"] = time.monotonic() - started
                        if files:
                            files.close()
                        save(path, report)
                elif "result" not in report:
                    report["result"] = {"status": "interrupted", "answer": "", "sources": []}
                    save(path, report)
                print(json.dumps({"stage": "generate", "group": group, "id": qid, "status": report["result"]["status"]}), flush=True)
                snapshot(group)
            if "judgement" not in report and not stopped.is_set():
                async with judging:
                    if stopped.is_set():
                        return
                    if report["result"]["status"] != "finished":
                        report["judgement"] = {"correct": False, "facts_supported": [False]*len(case["facts"]),
                                               "reason": "Generation failed; counted as incorrect."}
                    else:
                        payload = {"question": case["question"], "reference_answer": case["gold"],
                                   "reference_facts": case["facts"], "candidate_answer": report["result"]["answer"]}
                        report["judge_input"] = payload
                        try:
                            with tempfile.TemporaryDirectory(prefix="code-tools-judge-") as cwd:
                                report["judgement"], _ = await assess(
                                    lambda: rag.CodexModel(client, cwd, args.model, cfg["timeout"], effort="medium", system_prompt=PROMPT),
                                    payload, len(case["facts"]), report.setdefault("judge_attempts", []), lambda: save(path, report))
                        except Exception as error:
                            report["judge_error"] = {"type": type(error).__name__, "detail": str(error)}
                            if benchmark.fatal(error):
                                stopped.set()
                    save(path, report)
                    print(json.dumps({"stage": "judge", "group": group, "id": qid, "correct": report.get("judgement", {}).get("correct")}), flush=True)
                    snapshot(group)

        await asyncio.gather(*(one(group, case) for i, case in enumerate(cases)
                               for group in (("A", "B") if i % 2 == 0 else ("B", "A"))))
    results = {group: snapshot(group) for group in reports}
    print(json.dumps({group: {key: r[key] for key in ("completed", "complete", "correct", "requests")}
                      for group, r in results.items()}), flush=True)
    return 0 if all(r["complete"] for r in results.values()) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--files", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.live or args.concurrency < 1:
        parser.error("--live and a positive concurrency are required")
    raise SystemExit(asyncio.run(main(args)))
