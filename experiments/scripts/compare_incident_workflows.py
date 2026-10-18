"""Fixed known-dev comparison: ordinary QA vs evidence-linked incident handoff.

One generation and at most one local model judgment per arm/question. No retries,
no resume, no best-of-N. The shared cap includes generation and judging.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile
import time

from minimal_rag.codex_adapter import CodexModel, SYSTEM_PROMPT, rag_codex_config
from minimal_rag.loop import Reply, run_agent
from minimal_rag_experiments.evidence import CallBudget, CallLimitReached
from minimal_rag.indexed_rag import IndexedKnowledgeBase, indexed_tools
from minimal_rag_experiments.applications.investigation import investigate
from minimal_rag_experiments.applications.investigation_cli import usage_record
from minimal_rag_experiments.applications.investigation_report import save_checkpoint, save_report
from minimal_rag_experiments.rag_judge import Assessment, PROMPT as JUDGE_PROMPT
from minimal_rag_experiments.rag_optimization_eval import aggregate, normalized_case, score
from minimal_rag_experiments.source_previews import source_preview_tools


def scoring_report(report):
    passages = {e["id"]: e["document_id"] for e in report["evidence"]}
    return {**report, "native_answer": True, "result": {**report["result"], "sources":
        list(dict.fromkeys(passages.get(x, x) for x in report["result"].get("sources", [])))}}


def cohort_metrics(rows, expected):
    metrics = aggregate(rows)
    complete = len(rows) == expected and all(row.get("graded") for row in rows)
    if not complete:
        for key in ("correct", "completeness", "overall"):
            metrics[key] = None
    return {**metrics, "requested": expected, "quality_metrics_complete": complete}


async def plain_qa(model, kb, question, budget, max_calls, timeout, persist, *, source_previews=0):
    report = {"question": question, "events": [], "model_runs": model.runs, "native_answer": True}
    async def limited(messages, tools):
        if time.monotonic() - started >= timeout:
            raise TimeoutError("investigation deadline reached before dispatch")
        budget.reserve()
        persist({**report, "budget": budget.snapshot(), "status": "running"})
        response = await model(messages, tools)
        if time.monotonic() - started >= timeout:
            raise TimeoutError("investigation deadline reached after dispatch")
        return response
    def emit(event):
        report["events"].append(event)
        persist({**report, "budget": budget.snapshot(), "status": "running"})
    started = time.monotonic()
    try:
        async with asyncio.timeout(timeout):
            report["result"] = await run_agent(limited, source_preview_tools(indexed_tools(kb), source_previews),
                "请根据资料用中文回答问题，并在外层 sources 中给出来源文档 ID。\n" + question, max_calls, emit)
    except CallLimitReached:
        report["result"] = {"status": "budget_exhausted", "answer": "", "sources": []}
    except TimeoutError:
        report["result"] = {"status": "timeout", "answer": "", "sources": []}
    except Exception as error:
        report["result"] = {"status": "runtime_error", "answer": "", "sources": [],
                            "error": type(error).__name__, "detail": str(error)}
    report["seconds"] = time.monotonic() - started
    if report["result"]["status"] == "finished" and report["seconds"] >= timeout:
        report["result"] = {**report["result"], "status": "timeout"}
    return report


async def judge_once(client, cwd, case, answer, budget, directory):
    model = CodexModel(client, cwd, "gpt-6.1-sol", timeout=120, effort="medium", system_prompt=JUDGE_PROMPT, max_calls=1)
    payload = {"question": case["question"], "reference_answer": case["gold"],
               "reference_facts": case["facts"], "candidate_answer": answer}
    record = {"model_runs": model.runs, "status": "started"}
    try:
        budget.reserve()
        save_checkpoint({**record, "budget": budget.snapshot()}, directory)
        raw = await model([{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}], [])
        record["raw"] = raw
        reply = Reply.model_validate_json(raw)
        if reply.tool_calls:
            raise ValueError("judge attempted a tool call")
        grade = Assessment.model_validate_json(reply.text)
        if len(grade.facts_supported) != len(case["facts"]):
            raise ValueError("judge reference-fact count differs")
        record.update(status="graded", assessment=grade.model_dump())
    except Exception as error:
        record.update(status="ungraded", error=type(error).__name__, detail=str(error))
    (directory / "judgment.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return record.get("assessment")


async def run(args):
    from openai_codex import AsyncCodex
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path("data/benchmarks/enterprise/dev.json")
    cases = [normalized_case("enterprise", row) for row in json.loads(source.read_text())
             if row["question_type"] == "project_related"]
    if len(cases) != 8:
        raise ValueError("expected the frozen eight-case known development cohort")
    budget, gate = CallBudget(args.total_calls), asyncio.Semaphore(args.concurrency)
    paths = (list(Path("src/minimal_rag").rglob("*.py"))
             + list(Path("experiments/src/minimal_rag_experiments").rglob("*.py"))
             + [Path(__file__), Path("pyproject.toml"), Path("experiments/pyproject.toml")])
    identity = {}
    for path in paths:
        relative = path.resolve().relative_to(Path.cwd())
        target = args.output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        identity[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output / "source_snapshot/manifest.json").write_text(json.dumps(identity, indent=2) + "\n")
    plan = {"model": "gpt-6.1-sol", "judge_model": "gpt-6.1-sol", "effort": "medium",
            "ids": [c["id"] for c in cases], "cohort": "all eight known enterprise development project_related cases",
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "arms": args.arms,
            "per_generation_calls": args.max_calls, "judge_calls_per_answer": 1, "total_call_cap": args.total_calls,
            "retrieval": "bm25", "scope_index": str(args.scope_index), "style": args.style,
            "source_previews": args.source_previews,
            "judge_prompt": JUDGE_PROMPT, "random_order_seed": 20261012,
            "note": "Known development diagnostic. Same model/budget, single draw; same-family local judge, no human calibration, not official or production validation."}
    (args.output / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    rows = {arm: [] for arm in args.arms}
    def summarize():
        summary = {"plan": plan, "metrics": {arm: cohort_metrics(values, len(cases)) for arm, values in rows.items()},
                   "rows": rows, "budget": budget.snapshot(),
                   "complete": all(len(values) == len(cases) and all(r.get("graded") for r in values) for values in rows.values())}
        temporary = args.output / "summary.json.tmp"
        temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(args.output / "summary.json")
        return summary
    summarize()
    with tempfile.TemporaryDirectory(prefix="rag-incident-comparison-") as cwd:
        async with AsyncCodex(rag_codex_config(cwd)) as client:
            wrapper = await client.account(refresh_token=False)
            if getattr(getattr(wrapper.account, "root", wrapper.account), "type", None) != "chatgpt":
                raise RuntimeError("requires existing ChatGPT login")
            async def one(case, arm):
                async with gate:
                    directory = args.output / arm / case["id"]
                    directory.mkdir(parents=True)
                    persist = lambda state: save_checkpoint(state, directory)
                    model = CodexModel(client, cwd, "gpt-6.1-sol", timeout=args.timeout,
                                       effort="medium", system_prompt=SYSTEM_PROMPT, max_calls=args.max_calls,
                                       as_of="2026-10-12")
                    kb = IndexedKnowledgeBase(args.index, scope_index=args.scope_index)
                    print(json.dumps({"event": "starting", "arm": arm, "id": case["id"], "budget": budget.snapshot()}), flush=True)
                    try:
                        if arm == "plain":
                            report = await plain_qa(model, kb, case["question"], budget, args.max_calls, args.timeout, persist,
                                                    source_previews=args.source_previews)
                            (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                            scored = report
                        else:
                            report = await investigate(model, source_preview_tools(indexed_tools(kb), args.source_previews),
                                                       kb.read, case["question"], budget=budget,
                                                       max_turns=args.max_calls, timeout=args.timeout, style=args.style, checkpoint=persist)
                            report["model_runs"] = model.runs
                            save_report(report, directory, prepared=True)
                            # Grade answer content with document IDs, not receipt IDs. Gold
                            # never enters generation or retrieval, only this post-run step.
                            scored = scoring_report(report)
                        if scored["result"]["status"] != "finished":
                            grade = {"correct": False, "facts_supported": [False] * len(case["facts"]), "reason": "Generation did not finish."}
                        else:
                            judge_dir = directory / "judge"
                            judge_dir.mkdir()
                            grade = await judge_once(client, cwd, case, scored["result"]["answer"], budget, judge_dir)
                        if grade is not None:
                            scored["judgement"] = grade
                        row = score(case, scored, "enterprise", None)
                        row["token_usage_complete"] = usage_record(model.runs)["token_usage_complete"]
                        row["answer_chars"] = len(scored["result"].get("answer", ""))
                        row["draft_ready"] = report.get("status") == "draft_for_review" if arm == "handoff" else None
                        row["provenance_errors"] = report.get("validation", {}).get("errors", [])
                        rows[arm].append(row)
                        rows[arm].sort(key=lambda r: r["id"])
                        (directory / "scored.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
                        summarize()
                        print(json.dumps({"arm": arm, "id": case["id"], "correct": row.get("correct"),
                                          "overall": row.get("overall"), "draft_ready": row["draft_ready"],
                                          "calls": row["model_calls"], "budget_used": budget.used}, ensure_ascii=False), flush=True)
                    finally:
                        kb.close()
            jobs = [(case, arm) for case in cases for arm in args.arms]
            random.Random(20261012).shuffle(jobs)
            await asyncio.gather(*(one(case, arm) for case, arm in jobs))
    return summarize()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--scope-index", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--total-calls", type=int, required=True)
    p.add_argument("--max-calls", type=int, default=4)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--style", choices=["standard", "compact", "coverage"], default="compact")
    p.add_argument("--source-previews", type=int, choices=range(16), default=0)
    p.add_argument("--arms", choices=["plain", "handoff"], nargs="+", default=["plain", "handoff"])
    a = p.parse_args()
    if min(a.total_calls, a.max_calls, a.timeout, a.concurrency) <= 0:
        p.error("budgets must be positive")
    if len(set(a.arms)) != len(a.arms):
        p.error("each arm must appear only once")
    asyncio.run(run(a))
