"""Bounded live diagnostics for the incident handoff task; no model judge or retries.

These are known diagnostic cases, not a benchmark accuracy estimate.
A single shared budget covers every question; no per-batch approval is required.
"""
import argparse
import asyncio
import json
import hashlib
import shutil
from pathlib import Path
from types import SimpleNamespace

from minimal_rag_experiments.evidence import CallBudget
from minimal_rag_experiments.applications.investigation_cli import live
from minimal_rag_experiments.applications.investigation_demo import QUESTION
from minimal_rag_experiments.applications.investigation_report import save_checkpoint, save_report


CASES = [
    {"id": "normal", "question": QUESTION,
     "review_focus": "Preserve customer-specific 1.21.1, pool canary and rollback conditions; distinguish historical mitigation from current health."},
    {"id": "scope", "question": "某 H100 客户池在一次版本升级后出现 p99 上升。目前没有其 request IDs、kernel 选择轨迹或配置记录。能否直接套用 Northwind L40S 的 v2 关闭和 LONG p99 回退超过 10% 的条件？请为值班工程师整理调查交接，区分可借鉴的核查步骤与尚未证实适用于 H100 的原因和处置条件。",
     "review_focus": "Do not transfer a customer/SKU-specific root cause or threshold to H100 without evidence; collect missing facts."},
    {"id": "current_state", "question": "请确认 Northwind Analytics 的 nw-analytics-ded-01 池现在是否已完成 Runtime 1.21.1 修复并可以重新开启 kernel auto-selection v2。现有输入只有历史知识库，没有今天的监控、部署记录或 gate/config 快照。请生成调查交接，明确文档能够确认什么，以及哪些现场信息取得前不能作出恢复确认。",
     "review_focus": "Do not claim current deployment, health or approval from historical documentation; list concrete missing checks."},
]


async def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    identity = {}
    for path in (list(Path("src/minimal_rag").rglob("*.py"))
                 + list(Path("experiments/src/minimal_rag_experiments").rglob("*.py"))
                 + list(Path("src/minimal_rag/fixtures").glob("*"))
                 + [Path(__file__), Path("pyproject.toml"), Path("experiments/pyproject.toml")]):
        if not path.is_file():
            continue
        relative = path.resolve().relative_to(Path.cwd())
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        identity[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (snapshot / "manifest.json").write_text(json.dumps(identity, indent=2) + "\n")
    budget = CallBudget(args.total_calls)
    selected = CASES[:args.cases]
    plan = {"kind": "known_diagnostic_cases", "model": "gpt-6.1-sol", "cases": selected,
            "per_case_calls": args.per_case_calls, "total_calls": args.total_calls,
            "retrieval": args.retrieval, "style": args.style,
            "scope_index": str(args.scope_index) if args.scope_index else None,
            "judge_calls": 0, "automatic_retries": 0,
            "interpretation": "Structure/provenance checks and qualitative source review; not a benchmark accuracy score."}
    (output / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    summaries = []
    for case in selected:
        if budget.used >= budget.limit:
            break
        directory = output / case["id"]
        directory.mkdir()
        parameters = SimpleNamespace(index=args.index, retrieval=args.retrieval, models=None, dense_index=None,
                                     model="gpt-6.1-sol", effort="medium", max_calls=args.per_case_calls,
                                     timeout=args.timeout, question=case["question"], style=args.style,
                                     scope_index=args.scope_index)
        print(json.dumps({"event": "starting", "case": case["id"], "budget": budget.snapshot()}), flush=True)
        checkpoint = lambda state: save_checkpoint(state, directory)
        try:
            report = await live(parameters, budget=budget, checkpoint=checkpoint)
        except Exception as error:
            failure = {"case": case["id"], "status": "setup_failed", "error": type(error).__name__,
                       "detail": str(error), "budget": budget.snapshot()}
            checkpoint(failure)
            summaries.append(failure)
            break  # No silent restart after setup/authentication failures.
        save_report(report, directory, prepared=True)
        summaries.append({"case": case["id"], "status": report["status"],
                          "result_status": report["result"]["status"],
                          "checks_passed": report["validation"]["checks_passed"],
                          "validation_errors": report["validation"]["errors"],
                          "calls": report["outer_calls_this_run"], "seconds": report["seconds"],
                          "token_usage": report["token_usage"]})
        summary = {"plan": plan, "results": summaries, "budget": budget.snapshot()}
        save_checkpoint(summary, output)
        print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)
    summary = {"plan": plan, "results": summaries, "budget": budget.snapshot()}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--total-calls", required=True, type=int)
    parser.add_argument("--per-case-calls", required=True, type=int)
    parser.add_argument("--cases", type=int, choices=[1, 3], default=3)
    parser.add_argument("--retrieval", choices=["bm25", "hybrid"], default="bm25")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--style", choices=["standard", "compact"], default="standard")
    parser.add_argument("--scope-index", type=Path)
    args = parser.parse_args()
    if min(args.total_calls, args.per_case_calls, args.timeout) <= 0:
        parser.error("budgets must be positive")
    asyncio.run(run(args))
