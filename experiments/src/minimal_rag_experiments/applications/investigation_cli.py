"""Create a reviewable incident handoff; default operation requires no model."""
import argparse
import asyncio
import json
import math
import tempfile
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from minimal_rag.codex_adapter import token_usage
from minimal_rag_experiments.evidence import CallBudget
from minimal_rag_experiments.applications.investigation import investigate
from minimal_rag_experiments.applications.investigation_report import save_checkpoint, save_failure, save_report


def usage_record(runs):
    missing = [i for i, run in enumerate(runs, 1)
               if any(((run.get("usage") or {}).get("total") or {}).get(key) is None
                      for key in ("input_tokens", "cached_input_tokens", "output_tokens"))]
    return {"token_usage": token_usage(runs), "token_usage_complete": not missing,
            "attempts_without_complete_usage": missing}


async def live(args, *, budget=None, checkpoint=lambda _: None):
    started = time.monotonic()
    from openai_codex import AsyncCodex
    from minimal_rag.codex_adapter import CodexModel, rag_codex_config
    from minimal_rag.indexed_rag import IndexedKnowledgeBase, indexed_tools

    dense = None
    budget = budget if budget is not None else CallBudget(args.max_calls)
    if args.retrieval == "hybrid":
        from minimal_rag_experiments.hybrid_rag import DEFAULT_MODELS, DenseIndex, HybridKnowledgeBase, LocalModels, hybrid_tools, load_model_config
        if load_model_config(args.models or DEFAULT_MODELS).get("prefilter"):
            raise ValueError("Experimental prefilters must run through the experiments evaluation entry")
        models = LocalModels(args.models or DEFAULT_MODELS)
        dense = DenseIndex(args.dense_index or Path(args.index).with_suffix(".dense"), args.index, models)
        try:
            kb = HybridKnowledgeBase(args.index, models, dense, scope_index=getattr(args, "scope_index", None))
        except BaseException:
            dense.close()
            raise
        tools = hybrid_tools(kb)
    else:
        kb = IndexedKnowledgeBase(args.index, scope_index=getattr(args, "scope_index", None))
        tools = indexed_tools(kb)
    try:
        if getattr(args, "source_previews", 0):
            raise ValueError("Source previews must run through the experiments evaluation entry")
        with tempfile.TemporaryDirectory(prefix="rag-investigation-") as cwd:
            async with AsyncCodex(rag_codex_config(cwd)) as client:
                wrapper = await client.account(refresh_token=False)
                if getattr(getattr(wrapper.account, "root", wrapper.account), "type", None) != "chatgpt":
                    raise RuntimeError("requires existing ChatGPT login; no API-key fallback")
                model = CodexModel(client, cwd, args.model, timeout=args.timeout,
                                   as_of=datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat(),
                                   effort=args.effort, max_calls=args.max_calls)
                setup_seconds = time.monotonic() - started
                report = await investigate(model, tools, kb.read, args.question,
                                           budget=budget, max_turns=args.max_calls,
                                           timeout=args.timeout, checkpoint=checkpoint,
                                           style=getattr(args, "style", "compact"))
                report["model_runs"] = model.runs
                report.update(usage_record(model.runs))
                report["model"] = args.model
                report["generation_calls"] = report["outer_calls_this_run"]
                report["judge_calls"] = 0
                report["setup_seconds"] = setup_seconds
                report["end_to_end_seconds"] = time.monotonic() - started
                report["corpus"] = {"path": str(Path(args.index).resolve()), "retrieval": args.retrieval,
                                    "authorization": "caller must supply an already authorized corpus",
                                    "source_previews": getattr(args, "source_previews", 0),
                                    "scope_index": str(args.scope_index) if getattr(args, "scope_index", None) else None}
                return report
    finally:
        kb.close()
        if dense is not None:
            dense.close()


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="Authored replay with local tools; zero model calls")
    mode.add_argument("--live", action="store_true", help="Consumes allowance; requires explicit --max-calls")
    parser.add_argument("--output", required=True, type=Path, help="New directory for report.html and report.json")
    parser.add_argument("--index", type=Path)
    parser.add_argument("--question")
    parser.add_argument("--max-calls", type=int, help="Hard cap on attempted outer model calls; includes failed calls")
    parser.add_argument("--timeout", type=float, default=180, help="Whole investigation deadline in seconds")
    parser.add_argument("--retrieval", choices=["bm25", "hybrid"], default="bm25")
    parser.add_argument("--models", type=Path)
    parser.add_argument("--dense-index", type=Path)
    parser.add_argument("--scope-index", type=Path, help="Corpus-bound passage metadata directory; loads no vector/model weights")
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--effort", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--style", choices=["standard", "compact", "coverage"], default="compact")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output directory already exists; previous evidence is never overwritten")
    if not math.isfinite(args.timeout) or args.timeout <= 0 or (args.max_calls is not None and args.max_calls <= 0):
        parser.error("timeout and max-calls must be positive")
    if args.live and (args.max_calls is None or args.index is None or not args.question or not args.question.strip()):
        parser.error("live mode requires --index, --question and an explicit --max-calls")
    if args.demo and (args.index is not None or args.question is not None or args.models is not None
                      or args.dense_index is not None or args.scope_index is not None or args.retrieval != "bm25"
                      ):
        parser.error("demo uses its packaged fixture and authored question; custom corpora require live mode")
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = lambda state: save_checkpoint(state, args.output)
    checkpoint({"status": "starting", "mode": "demo" if args.demo else "model",
                "question": args.question, "max_calls": args.max_calls or 4,
                "budget": CallBudget(args.max_calls or 4).snapshot()})
    stage = "investigation"
    try:
        if args.demo:
            from minimal_rag_experiments.applications.investigation_demo import demo
            report = asyncio.run(demo(max_calls=args.max_calls or 4, timeout=args.timeout, checkpoint=checkpoint))
        else:
            report = asyncio.run(live(args, checkpoint=checkpoint))
        stage = "delivery"
        checkpoint({**report, "execution_status": "delivering"})
        path = save_report(report, args.output, prepared=True)
        checkpoint({**report, "execution_status": "completed", "report_file": str(path.resolve())})
    except BaseException as error:
        try:
            save_failure(error, args.output, stage=stage)
        except OSError:
            pass  # A full/unwritable disk must not mask the original exception.
        raise
    print(json.dumps({"status": report["status"], "mode": report["mode"], "report": str(path.resolve()),
                      "generation_calls": report["generation_calls"], "budget": report["budget"]}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "draft_for_review" else 2
