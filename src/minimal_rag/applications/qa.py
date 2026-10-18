"""Run the minimal Agent against a prepared, read-authorized SQLite corpus.

minimal-rag ask --live --index PATH --question QUESTION
Choose Codex subscription or a direct Responses API backend explicitly.
"""
import argparse
import asyncio
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from minimal_rag.paths import ROOT

from minimal_rag.codex_adapter import SYSTEM_PROMPT
from minimal_rag.model_backend import model_session, TokenBudgetExceeded
from minimal_rag.loop import run_agent
from minimal_rag.indexed_rag import IndexedKnowledgeBase, indexed_tools

async def main(args):
    if args.backend == "api":
        args.base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    output = Path(args.output) if args.output else ROOT / "runs/minimal-rag" / f"{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}.json"
    kb = IndexedKnowledgeBase(args.index, args.context_chars)
    tools = indexed_tools(kb)
    report = {"question": args.question, "index": str(Path(args.index).resolve()), "model": args.model,
              "reasoning_effort": args.effort, "verbosity": args.verbosity,
              "max_turns": args.max_turns, "context_chars": args.context_chars,
              "retrieval": "bm25", "backend": args.backend, "base_url": args.base_url,
              "case_timeout": args.case_timeout, "max_total_tokens": args.max_total_tokens,
              "system_prompt": SYSTEM_PROMPT,
              "events": [], "model_runs": [], "note": "finished means the model stopped; it is not an automatic correctness certificate."}

    def save_report():
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(output)

    def emit(event):
        report["events"].append(event)
        save_report()
        if event["role"] == "assistant" and event["tool_calls"]:
            print(json.dumps({"tools": event["tool_calls"]}, ensure_ascii=False), flush=True)

    try:
        with tempfile.TemporaryDirectory(prefix="minimal-knowledge-agent-") as cwd:
            async with model_session(args.backend, args.model, args.timeout, args.base_url,
                                     args.max_total_tokens, args.verbosity) as factory:
                model = factory(cwd, effort=args.effort)
                report["model_runs"] = model.runs
                report["result"] = await asyncio.wait_for(
                    run_agent(model, tools, args.question, args.max_turns, emit), args.case_timeout)
    except TokenBudgetExceeded as error:
        report["result"] = {"status": "budget_exhausted", "answer": "", "sources": [], "detail": str(error)}
    except TimeoutError:
        report["result"] = {"status": "timeout", "answer": "", "sources": []}
    except Exception as error:
        report["result"] = {"status": "runtime_error", "error": type(error).__name__, "detail": str(error)}
        raise
    finally:
        kb.close()
        save_report()
    print(json.dumps(report["result"], ensure_ascii=False, indent=2))
    print(f"Trace: {output}")


def cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live", action="store_true", help="Allow model calls consuming subscription allowance or API credits")
    p.add_argument("--index", required=True)
    p.add_argument("--retrieval", choices=["bm25"], default="bm25", help="The single supported retrieval implementation")
    p.add_argument("--question", required=True)
    p.add_argument("--backend", choices=["codex", "api"], default="codex")
    p.add_argument("--model", help="Required for API; must be available at your endpoint")
    p.add_argument("--base-url", help="Responses API base URL; defaults to OPENAI_BASE_URL or OpenAI")
    p.add_argument("--effort", choices=["none", "low", "medium", "high", "xhigh", "max"], default="high")
    p.add_argument("--verbosity", choices=["low", "medium", "high"], help="API only; default low, recorded explicitly")
    p.add_argument("--context-chars", type=int, default=4000)
    p.add_argument("--max-turns", type=int, default=0, help="0: model decides when to finish; otherwise a hard stop, no forced answer")
    p.add_argument("--timeout", type=float, default=300, help="Per-request timeout in seconds")
    p.add_argument("--case-timeout", type=float, default=3600, help="Per-question timeout; 0 disables it")
    p.add_argument("--max-total-tokens", type=int, default=0, help="Optional input+output limit per question, including cache; 0 disables it")
    p.add_argument("--output")
    args = p.parse_args(argv)
    if not args.live:
        p.error("live inference consumes allowance or API credits; pass --live")
    if not 1 <= args.context_chars <= 16000 or args.timeout <= 0 or min(args.max_turns, args.case_timeout, args.max_total_tokens) < 0:
        p.error("context-chars must be 1..16000; timeout positive; optional limits nonnegative")
    if args.backend == "api" and not args.model:
        p.error("--backend api requires an explicit --model")
    if args.backend == "codex" and args.base_url:
        p.error("--base-url is only for --backend api")
    if args.backend == "codex" and args.verbosity:
        p.error("--verbosity is only for --backend api")
    if args.backend == "api":
        args.verbosity = args.verbosity or "low"
    args.model = args.model or "gpt-6.1-sol"
    args.max_turns, args.case_timeout = args.max_turns or None, args.case_timeout or None
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
