"""Run the minimal Agent against a prepared, read-authorized SQLite corpus.

minimal-rag ask --live --index PATH --question QUESTION
Uses the existing ChatGPT subscription through the pinned Codex SDK; no API-key fallback.
"""
import argparse
import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from minimal_rag.paths import ROOT

from minimal_rag.codex_adapter import CodexModel, SYSTEM_PROMPT, rag_codex_config
from minimal_rag.loop import run_agent
from minimal_rag.indexed_rag import IndexedKnowledgeBase, indexed_tools

async def main(args):
    from openai_codex import AsyncCodex
    output = Path(args.output) if args.output else ROOT / "runs/minimal-rag" / f"{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}.json"
    kb = IndexedKnowledgeBase(args.index, args.context_chars)
    tools = indexed_tools(kb)
    report = {"question": args.question, "index": str(Path(args.index).resolve()), "model": args.model,
              "reasoning_effort": args.effort, "max_turns": args.max_turns, "context_chars": args.context_chars,
              "retrieval": "bm25",
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
            async with AsyncCodex(rag_codex_config(cwd)) as client:
                wrapper = await client.account(refresh_token=False)
                if getattr(getattr(wrapper.account, "root", wrapper.account), "type", None) != "chatgpt":
                    raise RuntimeError("requires existing ChatGPT login; no API-key fallback")
                model = CodexModel(client, cwd, args.model, args.timeout, effort=args.effort,
                                   system_prompt=SYSTEM_PROMPT, max_calls=args.max_turns)
                report["model_runs"] = model.runs
                report["result"] = await run_agent(model, tools, args.question, args.max_turns, emit)
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
    p.add_argument("--live", action="store_true", help="Explicitly allow a model call that consumes allowance")
    p.add_argument("--index", required=True)
    p.add_argument("--retrieval", choices=["bm25"], default="bm25", help="The single supported retrieval implementation")
    p.add_argument("--question", required=True)
    p.add_argument("--model", default="gpt-6.1-sol")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default="high")
    p.add_argument("--context-chars", type=int, default=4000)
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--output")
    args = p.parse_args(argv)
    if not args.live:
        p.error("live inference consumes allowance; pass --live and set the call cap with --max-turns")
    if not 1 <= args.context_chars <= 16000 or min(args.max_turns, args.timeout) <= 0:
        p.error("context-chars must be 1..16000; budgets must be positive")
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
