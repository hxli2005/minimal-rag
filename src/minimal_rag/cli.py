"""Offline first: inspect the loop, index JSONL documents, search, then opt into live QA."""
import argparse
import asyncio
import json
import sys
from pathlib import Path


async def demo():
    """Fixed model decisions over real local search/read tools; no model inference."""
    from minimal_rag.loop import run_agent
    from minimal_rag.demo import fixture_index
    from minimal_rag.indexed_rag import indexed_tools

    decisions = iter([
        {"text": "", "tool_calls": [{"name": "search", "arguments_json": json.dumps({"query": "MUG-C02 定制 退货"})}]},
        {"text": "", "tool_calls": [
            {"name": "read", "arguments_json": json.dumps({"document_id": "C"})},
            {"name": "read", "arguments_json": json.dumps({"document_id": "B"})},
        ]},
        {"text": "MUG-C02 属于定制商品[C]；无质量问题时不支持无理由退货[B]。",
         "tool_calls": [], "sources": ["B", "C"]},
    ])

    async def replay(messages, tools):
        return json.dumps(next(decisions), ensure_ascii=False)

    events = []
    with fixture_index() as kb:
        result = await run_agent(replay, indexed_tools(kb),
                                 "MUG-C02 签收3天，没有质量问题，可以无理由退货吗？", emit=events.append)
    return {"mode": "offline_scripted_demo", "note": "固定决策回放，使用真实本地检索；不调用模型，不代表模型准确率。",
            "result": result, "events": events}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "ask":
        from minimal_rag.applications.qa import cli
        return cli(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("demo", help="Run a deterministic offline example; consumes no model allowance")
    index = commands.add_parser("index", help="Index JSONL documents with id/title/text/source fields")
    index.add_argument("documents", type=Path)
    index.add_argument("--output", required=True, type=Path)
    search = commands.add_parser("search", help="Search an existing SQLite corpus offline")
    search.add_argument("--index", required=True, type=Path)
    search.add_argument("--query", required=True)
    search.add_argument("--limit", type=int, default=5, choices=range(1, 21), metavar="1..20")
    commands.add_parser("ask", help="Answer with a live model (requires --live; consumes allowance)")
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "demo":
        result = asyncio.run(demo())
    else:
        from minimal_rag.indexed_rag import IndexedKnowledgeBase
        if args.command == "index":
            with args.documents.open(encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
            args.output.parent.mkdir(parents=True, exist_ok=True)
            result = IndexedKnowledgeBase.build(args.output, rows)
        else:
            kb = IndexedKnowledgeBase(args.index)
            try:
                result = kb.search(args.query, limit=args.limit)
            finally:
                kb.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
