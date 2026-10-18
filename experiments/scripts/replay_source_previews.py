"""Replay the first unscoped search from every saved answer; no model calls.

Verify original passages/pagination are unchanged and record extra title pointers.
Gold labels and expected document IDs do not participate in retrieval.
"""
import argparse
import asyncio
import hashlib
import inspect
import json
from pathlib import Path
import time

from minimal_rag.indexed_rag import IndexedKnowledgeBase, indexed_tools
from minimal_rag_experiments.source_previews import source_preview_tools


async def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    checks = []
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    manifest = {}
    for path in (Path(__file__), Path(inspect.getfile(IndexedKnowledgeBase)), Path(inspect.getfile(source_preview_tools))):
        (snapshot / path.name).write_bytes(path.read_bytes())
        manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (snapshot / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    kb = IndexedKnowledgeBase(args.index, scope_index=args.scope_index)
    search = source_preview_tools(indexed_tools(kb), args.count)["search"]
    try:
        for path in sorted(args.traces.glob("*/*/report.json")):
            for event in json.loads(path.read_text())["events"]:
                if event.get("role") != "tool" or event.get("name") != "search":
                    continue
                arguments = json.loads(event["arguments_json"])
                if arguments.get("scope") or not arguments.get("query", "").strip():
                    continue
                expected = {**event["content"], "hits": [
                    {key: value for key, value in row.items() if key not in {"evidence_id", "passage_sha256"}}
                    for row in event["content"]["hits"]]}
                start = time.monotonic()
                observed = await search.execute(**arguments)
                extras = observed.pop("other_matches", [])
                checks.append({"source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "arguments": arguments, "seconds": time.monotonic() - start,
                               "original_observation_unchanged": observed == expected,
                               "passage_document_ids": [row["document_id"] for row in observed["hits"]],
                               "unread_source_pointers": extras})
                print(json.dumps({"completed": len(checks), "source": str(path),
                                  "unchanged": checks[-1]["original_observation_unchanged"],
                                  "pointers": len(extras)}), flush=True)
                break
    finally:
        kb.close()
    result = {"model_calls": 0, "checks": checks, "preview_count": args.count,
              "all_original_observations_unchanged": bool(checks) and all(row["original_observation_unchanged"] for row in checks),
              "note": "Unread title discoverability, not observed evidence, model selection or answer-quality validation."}
    (args.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--scope-index", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--count", type=int, choices=range(16), default=15)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
