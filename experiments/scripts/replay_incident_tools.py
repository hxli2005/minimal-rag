"""Check saved tool observations against current retrieval, without model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from minimal_rag.indexed_rag import IndexedKnowledgeBase


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items() if k not in {"evidence_id", "passage_sha256"}}
    if isinstance(value, list):
        return [plain(v) for v in value]
    return value


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    checks, seen = [], set()
    kb = IndexedKnowledgeBase(args.index, scope_index=args.scope_index)
    try:
        for root in args.traces:
            for path in sorted(root.glob("*/report.json")):
                report = json.loads(path.read_text())
                for event in report["events"]:
                    if event.get("role") != "tool" or event.get("name") not in {"search", "read"}:
                        continue
                    arguments = json.loads(event["arguments_json"])
                    expected = plain(event["content"])
                    key = json.dumps([event["name"], arguments, expected], ensure_ascii=False, sort_keys=True)
                    if key in seen:
                        continue
                    seen.add(key)
                    start = time.monotonic()
                    actual = getattr(kb, event["name"])(**arguments)
                    row = {"source": str(path), "tool": event["name"], "arguments": arguments,
                           "seconds": time.monotonic() - start, "identical": actual == expected,
                           "observation_sha256": hashlib.sha256(json.dumps(expected, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
                    checks.append(row)
                    if not row["identical"]:
                        (args.output / f"difference-{len(checks)}.json").write_text(json.dumps({"expected": expected, "actual": actual}, ensure_ascii=False, indent=2))
                    print(json.dumps({"checked": len(checks), "tool": row["tool"], "identical": row["identical"]}), flush=True)
    finally:
        kb.close()
    result = {"kind": "known_tool_observation_replay", "model_calls": 0, "checks": checks,
              "all_identical": bool(checks) and all(r["identical"] for r in checks),
              "candidate_sha256": hashlib.sha256(Path("src/minimal_rag/indexed_rag.py").read_bytes()).hexdigest(),
              "note": "Checks retrieval behavior only; does not re-evaluate answer quality or measure production load."}
    (args.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"checked": len(checks), "all_identical": result["all_identical"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--scope-index", type=Path, required=True)
    parser.add_argument("--traces", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
