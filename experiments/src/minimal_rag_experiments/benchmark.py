"""Current benchmark entry: pure BM25 search/read with model-directed tools.

Historical retrieval variants and frozen runners remain under experiments. This
entry never restores snapshots or loads embedding models locally.
"""
import argparse
import ast
import asyncio
from collections import Counter
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import tempfile
import time

from minimal_rag import codex_adapter as rag
from minimal_rag.indexed_rag import IndexedKnowledgeBase, indexed_tools
from minimal_rag.loop import run_agent
from minimal_rag.paths import ROOT
from minimal_rag_experiments.benchmark_metrics import (
    BASE, EXTRACTIVE_SHORT, NATIVE_FORMAT, aggregate, normalized_case, score,
)
from minimal_rag_experiments.hotpot_benchmark import load_scorer, save, sha256
from minimal_rag_experiments.rag_judge import PROMPT as JUDGE_PROMPT, assess

DATASETS = ("enterprise", "financebench", "hotpotqa", "browsecomp")


def settings(dataset):
    # Preserve each benchmark's existing observation and call budgets.
    browse = dataset == "browsecomp"
    return {"excerpt_chars": 2000 if browse else 4000, "max_turns": 30 if browse else 12,
            "case_timeout": 1800 if browse else 900, "timeout": 180, "effort": "high",
            "prompt": rag.SYSTEM_PROMPT}


def load_cases(dataset, split):
    if dataset != "browsecomp":
        path = BASE / dataset / f"{split}.json"
        return path, [normalized_case(dataset, row) for row in json.loads(path.read_text())]
    path = BASE / "browsecomp-plus/queries.jsonl"
    rows = {str(row["query_id"]): row for row in map(json.loads, path.read_text().splitlines())}
    cohorts = ("dev100", "remaining730") if split == "all" else ("dev100" if split == "dev" else "remaining730",)
    ids = [qid for name in cohorts for qid in json.loads(
        (ROOT / "experiments/browsecomp-full-20261015" / f"sample-{name}.json").read_text())["query_ids"]]
    return path, [{"id": str(qid), "question": rows[str(qid)]["query"], "category": "browsecomp",
                   "gold": rows[str(qid)]["answer"], "gold_ids": []} for qid in ids]


def format_request(dataset):
    if dataset == "browsecomp":
        path = BASE / "browsecomp-plus/official/search_agent/prompts.py"
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "QUERY_TEMPLATE" for t in node.targets):
                value = node.value.func.value if isinstance(node.value, ast.Call) else node.value
                return ast.literal_eval(value).strip().replace("get_document", "read")
        raise ValueError("official QUERY_TEMPLATE is missing")
    return EXTRACTIVE_SHORT if dataset == "hotpotqa" else NATIVE_FORMAT


@contextmanager
def knowledge_base(dataset, case, directory):
    if dataset == "hotpotqa":
        path = Path(directory) / "corpus.sqlite"
        IndexedKnowledgeBase.build(path, [
            {"id": f"D{i}", "title": title, "text": "".join(parts), "source": f"hotpotqa://{case['id']}/D{i}"}
            for i, (title, parts) in enumerate(case["raw"]["context"])])
    else:
        path = BASE / ("browsecomp-plus" if dataset == "browsecomp" else dataset) / "corpus.sqlite"
    kb = IndexedKnowledgeBase(path, settings(dataset)["excerpt_chars"])
    try:
        yield kb
    finally:
        kb.close()


def official_export(report):
    """BrowseComp's official retrieval metric counts search hits, not reads."""
    ids, counts = set(), Counter()
    for event in report.get("events", []):
        if event.get("role") == "tool":
            counts[event["name"]] += 1
            content = event["content"]
            if event["name"] == "search" and isinstance(content, dict):
                ids.update(str(hit["document_id"]) for hit in content.get("hits", []))
    result = report.get("result", {})
    return {"query_id": report["id"], "tool_call_counts": dict(counts), "retrieved_docids": sorted(ids),
            "status": "completed" if result.get("status") == "finished" else result.get("status", "interrupted"),
            "result": [{"type": "output_text", "output": result.get("answer", "")}]}


def summarize(config, cases, reports):
    dataset = config["dataset"]
    scorer = load_scorer() if dataset == "hotpotqa" else None
    rows = [score(case, reports[case["id"]], dataset, scorer)
            for case in cases if "result" in reports.get(case["id"], {})]
    metrics = aggregate(rows)
    graded = metrics["graded"]
    correct = sum(row.get("correct") or 0 for row in rows)
    requests = Counter()
    for report in reports.values():
        requests["generation"] += len(report.get("model_runs", []))
        requests["judge"] += sum(len(attempt.get("runs", [])) for attempt in report.get("judge_attempts", []))
    return {"config": config, "requested": len(cases), "completed": len(rows),
            "complete": len(rows) == len(cases) and (dataset == "browsecomp" or graded == len(cases)),
            "metrics": metrics, "correct_over_requested": correct / len(cases) if dataset in ("enterprise", "financebench") else None,
            "ungraded": len(cases) - graded, "requests": dict(requests),
            "by_category": {key: aggregate([r for r in rows if r["category"] == key]) for key in sorted({r['category'] for r in rows})},
            "cases": rows,
            "limitations": ["Single attempt; failures stay in the requested denominator.",
                            "Enterprise/Finance use the unchanged local semantic judge, not official leaderboard scores.",
                            "BrowseComp produces official-format exports only; accuracy requires the separate official judge.",
                            "Application requests include failed attempts; SDK-internal retries are not observable."]}


def fatal(error):
    message = str(error).lower()
    return any(term in message for term in ("usage limit", "usage_limit", "quota exceeded", "insufficient credits", "unauthorized", "forbidden"))


async def main(args):
    from openai_codex import AsyncCodex
    dataset, output = args.dataset, Path(args.output)
    source, cases = load_cases(dataset, args.split)
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        raise ValueError("empty cohort")
    cfg, template = settings(dataset), format_request(dataset)
    config = {"implementation": "bm25-agent", "dataset": dataset, "split": args.split,
              "evaluation_kind": "development" if args.split == "dev" else "regression",
              "ids": [c["id"] for c in cases], "sample_sha256": sha256(source),
              "retrieval": "bm25", "settings": cfg, "model": args.model, "judge_model": args.model,
              "judge_effort": "medium", "judge_prompt": JUDGE_PROMPT if dataset in ("enterprise", "financebench") else None,
              "format_request": template, "concurrency": args.concurrency, "judge_concurrency": args.judge_concurrency,
              "source_sha256": {Path(p).name: sha256(p) for p in (
                  __file__, rag.__file__, inspect.getfile(IndexedKnowledgeBase), inspect.getfile(run_agent),
                  inspect.getfile(score), inspect.getfile(assess))}}
    if dataset != "hotpotqa":
        corpus = BASE / ("browsecomp-plus" if dataset == "browsecomp" else dataset)
        config["corpus"] = str(corpus / "corpus.sqlite")
        config["index"] = json.loads((corpus / "index.json").read_text())
    config_path = output / "config.json"
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != config:
            raise ValueError("existing run differs; use a new directory or --resume with the same configuration")
    else:
        save(config_path, config)
    reports = {c["id"]: json.loads(p.read_text()) for c in cases if (p := output / "traces" / f"{c['id']}.json").exists()}
    generation_gate, judge_gate = asyncio.Semaphore(args.concurrency), asyncio.Semaphore(args.judge_concurrency)
    stopped = asyncio.Event()

    def snapshot():
        result = summarize(config, cases, reports)
        save(output / "summary.json", result)
        return result

    async with AsyncCodex(rag.rag_codex_config("/private/tmp")) as client:
        account = (await client.account(refresh_token=False)).account
        if getattr(getattr(account, "root", account), "type", None) != "chatgpt":
            raise RuntimeError("requires existing ChatGPT login; no API-key fallback")

        async def one(case):
            qid, trace = case["id"], output / "traces" / f"{case['id']}.json"
            async with generation_gate:
                if stopped.is_set():
                    return
                report = reports.get(qid)
                if report is None:
                    report = {"id": qid, "question": case["question"], "native_answer": True, "events": [], "model_runs": []}
                    reports[qid] = report
                    started = time.monotonic()
                    def emit(event):
                        report["events"].append(event)
                        save(trace, report)
                    save(trace, report)
                    try:
                        with tempfile.TemporaryDirectory(prefix="bm25-agent-") as cwd:
                            with knowledge_base(dataset, case, cwd) as kb:
                                model = rag.CodexModel(client, cwd, args.model, cfg["timeout"], effort=cfg["effort"],
                                                       system_prompt=cfg["prompt"], max_calls=cfg["max_turns"])
                                report["model_runs"] = model.runs
                                question = template.format(Question=case["question"]) if dataset == "browsecomp" else template + case["question"]
                                tools = indexed_tools(kb)
                                report["result"] = await asyncio.wait_for(
                                    run_agent(model, tools, question, cfg["max_turns"], emit), cfg["case_timeout"])
                    except Exception as error:
                        report["result"] = {"status": "runtime_error", "answer": "", "sources": [], "error": type(error).__name__, "detail": str(error)}
                        if fatal(error):
                            stopped.set()
                    finally:
                        report["seconds"] = time.monotonic() - started
                        save(trace, report)
                elif "result" not in report:
                    # Do not resample a question after a process interruption.
                    report["result"] = {"status": "interrupted", "answer": "", "sources": []}
                    save(trace, report)
                snapshot()
                print(json.dumps({"stage": "generate", "id": qid, "status": report["result"]["status"]}), flush=True)
            if dataset == "browsecomp":
                save(output / "official" / f"{qid}.json", official_export(report))
            elif dataset != "hotpotqa" and "judgement" not in report and not stopped.is_set():
                async with judge_gate:
                    if stopped.is_set():
                        return
                    if report["result"]["status"] != "finished":
                        report["judgement"] = {"correct": False, "facts_supported": [False] * len(case["facts"]), "reason": "Generation failed; counted as incorrect."}
                    else:
                        payload = {"question": case["question"], "reference_answer": case["gold"],
                                   "reference_facts": case["facts"], "candidate_answer": report["result"]["answer"]}
                        report["judge_input"] = payload
                        try:
                            with tempfile.TemporaryDirectory(prefix="bm25-judge-") as cwd:
                                report["judgement"], report["judge_runs"] = await assess(
                                    lambda: rag.CodexModel(client, cwd, args.model, cfg["timeout"], effort="medium", system_prompt=JUDGE_PROMPT),
                                    payload, len(case["facts"]), report.setdefault("judge_attempts", []), lambda: save(trace, report))
                        except Exception as error:
                            report["judge_error"] = {"type": type(error).__name__, "detail": str(error)}
                            if fatal(error):
                                stopped.set()
                    save(trace, report)
                    print(json.dumps({"stage": "judge", "id": qid, "correct": report.get("judgement", {}).get("correct")}), flush=True)
            snapshot()

        await asyncio.gather(*(one(case) for case in cases))
    result = snapshot()
    print(json.dumps({k: result[k] for k in ("requested", "completed", "complete", "metrics", "requests")}, indent=2), flush=True)
    return 0 if result["complete"] else 1


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--output", required=True)
    parser.add_argument("--live", action="store_true", help="Allow model calls consuming subscription allowance")
    parser.add_argument("--limit", type=int, help="Development smoke test only")
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--judge-concurrency", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not args.live:
        parser.error("pass --live to allow model calls")
    if min(args.concurrency, args.judge_concurrency) <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("budgets must be positive")
    if args.split == "all" and args.dataset != "browsecomp":
        parser.error("--split all is only for BrowseComp's complete 830-question cohort")
    if args.split in ("test", "all") and args.limit:
        parser.error("test uses the complete cohort (no --limit)")
    return asyncio.run(main(args))


if __name__ == "__main__":
    raise SystemExit(cli())
