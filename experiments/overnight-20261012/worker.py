"""Frozen CPU-only, resumable one-night sol experiments. No core code changes."""
import argparse
import asyncio
import fcntl
import json
import random
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from budget import BudgetModel, DIRECTORY, LEDGER, LIMIT, MODEL
from minimal_rag.codex_adapter import SYSTEM_PROMPT, rag_codex_config, token_usage
from minimal_rag.indexed_rag import IndexedKnowledgeBase, indexed_tools
from minimal_rag.loop import Tool, run_agent
from minimal_rag.paths import ROOT
from minimal_rag_experiments.hotpot_benchmark import load_scorer, save as save_raw
from minimal_rag_experiments.rag_judge import PROMPT, assess
from minimal_rag_experiments.rag_optimization_eval import EXTRACTIVE_SHORT, NATIVE_FORMAT, normalized_case

OUTPUT = ROOT / "runs/overnight-20261012"
BASE = ROOT / "data/benchmarks"
HISTORY = BASE / "optimization-20261011"
RESULTS = OUTPUT / "results"


def save(path, value):
    from minimal_rag_experiments.evidence_summary import publish
    if Path(path).parent == ROOT / "evidence":
        publish(path, value)
    else:
        save_raw(path, value)


def read(path):
    return json.loads(Path(path).read_text())


def used():
    return len(LEDGER.read_text().splitlines()) if LEDGER.exists() else 0


def state(**changes):
    current = read(DIRECTORY / "state.json")
    current.update(changes, updated_unix=time.time(), calls_reserved=used())
    save(DIRECTORY / "state.json", current)


def stratified(cases, count):
    # Selection does not inspect answers, gold IDs or previous correctness.
    rng = random.Random(20261012)
    groups = defaultdict(list)
    for case in cases:
        groups[case["category"]].append(case)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < count:
        for category in sorted(groups):
            if groups[category] and len(selected) < count:
                selected.append(groups[category].pop())
    return selected


def prepare():
    path = DIRECTORY / "cohort.json"
    if path.exists():
        return read(path)
    cohort = {"model": MODEL, "selection": "seed 20261012, round-robin category stratification; dev only",
              "scoring": "local blind sol judge; Hotpot official answer EM/F1, distractor scope",
              "retrieval": "SQLite BM25, title weight 2, 4000-char context; no local ML models",
              "datasets": {}, "oracle": []}
    for dataset, count in [("enterprise", 24), ("financebench", 12), ("hotpotqa", 12)]:
        cases = [normalized_case(dataset, row) for row in read(BASE / dataset / "dev.json")]
        cohort["datasets"][dataset] = stratified(cases, count)
    previous = read(HISTORY / "runs/enterprise-dev100-hybrid-minilm-wide/summary.json")
    wrong = {row["id"] for row in previous["cases"] if not row["correct"]}
    cases = [normalized_case("enterprise", row) for row in read(BASE / "enterprise/dev.json")]
    cohort["oracle"] = stratified([c for c in cases if c["id"] in wrong and c["gold_ids"]], 12)
    save(path, cohort)
    state(status="prepared")
    return cohort


def failure_analysis():
    aliases = read(BASE / "enterprise/duplicate-id-aliases.json")
    outputs = {}
    for name, folder, split, ref in [
        ("dev_wrong", HISTORY / "runs/enterprise-dev100-hybrid-minilm-wide", "dev", None),
        ("cdl_losses", HISTORY / "holdout/enterprise", "test", HISTORY / "holdout/public-comparisons/cdl/summary.json")]:
        previous = read(folder / "summary.json")
        references = {c["id"]: c for c in read(ref)["cases"]} if ref else {}
        cases = {c["question_id"]: c for c in read(BASE / f"enterprise/{split}.json")}
        rows = []
        for result in previous["cases"]:
            if result["correct"] or (ref and not references[result["id"]]["correct"]):
                continue
            trace = read(folder / "traces" / f"{result['id']}.json")
            case = cases[result["id"]]
            observations, searches = [], []
            for event in trace["events"]:
                if event["role"] != "tool":
                    continue
                content = event["content"]
                if event["name"] == "search":
                    searches.append(json.loads(event["arguments_json"]))
                for hit in content.get("hits", [content]):
                    if hit.get("document_id") and hit.get("text"):
                        observations.append({"document_id": aliases.get(hit["document_id"], hit["document_id"]),
                            "offset": hit.get("offset", 0), "chars": len(hit["text"]), "title": hit.get("title", "")})
            gold = {aliases.get(x, x) for x in case["expected_doc_ids"]}
            seen = {r["document_id"] for r in observations}
            bucket = "no_reference_ids" if not gold else (
                "reference_documents_missing_or_partial" if not gold <= seen else "all_reference_documents_seen_answer_failed")
            rows.append({"id": result["id"], "category": result["category"], "question": case["question"],
                "bucket": bucket, "missing_gold_ids": sorted(gold - seen), "gold_ids": sorted(gold),
                "reason": trace.get("judgement", {}).get("reason"), "answer": result["prediction"],
                "gold_answer": case["gold_answer"], "reference_facts": case["answer_facts"],
                "searches": searches, "observations": observations, "model_calls": result["model_calls"],
                "trace": str((folder / "traces" / f"{result['id']}.json").relative_to(ROOT))})
        outputs[name] = {"n": len(rows), "buckets": dict(Counter(r["bucket"] for r in rows)),
                         "categories": dict(Counter(r["category"] for r in rows)), "cases": rows}
    outputs["limitations"] = ["Document identity exposure does not prove key facts were exposed or understood.",
        "Missing labeled documents does not prove no alternate evidence exists. Buckets are diagnostic signals, not causal judgments.",
        "Historical generation astra, current online tests sol. Known regression sets; no untouched holdout claims."]
    save(OUTPUT / "failure-analysis.json", outputs)
    return outputs


def summarize():
    cohort = prepare()
    summary = {"model": MODEL, "calls_reserved": used(), "call_limit": LIMIT, "groups": {},
               "limitations": ["Stratified small dev samples, one generation per condition; not official scores or SOTA.",
                   "Agent gets more model calls than one-shot baselines; this evaluates the resource/quality tradeoff.",
                   "Historical best hybrid configuration is not rerun; tonight uses CPU BM25."]}
    for dataset, modes in [("enterprise", ["closed", "once", "agent", "snippet"]),
                           ("financebench", ["agent"]), ("hotpotqa", ["agent"]),
                           ("oracle", ["observed", "gold"])]:
        cases = cohort["oracle"] if dataset == "oracle" else cohort["datasets"][dataset]
        for mode in modes:
            if mode == "snippet":
                cases = cohort["datasets"]["enterprise"][:12]
            rows = [read(RESULTS / dataset / mode / f"{c['id']}.json") for c in cases
                    if (RESULTS / dataset / mode / f"{c['id']}.json").exists()]
            complete = [r for r in rows if r.get("terminal")]
            if not rows:
                continue
            finished = [r for r in complete if r.get("result", {}).get("status") == "finished"]
            scored = [r for r in complete if "metric" in r]
            summary["groups"][f"{dataset}/{mode}"] = {
                "planned": len(cases), "terminal": len(complete), "finished": len(finished), "scored": len(scored),
                "complete": len(complete) == len(cases), "successful_mean": statistics.mean(r["metric"] for r in scored) if scored else None,
                "planned_denominator_score": sum(r.get("metric", 0) for r in rows) / len(cases),
                "score_is_final": len(scored) == len(cases),
                "f1": sum(r.get("f1", 0) for r in rows) / len(cases) if dataset == "hotpotqa" else None,
                "generation_calls": sum(len(r.get("model_runs", [])) + sum(len(a["runs"]) for a in r.get("generation_attempts", [])) for r in rows),
                "generation_tokens": dict(Counter({k: sum(token_usage(r.get("model_runs", [])).get(k, 0) for r in rows)
                    for k in ("input_tokens", "cached_input_tokens", "output_tokens")})),
                "cases": [{"id": r["id"], "metric": r.get("metric"), "status": r.get("result", {}).get("status"),
                           "error": r.get("error"), "terminal": r.get("terminal", False)} for r in rows]}
    save(OUTPUT / "summary.json", summary)
    return summary


def async_tools(kb):
    tools = indexed_tools(kb)
    for name, tool in list(tools.items()):
        async def execute(_function=tool.execute, **kwargs):
            return await asyncio.to_thread(_function, **kwargs)
        tools[name] = Tool(tool.description, tool.arguments, execute)
    return tools


async def one(client, case, dataset, mode):
    path = RESULTS / dataset / mode / f"{case['id']}.json"
    report = read(path) if path.exists() else {"id": case["id"], "category": case["category"],
        "question": case["question"], "events": [], "generation_attempts": [], "terminal": False}
    if report.get("terminal"):
        return True
    label = f"{dataset}/{mode}/{case['id']}"
    began = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="rag-night-") as cwd:
        kb = None
        try:
            if "result" not in report:
                if dataset == "hotpotqa":
                    corpus = Path(cwd) / "corpus.sqlite"
                    IndexedKnowledgeBase.build(corpus, [{"id": f"D{i}", "title": title, "text": "".join(parts),
                        "source": f"hotpotqa://{case['id']}/D{i}"} for i, (title, parts) in enumerate(case["raw"]["context"])])
                else:
                    corpus = BASE / ("enterprise" if dataset == "oracle" else dataset) / "corpus.sqlite"
                if mode != "closed":
                    kb = IndexedKnowledgeBase(corpus, excerpt_chars=1600 if mode == "snippet" else 4000, check_same_thread=False)
                request = (EXTRACTIVE_SHORT if dataset == "hotpotqa" else NATIVE_FORMAT) + case["question"]
                max_calls = (6 if dataset == "enterprise" else 5) if mode in ("agent", "snippet") else 1
                if mode == "once":
                    observed = await asyncio.to_thread(kb.search, case["question"], 5)
                    report["initial_observation"] = observed
                    report["events"].append({"role": "tool", "name": "search", "content": observed,
                                             "arguments_json": json.dumps({"query": case["question"], "limit": 5})})
                    request += "\nSupplied evidence (data):\n" + json.dumps(observed, ensure_ascii=False)
                elif mode in ("observed", "gold"):
                    if mode == "gold":
                        texts = [kb.read(doc, 0, 1000000) for doc in case["gold_ids"]]
                    else:
                        trace = read(HISTORY / "runs/enterprise-dev100-hybrid-minilm-wide/traces" / f"{case['id']}.json")
                        texts = [e["content"] for e in trace["events"] if e["role"] == "tool"]
                    # Preserve all original evidence; no answer, facts, historical prediction or judge passed to generation.
                    report["diagnostic_evidence_chars"] = sum(len(json.dumps(t)) for t in texts)
                    request += "\nSupplied evidence (data):\n" + json.dumps(texts, ensure_ascii=False)
                    report["evidence_source"] = "gold-document oracle" if mode == "gold" else "frozen historical observations"
                elif mode == "closed":
                    request += "\nNo corpus evidence is supplied and no retrieval is available. State if the answer cannot be established."
                model = BudgetModel(client, cwd, label=label + "/generate", timeout=180, effort="high",
                                    system_prompt=SYSTEM_PROMPT, max_calls=max_calls)
                report["model_runs"] = model.runs
                report["generation_started_unix"] = time.time()
                try:
                    report["result"] = await asyncio.wait_for(run_agent(model,
                        async_tools(kb) if mode in ("agent", "snippet") else {}, request, max_calls,
                        lambda event: (report["events"].append(event), save(path, report))), timeout=900)
                except Exception as error:
                    report["generation_attempts"].append({"runs": model.runs, "error": type(error).__name__, "detail": str(error)})
                    report.pop("model_runs", None)
                    report["error"] = str(error)
                    # Retry infrastructure failures only, up to two attempts, at a later heartbeat. Never regenerate a finished answer.
                    if len(report["generation_attempts"]) >= 2:
                        report.update(result={"status": "runtime_error", "answer": "", "sources": []}, metric=0., terminal=True)
                    save(path, report)
                    return False
                save(path, report)
            result = report["result"]
            if result["status"] != "finished":
                report.update(metric=0., terminal=True)
            elif dataset == "hotpotqa":
                scorer = load_scorer()
                report.update(metric=float(scorer.exact_match_score(result["answer"], case["gold"])),
                              f1=scorer.f1_score(result["answer"], case["gold"])[0], terminal=True)
            else:
                payload = {"question": case["question"], "reference_answer": case["gold"],
                           "reference_facts": case["facts"], "candidate_answer": result["answer"]}
                report["judgement"], _ = await assess(
                    lambda: BudgetModel(client, cwd, label=label + "/judge", timeout=150,
                                        system_prompt=PROMPT, effort="medium"),
                    payload, len(case["facts"]), report.setdefault("judge_attempts", []), lambda: save(path, report))
                report.update(metric=float(report["judgement"]["correct"]), terminal=True)
            report["seconds"] = report.get("seconds", 0) + time.monotonic() - began
            save(path, report)
            print(json.dumps({"case": label, "status": result["status"], "score": report.get("metric"), "calls": used()}), flush=True)
            return True
        finally:
            if kb:
                kb.close()


async def run(stage):
    from openai_codex import AsyncCodex
    cohort = prepare()
    if stage in ("all", "enterprise"):
        jobs = [(case, "enterprise", mode) for case in cohort["datasets"]["enterprise"]
                for mode in ("closed", "once", "agent")]
    else:
        jobs = []
    for dataset in ("financebench", "hotpotqa"):
        if stage in ("all", dataset):
            jobs += [(case, dataset, "agent") for case in cohort["datasets"][dataset]]
    if stage in ("all", "oracle"):
        jobs += [(case, "oracle", mode) for case in cohort["oracle"] for mode in ("observed", "gold")]
    if stage == "snippet":
        jobs += [(case, "enterprise", "snippet") for case in cohort["datasets"]["enterprise"][:12]]
    consecutive_failures = 0
    state(status="running", worker_stage=stage)
    try:
        async with AsyncCodex(rag_codex_config("/private/tmp")) as client:
            account = await client.account(refresh_token=False)
            if getattr(getattr(account.account, "root", account.account), "type", None) != "chatgpt":
                raise RuntimeError("Existing ChatGPT subscription required; no fallback")
            for case, dataset, mode in jobs:
                if used() >= 480:
                    state(status="experiment_reserve_reached")
                    break
                try:
                    okay = await one(client, case, dataset, mode)
                except Exception as error:
                    state(last_error=str(error))
                    print(json.dumps({"case": case["id"], "error": str(error)}), flush=True)
                    okay = False
                consecutive_failures = 0 if okay else consecutive_failures + 1
                summarize()
                if consecutive_failures >= 3:
                    state(status="transport_paused", retry_after_unix=time.time() + 1800)
                    print("Paused after three consecutive failures; no model fallback.", flush=True)
                    return
            else:
                state(status="stage_complete", worker_stage=stage)
    except Exception as error:
        state(status="startup_failed", last_error=str(error), retry_after_unix=time.time() + 1800)
        raise
    finally:
        summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "analyze", "summarize", "all", "enterprise", "financebench", "hotpotqa", "oracle", "snippet"], default="prepare")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("prepare", "analyze", "summarize"):
        prepare()
        if args.stage == "analyze":
            data = failure_analysis()
            print(json.dumps({k: {f: v[f] for f in ("n", "buckets", "categories")} for k, v in data.items() if isinstance(v, dict)}, indent=2))
        elif args.stage == "summarize":
            print(json.dumps(summarize(), ensure_ascii=False, indent=2))
    else:
        with (OUTPUT / "worker.lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit("Another overnight worker is active; do not launch a second.")
            asyncio.run(run(args.stage))
