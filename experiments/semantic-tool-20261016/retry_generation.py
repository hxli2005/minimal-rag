"""Retry infrastructure failures under the user's amended A/B protocol."""
import argparse
import asyncio
from collections import Counter
import fcntl
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

from minimal_rag_experiments import benchmark

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "runs/semantic-tool-20261016"
RETRYABLE = {"runtime_error", "interrupted"}


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def main(run):
    lock = (run / "generation-retry.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = read(run / "config.json")
    source, cases = benchmark.load_cases(config["dataset"], config["split"])
    assert [c["id"] for c in cases] == config["ids"]
    retries = run / "generation-retries"
    retries.mkdir(exist_ok=True)
    initial_summary = retries / "initial-summary.json"
    if not initial_summary.exists():
        shutil.copy2(run / "summary.json", initial_summary)
    initial = read(initial_summary)
    while not (EXPERIMENT / "STOP").exists():
        reports = {c["id"]: read(run / "traces" / f"{c['id']}.json") for c in cases}
        pending = [q for q, t in reports.items() if t.get("result", {}).get("status") in RETRYABLE]
        if not pending:
            break
        quota = read(EXPERIMENT / "quota-state.json")
        if quota.get("ordinary_usage_allowed") is not True:
            save(run / "generation-retry-state.json", {"status": "waiting_for_quota", "pending": pending})
            time.sleep(5)
            continue
        rounds = sorted(retries.glob("round-*"))
        if rounds and not (rounds[-1] / "merged.json").exists():
            attempt = rounds[-1]
            ids = read(attempt / "requested.json")["ids"]
        else:
            attempt = retries / f"round-{len(rounds)+1:03d}"
            ids = pending
            save(attempt / "requested.json", {"ids": ids, "created_at": time.time(),
                 "selection": "Runtime error or interrupted status only; no judgement-based selection"})
        save(run / "generation-retry-state.json", {"status": "running", "attempt": str(attempt), "ids": ids})
        for qid in ids:
            archive = retries / "initial-failures" / f"{qid}.json"
            if not archive.exists():
                archive.parent.mkdir(exist_ok=True)
                shutil.copy2(run / "traces" / f"{qid}.json", archive)
        wanted = set(ids)
        benchmark.load_cases = lambda dataset, split: (source, [c for c in cases if c["id"] in wanted])
        args = SimpleNamespace(dataset=config["dataset"], split=config["split"], output=str(attempt),
            limit=None, model=config["model"], concurrency=config["concurrency"],
            judge_concurrency=config["judge_concurrency"], resume=(attempt / "config.json").exists(),
            semantic_url=config.get("semantic", {}).get("endpoint"))
        asyncio.run(benchmark.main(args))
        selected = []
        for qid in ids:
            path = attempt / "traces" / f"{qid}.json"
            if not path.exists():
                continue
            candidate = read(path)
            result = candidate.get("result", {})
            if result and result["status"] not in RETRYABLE and reports[qid]["result"]["status"] in RETRYABLE:
                candidate["generation_retry"] = {"selected_attempt": str(path.relative_to(ROOT)),
                    "initial_failure": str((retries / "initial-failures" / f"{qid}.json").relative_to(ROOT)),
                    "policy": "First non-infrastructure terminal result; correctness is not a selection criterion"}
                save(run / "traces" / f"{qid}.json", candidate)
                reports[qid] = candidate
                if config["dataset"] == "browsecomp":
                    export = benchmark.official_export(candidate)
                    save(run / "official" / f"{qid}.json", export)
                    # The A bundle has no judge results yet; keep its exact selected answers in sync.
                    if run.name == "browsecomp-A":
                        save(EXPERIMENT / "judge-bundle/runs/A" / f"{qid}.json", export)
                selected.append(qid)
        save(attempt / "merged.json", {"selected": selected, "at": time.time()})
        summaries = [read(p) for p in retries.glob("round-*/summary.json")]
        counts = Counter(initial["requests"])
        for s in summaries:
            counts.update(s["requests"])
        accounting = {"initial_generation_status": initial["metrics"]["status"], "rounds": len(summaries),
            "all_attempts_requests": dict(counts), "selected_retry_questions": [q for q,t in reports.items() if "generation_retry" in t],
            "all_attempts_tokens": {key: initial["metrics"].get(key, 0) + sum(s["metrics"].get(key, 0) for s in summaries)
                                    for key in ("input_tokens", "cached_input_tokens", "output_tokens")},
            "policy": "Same technical-failure retries for A and B; first valid judgement retained"}
        save(run / "generation-retry-accounting.json", accounting)
        summary = benchmark.summarize(config, cases, reports)
        summary["generation_retry_accounting"] = accounting
        save(run / "summary.json", summary)
        if any(t["result"]["status"] in RETRYABLE for t in reports.values()):
            time.sleep(30)
    save(run / "generation-retry-state.json", {"status": "complete" if not (EXPERIMENT / "STOP").exists() else "stopped",
         "updated": time.time(), "run": str(run)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    main(parser.parse_args().run.resolve())
