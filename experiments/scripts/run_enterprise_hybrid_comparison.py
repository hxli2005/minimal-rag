"""One-shot build dependency: after the full index publishes, run two dev cohorts.

This is not a recurring scheduler. No partial corpus, test questions, or automatic
parameter changes are allowed here. Logs and failures are preserved.
"""
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from minimal_rag.paths import ROOT
BASE = ROOT / "data/benchmarks/optimization-20261011"
MANIFEST = ROOT / "data/benchmarks/enterprise/corpus.dense/manifest.json"


def main():
    jobs = [
        ("enterprise-dev100-hybrid-minilm-wide", "models-minilm.json", 200),
        ("enterprise-dev100-hybrid-mxbai60", "models.json", 60),
    ]
    deadline = time.monotonic() + 7200
    print(json.dumps({"stage": "waiting_for_complete_index", "manifest": str(MANIFEST)}), flush=True)
    while not MANIFEST.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("full enterprise index was not published within two hours")
        time.sleep(20)
    if not json.loads(MANIFEST.read_text()).get("complete"):
        raise RuntimeError("index is incomplete")
    children, logs = [], []
    try:
        for name, models, rerank in jobs:
            output = BASE / "runs" / name
            if output.exists():
                raise FileExistsError(output)
            log = (BASE / "hybrid" / f"{name}.log").open("x")
            logs.append(log)
            command = [sys.executable, "-m", "minimal_rag_experiments.rag_optimization_eval", "--dataset", "enterprise",
                       "--variant", "v6", "--model", "gpt-6-astra", "--retrieval", "hybrid",
                       "--models", str(ROOT / "models/hybrid-rag" / models), "--rerank", str(rerank),
                       "--timeout", "180", "--case-timeout", "900", "--concurrency", "3", "--output", str(output)]
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            children.append((name, child, output))
            print(json.dumps({"stage": "started", "run": name, "pid": child.pid}), flush=True)
        remaining = list(children)
        while remaining:
            for entry in list(remaining):
                name, child, output = entry
                if child.poll() is not None:
                    summary = output / "summary.json"
                    metrics = json.loads(summary.read_text()).get("metrics") if summary.exists() else None
                    print(json.dumps({"stage": "finished", "run": name, "returncode": child.returncode, "metrics": metrics}), flush=True)
                    remaining.remove(entry)
            if remaining:
                time.sleep(20)
        if any(child.returncode != 0 for _, child, _ in children):
            raise RuntimeError("a comparison run failed; inspect preserved logs and traces")
    finally:
        for _, child, _ in children:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
        for _, child, _ in children:
            if child.poll() is None:
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.terminate()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
