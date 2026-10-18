"""Detach nohup runs; no scheduling, retry, or budget logic."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess

RUNS = {
    "smoke-enterprise": ("enterprise", "dev", 3, 3),
    "smoke-financebench": ("financebench", "dev", 2, 2),
    "smoke-hotpotqa": ("hotpotqa", "dev", 2, 2),
    "enterprise-dev": ("enterprise", "dev", 3, None),
    "enterprise-test": ("enterprise", "test", 3, None),
    "financebench-test": ("financebench", "test", 2, None),
    "hotpotqa-test": ("hotpotqa", "test", 2, None),
}
ENV = {"PYTHONPATH": "src:experiments/src", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}


def command(name):
    dataset, split, concurrency, limit = RUNS[name]
    cmd = ["nohup", ".venv/bin/python", "experiments/sota-20261014/run_frozen.py",
           "--dataset", dataset, "--split", split, "--variant", "v6",
           "--output", "data/benchmarks/sota-20261014/" + name,
           "--model", "gpt-6.1-sol", "--judge-model", "gpt-6.1-sol", "--max-turns", "12",
           "--timeout", "180", "--case-timeout", "900", "--concurrency", str(concurrency),
           "--retrieval", "hybrid", "--candidates", "100", "--rerank", "200", "--models",
           "models/hybrid-rag/" + ("models-bge-m3-cascade100.json" if dataset == "enterprise" else "models-minilm.json")]
    if dataset == "enterprise":
        cmd += ["--preview-count", "15"]
    if dataset == "hotpotqa":
        cmd += ["--short-style", "extractive"]
    if limit:
        cmd += ["--limit", str(limit)]
    if split == "test":
        cmd += ["--known-regression"]
    return cmd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=RUNS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    directory = Path("data/benchmarks/sota-20261014")
    cmd = command(args.name) + (["--resume"] if args.resume else [])
    with (directory / (args.name + ".log")).open("a") as log:
        process = subprocess.Popen(cmd, env=dict(os.environ, **ENV), stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {"pid": process.pid, "started_utc": datetime.now(timezone.utc).isoformat(), "command": cmd, "environment": ENV}
    with (directory / (args.name + ".launches.jsonl")).open("a") as stream:
        stream.write(json.dumps(receipt) + "\n")
    (directory / (args.name + ".pid")).write_text(str(process.pid) + "\n")
    print(json.dumps(receipt))
