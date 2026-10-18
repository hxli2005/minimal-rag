"""Run a frozen, complete benchmark suite; record holdout versus known regression."""
import argparse
import importlib.metadata
import json
import platform
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from minimal_rag_experiments.hybrid_rag import ROOT, file_sha256


def verify(freeze):
    for filename, expected in freeze["files"].items():
        if file_sha256(filename) != expected:
            raise ValueError(f"frozen input changed: {filename}")
    for name, expected in freeze["versions"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"frozen package version changed: {name}")
    runtime = {"python": platform.python_version(), "sqlite": sqlite3.sqlite_version,
               "system": platform.system(), "machine": platform.machine()}
    if freeze.get("runtime", runtime) != runtime:
        raise ValueError("frozen runtime changed")


def run_jobs(jobs, output):
    children, logs = [], []
    try:
        for name, command in jobs:
            log = (output / f"{name}.log").open("x")
            logs.append(log)
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            children.append((name, process))
            print(json.dumps({"stage": "started", "name": name, "pid": process.pid}), flush=True)
        remaining = list(children)
        while remaining:
            for item in list(remaining):
                name, process = item
                if process.poll() is not None:
                    print(json.dumps({"stage": "finished", "name": name, "returncode": process.returncode}), flush=True)
                    remaining.remove(item)
            if remaining:
                time.sleep(20)
        if any(process.returncode != 0 for _, process in children):
            raise RuntimeError("a frozen run failed; preserve its traces and make a documented recovery decision")
    finally:
        for _, process in children:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for _, process in children:
            if process.poll() is None:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.terminate()
        for log in logs:
            log.close()


def main(args):
    freeze = json.loads(Path(args.freeze).read_text())
    verify(freeze)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "frozen.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n")
    config, jobs = freeze["configuration"], []
    for dataset in freeze["test_sizes"]:
        command = [sys.executable, "-m", "minimal_rag_experiments.rag_optimization_eval", "--dataset", dataset,
                   "--split", "test", "--known-regression" if config.get("evaluation_kind") == "known_regression" else "--final-holdout",
                   "--retrieval", "hybrid", "--variant", config["variant"],
                   "--model", config["model"], "--models", config["models"], "--rerank", str(config["rerank"]),
                   "--candidates", str(config["candidates"]), "--max-turns", str(config["max_turns"]),
                   "--lexical-prior", str(config.get("lexical_prior", 0)),
                   "--preview-count", str(config.get("preview_count", 0)),
                   "--timeout", str(config["timeout"]), "--case-timeout", str(config["case_timeout"]),
                   "--concurrency", str(config["concurrency"]), "--judge-model", config["judge_model"],
                   "--output", str(output / dataset)]
        if dataset == "hotpotqa":
            command += ["--short-style", config["hotpot_short_style"]]
        jobs.append((dataset, command))
    run_jobs(jobs, output)
    verify(freeze)
    summaries = {}
    for dataset, n in freeze["test_sizes"].items():
        summary = json.loads((output / dataset / "summary.json").read_text())
        if summary["completed"] != n or summary["metrics"]["graded"] != n:
            raise ValueError("test suite incomplete; do not present partial scores as final")
        summaries[dataset] = summary["metrics"]
    (output / "suite.json").write_text(json.dumps({"complete": True, "metrics": summaries}, indent=2) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze", required=True)
    p.add_argument("--output", required=True)
    main(p.parse_args())
