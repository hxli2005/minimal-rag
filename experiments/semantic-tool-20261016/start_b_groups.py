"""Start the two B benchmarks concurrently with their unchanged per-group settings."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

from minimal_rag_experiments.benchmark import settings

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs/semantic-tool-20261016"


def main(url):
    if (RUNS / "ARCHIVED").exists():
        print(json.dumps({"archived_by_user": True, "action": "no model calls or service restart"}), flush=True)
        return
    with urlopen(url.rstrip("/") + "/health", timeout=30) as response:
        health = json.load(response)
    assert health["documents"] == {"enterprise": 511962, "browsecomp": 100195}
    assert health["settings"]["model"] == "Qwen/Qwen3-Embedding-8B"
    assert health["settings"]["dtype"] == "bfloat16" and health["settings"]["dimensions"] == 4096
    commands = []
    for dataset, split, baseline in (
        ("enterprise", "test", ROOT / "runs/search-description-20261016/enterprise-test"),
        ("browsecomp", "all", RUNS / "browsecomp-A"),
    ):
        if (RUNS / f"{dataset}-B.PAUSED").exists():
            print(json.dumps({"dataset": dataset, "paused_by_user": True}), flush=True)
            continue
        config = json.loads((baseline / "config.json").read_text())
        assert settings(dataset) == config["settings"]
        assert config["concurrency"] == 5 and config["judge_concurrency"] == 1
        output = RUNS / f"{dataset}-B"
        metadata = RUNS / f"{dataset}-B-process.json"
        if metadata.exists():
            prior = json.loads(metadata.read_text())
            live = subprocess.run(["ps", "-p", str(prior["pid"]), "-o", "command="],
                                  capture_output=True, text=True).stdout
            if any(name in live for name in ("minimal_rag_experiments.benchmark", "rate_limited_run.py")) and str(output) in live:
                print(json.dumps({"dataset": dataset, "already_running": prior["pid"]}), flush=True)
                continue
        if (output / "summary.json").exists():
            summary = json.loads((output / "summary.json").read_text())
            retryable = sum(summary["metrics"]["status"].get(key, 0) for key in ("runtime_error", "interrupted"))
            if summary["complete"] and not retryable:
                print(json.dumps({"dataset": dataset, "generation_complete": True,
                                  "next": "Handle missing official judging separately"}), flush=True)
                continue
        command = [sys.executable, "-u", str(ROOT / "experiments/semantic-tool-20261016/rate_limited_run.py"), "--dataset", dataset,
                   "--split", split, "--live", "--concurrency", "5", "--judge-concurrency", "1",
                   "--model", config["model"], "--semantic-url", url, "--output", str(output)]
        if (output / "config.json").exists():
            command.append("--resume")
        commands.append((dataset, command, metadata))
    for dataset, command, metadata in commands:
        if metadata.exists():
            with metadata.with_name(f"{dataset}-B-process-history.jsonl").open("a") as history:
                history.write(metadata.read_text().strip().replace("\n", " ")+"\n")
        with (RUNS / f"{dataset}-B.log").open("a") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        metadata.write_text(json.dumps({"pid": process.pid, "command": command, "started": time.time(),
            "execution": "Concurrent groups with unchanged per-group settings; shared subscription request backpressure",
            "transport_policy": str(RUNS / "transport-policy.json")}, indent=2)+"\n")
        print(json.dumps({"dataset": dataset, "pid": process.pid}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    main(parser.parse_args().url)
