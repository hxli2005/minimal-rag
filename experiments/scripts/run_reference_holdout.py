"""After candidate tests finish, regrade public answers on the same frozen questions."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from minimal_rag_experiments.hybrid_rag import ROOT
from run_rag_holdout import run_jobs, verify


def main(args):
    suite = Path(args.suite).resolve()
    deadline = time.monotonic() + 14400
    print(json.dumps({"stage": "waiting_for_complete_candidate_suite"}), flush=True)
    while not (suite / "suite.json").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("candidate suite has not completed; no partial comparison was made")
        time.sleep(20)
    frozen = json.loads((suite / "frozen.json").read_text())
    verify(frozen)
    references = ROOT / "data/benchmarks/optimization-20261011/public-reference"
    output = suite / "public-comparisons"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    jobs = []
    for name, dataset, answers in [
        ("dewey", "financebench", references / "dewey/answers-opus46.jsonl"),
        ("cdl", "enterprise", references / "data/raw_data/answers_cdl.jsonl"),
    ]:
        command = [sys.executable, "-m", "minimal_rag_experiments.rag_judge", "--dataset", dataset,
                   "--split", "test", "--answers", str(answers), "--output", str(output / name),
                   "--model", frozen["configuration"]["judge_model"], "--concurrency", "3"]
        jobs.append((name, command))
    run_jobs(jobs, output)
    verify(frozen)
    subprocess.run([sys.executable, str(ROOT / "experiments/scripts/compare_rag_holdout.py"), "--suite", str(suite),
                    "--finance-reference", str(output / "dewey/summary.json"),
                    "--enterprise-reference", str(output / "cdl/summary.json"),
                    "--output", str(suite / "comparison.json")], cwd=ROOT, check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", required=True)
    main(p.parse_args())
