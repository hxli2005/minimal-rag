"""Paired reporting for completed frozen tests and public answers regraded locally."""
import argparse
import json
import random
import statistics
from pathlib import Path


def paired(left, right, metric, seed=20261012):
    left, right = {r["id"]: r for r in left}, {r["id"]: r for r in right}
    if not left or left.keys() != right.keys():
        raise ValueError("paired comparison requires identical complete question sets")
    deltas = [float(left[k][metric]) - float(right[k][metric]) for k in sorted(left)]
    rng, n = random.Random(seed), len(deltas)
    means = sorted(sum(rng.choices(deltas, k=n)) / n for _ in range(10000))
    return {"n": n, "difference": statistics.mean(deltas), "paired_bootstrap_95_interval": [means[249], means[9749]],
            "wins": sum(d > 0 for d in deltas), "losses": sum(d < 0 for d in deltas),
            "ties": sum(d == 0 for d in deltas), "bootstrap_samples": 10000, "seed": seed}


def main(args):
    suite = Path(args.suite)
    frozen = json.loads((suite / "frozen.json").read_text())
    datasets = {}
    raw = {}
    for dataset, n in frozen["test_sizes"].items():
        data = json.loads((suite / dataset / "summary.json").read_text())
        if (data["completed"] != n or data["metrics"]["graded"] != n
                or data["config"]["split"] != "test" or not data["config"]["final_holdout"]):
            raise ValueError("only complete frozen test results belong in this report")
        datasets[dataset] = data["metrics"]
        raw[dataset] = data["cases"]
    comparisons = {}
    for dataset, path in (("financebench", args.finance_reference), ("enterprise", args.enterprise_reference)):
        baseline = json.loads(Path(path).read_text())
        if not baseline["metrics"].get("complete"):
            raise ValueError("public-answer regrading is incomplete")
        comparisons[dataset] = {"reference_metrics": baseline["metrics"],
                               "paired_correctness": paired(raw[dataset], baseline["cases"], "correct"),
                               "paired_overall": paired(raw[dataset], baseline["cases"], "overall")}
    report = {"complete": True, "candidate": frozen["configuration"], "test_metrics": datasets,
              "public_answer_comparisons": comparisons,
              "limitations": [
                  "Reference answers are regraded by the same local judge; these are not official leaderboard scores.",
                  "Public systems use different generation models, tool budgets and infrastructure. This compares systems, not architecture in isolation.",
                  "Dewey uses 85 filings, while our financial corpus includes all 368 repository PDFs; its public results do not include source IDs.",
                  "Five FinanceBench test IDs had incidental public-baseline failure-metadata exposure; no cases were removed or specially tuned.",
                  "One generation per test question. Bootstrap intervals describe question-level variation, not repeated-model-run reliability.",
                  "Hotpot official EM/F1 are retained; local semantic diagnostics from development are not substituted.",
              ]}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"metrics": datasets, "comparisons": comparisons}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", required=True)
    p.add_argument("--finance-reference", required=True)
    p.add_argument("--enterprise-reference", required=True)
    p.add_argument("--output", required=True)
    main(p.parse_args())
