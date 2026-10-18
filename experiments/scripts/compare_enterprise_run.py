"""Compare one complete EnterpriseRAG run with frozen public-answer regrading."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from compare_rag_holdout import paired


def read(path):
    return json.loads(path.read_text())


def main(args):
    candidate, reference = Path(args.candidate), Path(args.reference)
    left, right = read(candidate / "summary.json"), read(reference / "summary.json")
    ref_config = read(reference / "config.json")
    config = left["config"]
    if (left["completed"] != left["requested"] or left["metrics"]["graded"] != left["requested"]
            or not right["metrics"].get("complete")):
        raise ValueError("only complete, fully graded cohorts can be compared")
    if (config["dataset"] != "enterprise" or config["dataset"] != ref_config["dataset"]
            or config["split"] != ref_config["split"]
            or config["sample_sha256"] != ref_config["reference_sha256"]
            or config["judge_model"] != ref_config["judge"]
            or config["judge_prompt"] != ref_config["prompt"]
            or config["agent_sha256"] != ref_config["adapter_sha256"]):
        raise ValueError("cohort or local judge is not comparable")
    for data in (left, right):
        if len(data["cases"]) != len({r["id"] for r in data["cases"]}):
            raise ValueError("duplicate question IDs")
    correct = paired(left["cases"], right["cases"], "correct")
    wins, losses = correct["wins"], correct["losses"]
    count = wins + losses
    correct["exact_mcnemar_two_sided_p"] = min(1.0,
        2 * sum(math.comb(count, i) for i in range(min(wins, losses) + 1)) / 2 ** count)
    refs = {r["id"]: r for r in right["cases"]}
    categories = {}
    for category in sorted({r["category"] for r in left["cases"]}):
        rows = [r for r in left["cases"] if r["category"] == category]
        categories[category] = {"n": len(rows), "candidate_correct": sum(r["correct"] for r in rows),
                                "reference_correct": sum(refs[r["id"]]["correct"] for r in rows)}
    report = {"complete": True, "evaluation_kind": config.get("evaluation_kind", "legacy_" + config["split"]),
        "candidate": str(candidate.resolve()), "reference": str(reference.resolve()),
        "candidate_metrics": left["metrics"], "reference_metrics": right["metrics"],
        "paired_correctness": correct, "paired_overall": paired(left["cases"], right["cases"], "overall"),
        "by_category": categories,
        "source_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (candidate / "summary.json", reference / "summary.json", reference / "config.json", Path(__file__))},
        "limitations": ["Same local judge and fixed references; these are not official leaderboard scores.",
            "Public systems have different generation models, retrieval budgets and infrastructure.",
            "Development scores reflect selection; reused test results are known-set regression, not new blind validation.",
            "One candidate generation per question in this run; paired intervals measure question variation, not repeated-run reliability."]}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"candidate": left["metrics"]["correct"], "reference": right["metrics"]["correct"],
                      "paired_correctness": correct}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
