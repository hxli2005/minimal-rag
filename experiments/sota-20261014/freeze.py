"""Verify smoke configurations against historical controls and freeze commands."""
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex

from count_requests import count
from run_frozen import HISTORY, ROOT
from start_run import ENV, command

HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / "data/benchmarks/sota-20261014"


def read(path):
    return json.loads(path.read_text())


def normalized(value):
    if isinstance(value, dict):
        return {k: normalized(v) for k, v in value.items() if k not in ("path", "source_path") and not k.endswith("_sha256")}
    if isinstance(value, list):
        return [normalized(v) for v in value]
    return value


def verify(dataset, directory):
    old, new = read(HISTORY[dataset] / "config.json"), read(directory / "config.json")
    keys = ["variant", "settings", "judge_model", "max_turns", "timeout", "case_timeout",
            "judge_prompt", "format_request", "no_gold_in_generation", "compact_pdf", "short_style"]
    differences = {k: [old.get(k), new.get(k)] for k in keys if old.get(k) != new.get(k)}
    if normalized(old["retrieval"]) != normalized(new["retrieval"]):
        differences["retrieval"] = [normalized(old["retrieval"]), normalized(new["retrieval"])]
    # Provenance checks supplement parameter equality, without treating moved files as behavior changes.
    assert old["agent_sha256"] == new["agent_sha256"], "historical adapter must be exact"
    assert old.get("index") == new.get("index"), "corpus/index metadata changed"
    assert old["retrieval"].get("dense_manifest") == new["retrieval"].get("dense_manifest"), "dense index metadata changed"
    for key in ("sample_sha256",):
        if old["split"] == new["split"]:
            assert old[key] == new[key]
    assert not differences, differences
    historical = read(HISTORY[dataset] / "code.json")["agent"]
    original = next(n for n in ast.parse(historical).body if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_agent")
    current = next(n for n in ast.parse((ROOT / "src/minimal_rag/loop.py").read_text()).body if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_agent")
    assert ast.dump(original) == ast.dump(current)
    return {"historical_config": str((HISTORY[dataset] / "config.json").relative_to(ROOT)),
            "checked_fields": keys + ["retrieval parameters, model IDs/revisions, prefilter, preview, dense manifest"],
            "parameter_differences": differences, "historical_adapter_identical": True,
            "core_loop_ast_identical": True,
            "allowed_metadata_differences": ["generation model astra -> sol", "relocated paths/imports",
                "smoke dev prefix versus full cohort", "known-regression scope on formal test", "brief-specified concurrency"]}


if __name__ == "__main__":
    audits = {}
    for dataset, n in (("enterprise", 3), ("financebench", 2), ("hotpotqa", 2)):
        directory = OUTPUT / ("smoke-" + dataset)
        summary = read(directory / "summary.json")
        assert summary["completed"] == n and summary["metrics"]["graded"] == n
        assert summary["metrics"]["status"] == {"finished": n} and summary["metrics"]["errors"] == 0
        audits[dataset] = {**verify(dataset, directory), "smoke_metrics": summary["metrics"], "smoke_requests": count(directory)}
    names = ["enterprise-dev", "hotpotqa-test", "financebench-test", "enterprise-test"]
    commands = {}
    for name in names:
        cmd = command(name)
        log = f"data/benchmarks/sota-20261014/{name}.log"
        commands[name] = {"argv": cmd, "environment": ENV,
                          "shell": " ".join(k + "=" + shlex.quote(v) for k, v in ENV.items()) + " " + shlex.join(cmd) + f" > {log} 2>&1 < /dev/null &"}
    frozen = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "model": "gpt-6.1-sol",
              "judge_model": "gpt-6.1-sol", "variant": "v6", "audits": audits, "commands": commands,
              "request_caps": {"enterprise-dev": 900, "hotpotqa-test": 600, "financebench-test": 900, "enterprise-test": 3200},
              "enterprise_test_gate": {"dev_correct_min": 82, "infrastructure_failures_max": 5},
              "poll_seconds": 300, "bootstrap_samples": 10000, "bootstrap_seed": 20261012,
              "assembly": "run_frozen.py restores the historical adapter/retrieval snapshots; only imports and asset paths are relocated. The specified current evaluator and unchanged core loop are used.",
              "limits": "Known regression, local semantic judge; no official leaderboard claim. Outer SDK calls include failed/judge retries; provider-internal HTTP retries are unobservable."}
    destination = HERE / "frozen.json"
    if destination.exists():
        if read(destination).get("status") != "pending_smoke":
            raise SystemExit("already frozen; do not overwrite")
        archive = OUTPUT / "offline-draft"
        archive.mkdir(exist_ok=True)
        destination.rename(archive / "frozen.json")
    destination.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n")
    print("All three smoke runs passed; configuration frozen.")
