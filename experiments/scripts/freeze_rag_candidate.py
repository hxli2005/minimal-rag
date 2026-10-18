"""Freeze the selected configuration before generation on a complete evaluation cohort."""
import argparse
import importlib.metadata
import hashlib
import inspect
import json
import platform
import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from minimal_rag_experiments.hybrid_rag import DEFAULT_MODELS, ROOT, file_sha256
from minimal_rag.loop import run_agent
from minimal_rag.paths import PACKAGE_DIR
from minimal_rag_experiments.paths import EXPERIMENT_PACKAGE_DIR
from minimal_rag_experiments.hybrid_rag import load_model_config


def main(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    paths = [PACKAGE_DIR / name for name in (
        "loop.py", "codex_adapter.py", "indexed_rag.py", "applications/qa.py")]
    paths += [EXPERIMENT_PACKAGE_DIR / name for name in (
        "hybrid_rag.py", "rerank_fusion.py", "preview_rag.py", "cascade_rag.py", "rag_optimization_eval.py", "rag_judge.py", "hotpot_benchmark.py")]
    paths += [Path(__file__).resolve().parent / name for name in (
        "freeze_rag_candidate.py", "run_rag_holdout.py", "compare_enterprise_run.py", "compare_rag_holdout.py")]
    paths += [Path(args.models).resolve()]
    model_config = load_model_config(args.models)
    for role in ("embedding", "reranker", *(["prefilter"] if model_config.get("prefilter") else [])):
        paths += [p for p in Path(model_config[role]["path"]).rglob("*")
                  if p.is_file() and (p.suffix in {".json", ".safetensors", ".jinja"} or p.name in {"vocab.txt", "merges.txt", "tokenizer.model", "spm.model", "sentencepiece.bpe.model"})]
    for dataset in args.datasets:
        paths.append(ROOT / "data/benchmarks" / dataset / "test.json")
        if dataset != "hotpotqa":
            paths.append(ROOT / "data/benchmarks" / dataset / "corpus.sqlite")
            index = ROOT / "data/benchmarks" / dataset / "corpus.dense/manifest.json"
            if not json.loads(index.read_text()).get("complete"):
                raise ValueError("full corpus index is not complete")
            paths.append(index)
            paths += [index.parent / "vectors.npy", index.parent / "metadata.sqlite"]
    if "hotpotqa" in args.datasets:
        paths.append(EXPERIMENT_PACKAGE_DIR / "_vendor/hotpot_evaluate_v1.py")
    if "enterprise" in args.datasets:
        paths += [ROOT / "data/benchmarks/enterprise/duplicate-id-aliases.json",
                  ROOT / "data/benchmarks/optimization-20261011/public-reference/data/raw_data/answers_cdl.jsonl"]
        reference = ROOT / "data/benchmarks/optimization-20261011/holdout/public-comparisons/cdl"
        if args.known_regression:
            paths += [reference / "summary.json", reference / "config.json"]
    if "financebench" in args.datasets:
        paths += [ROOT / "data/benchmarks/optimization-20261011/public-reference/dewey/answers-opus46.jsonl",
                  ROOT / "data/benchmarks/optimization-20261011/public-reference/dewey/incidental-exposure.json"]
    record = {"created": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "configuration": {"model": args.model, "variant": args.variant, "models": str(Path(args.models).resolve()),
                                "candidates": 100, "rerank": args.rerank, "max_turns": 12,
                                "lexical_prior": args.lexical_prior,
                                "preview_count": args.preview_count,
                                "evaluation_kind": "known_regression" if args.known_regression else "final_holdout",
                                "timeout": 180, "case_timeout": 900, "concurrency": 3,
                                "hotpot_short_style": "extractive", "judge_model": "gpt-6.1-sol"},
              "files": {str(path): file_sha256(path) for path in paths},
              "versions": {name: importlib.metadata.version(name) for name in ("openai-codex", "sentence-transformers", "transformers", "torch", "numpy", "pydantic")},
              "runtime": {"python": platform.python_version(), "sqlite": sqlite3.sqlite_version,
                          "system": platform.system(), "machine": platform.machine()},
              "test_sizes": {k: {"hotpotqa": 100, "financebench": 100, "enterprise": 400}[k] for k in args.datasets},
              "kernel_lines": len(inspect.getsource(run_agent).rstrip().splitlines()),
              "kernel_sha256": hashlib.sha256(inspect.getsource(run_agent).rstrip().encode()).hexdigest(),
              "rules": ["One attempt per test question; no best-of-N or configuration changes based on test results.",
                        "Judge retries are limited to transport/schema errors; the first valid judgment is retained, including negative grades.",
                        "Keep all original questions, including failures and ambiguous annotations.",
                        "Official Hotpot answer EM/F1 are not replaced by semantic diagnostics.",
                        "Local common-judge comparisons are not official leaderboard rankings."]}
    if "financebench" in args.datasets:
        record["rules"].append("FinanceBench has incidental public-baseline failure metadata exposure for five test IDs; disclose it.")
    if args.known_regression:
        record["rules"].append("Earlier results on this cohort have been inspected; this run is known-set regression, not untouched validation.")
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshots = output.with_suffix("")
    snapshots.mkdir()
    for path in paths:
        if path.suffix == ".py":
            (snapshots / path.name).write_bytes(path.read_bytes())
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="gpt-6.1-sol")
    p.add_argument("--variant", choices=["v6", "v7", "v8", "v9", "v10", "v11"], default="v6")
    p.add_argument("--models", default=str(DEFAULT_MODELS))
    p.add_argument("--rerank", type=int, required=True)
    p.add_argument("--lexical-prior", type=float, default=0)
    p.add_argument("--preview-count", type=int, default=0)
    p.add_argument("--datasets", nargs="+", choices=["hotpotqa", "financebench", "enterprise"], default=["hotpotqa", "financebench", "enterprise"])
    p.add_argument("--known-regression", action="store_true")
    a = p.parse_args()
    if a.rerank < 1 or a.preview_count < 0 or len(a.datasets) != len(set(a.datasets)):
        p.error("invalid budget or duplicate datasets")
    main(a)
