"""Run the existing evaluator with historical transport/retrieval, relocating imports only.

The current core loop is unchanged and AST-identical to the historical loop.
Saved benchmark code is authoritative; generated assembly files stay under data/.
"""
import importlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/benchmarks"
HISTORY = {
    "enterprise": BASE / "optimization-20261012/enterprise-dev100-bge-cascade-preview",
    "financebench": BASE / "optimization-20261011/holdout/financebench",
    "hotpotqa": BASE / "optimization-20261011/holdout/hotpotqa",
}
MODULES = {
    "agent": "minimal_rag.codex_adapter",
    "retriever": "minimal_rag.indexed_rag",
    "hybrid": "minimal_rag_experiments.hybrid_rag",
    "fusion": "minimal_rag_experiments.rerank_fusion",
    "preview": "minimal_rag_experiments.preview_rag",
    "cascade": "minimal_rag_experiments.cascade_rag",
}
LEGACY = dict(zip(
    ("agentic_rag", "indexed_rag", "hybrid_rag", "rerank_fusion", "preview_rag", "cascade_rag"),
    MODULES.values(),
))


def main():
    dataset = sys.argv[sys.argv.index("--dataset") + 1]
    source = json.loads((HISTORY[dataset] / "code.json").read_text())
    extra = json.loads((HISTORY["enterprise"] / "code.json").read_text())
    assembly = BASE / "sota-20261014/assembly" / dataset
    assembly.mkdir(parents=True, exist_ok=True)
    for key, name in MODULES.items():
        code = source.get(key, extra[key])
        for old, new in LEGACY.items():
            code = code.replace("from labs.context." + old + " import", "from " + new + " import")
        code = code.replace("ROOT = Path(__file__).resolve().parents[3]", "from minimal_rag.paths import ROOT")
        code = code.replace('ROOT / "model-cache/hybrid-rag/models-minilm.json"',
                            'ROOT / "models/hybrid-rag/models-minilm.json"')
        path = assembly / (key + ".py")
        path.write_text(code)
        parent_name, attr = name.rsplit(".", 1)
        parent = importlib.import_module(parent_name)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        setattr(parent, attr, module)
        spec.loader.exec_module(module)
    position = sys.argv.index("--models") + 1
    manifest = Path(sys.argv[position]).resolve()
    models = json.loads(manifest.read_text())
    for role in ("embedding", "reranker", "prefilter"):
        if role in models:
            path = Path(models[role]["path"])
            models[role]["path"] = str(path if path.is_absolute() else (manifest.parent / path).resolve())
    relocated = assembly / "models.json"
    relocated.write_text(json.dumps(models, ensure_ascii=False, indent=2) + "\n")
    sys.argv[position] = str(relocated)
    runpy.run_module("minimal_rag_experiments.rag_optimization_eval", run_name="__main__")


if __name__ == "__main__":
    main()
