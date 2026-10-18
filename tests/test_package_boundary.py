"""The deployable package must be usable without installing research code."""
import ast
from pathlib import Path


def test_core_does_not_import_experiment_package():
    source = Path(__file__).resolve().parents[1] / "src/minimal_rag"
    violations = []
    for path in source.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [node.module or ""] if isinstance(node, ast.ImportFrom) else (
                [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            if any(name.split(".")[0] == "minimal_rag_experiments" for name in names):
                violations.append(f"{path.relative_to(source)}:{node.lineno}")
    assert not violations, violations


def test_core_excludes_domain_applications_and_sentence_experiments():
    source = Path(__file__).resolve().parents[1] / "src/minimal_rag"
    assert not (source / "hybrid_rag.py").exists()
    assert not (source / "sentences.py").exists()
    assert not list((source / "applications").glob("investigation*.py"))
