"""Run the shipped tiny corpus through the actual core CLI under tonight's budget."""
import asyncio
from pathlib import Path

from budget import BudgetModel, MODEL
from minimal_rag.applications import qa
from minimal_rag.indexed_rag import IndexedKnowledgeBase
from worker import ROOT, OUTPUT, read, used
import json


class ShowcaseModel(BudgetModel):
    def __init__(self, client, cwd, model, timeout, **kwargs):
        if model != MODEL:
            raise ValueError("Tonight permits only sol")
        super().__init__(client, cwd, timeout=timeout, label="showcase/cloneable/generate", **kwargs)


if __name__ == "__main__":
    remaining = 500 - used()
    if remaining < 2:
        raise SystemExit("Insufficient shared budget; keep the offline fixture and existing real report.")
    qa.CodexModel = ShowcaseModel
    directory = OUTPUT / "cloneable-showcase"
    directory.mkdir(parents=True, exist_ok=True)
    corpus = directory / "corpus.sqlite"
    fixture = ROOT / "examples/sol-demo/documents.jsonl"
    if not corpus.exists():
        IndexedKnowledgeBase.build(corpus, [json.loads(line) for line in fixture.read_text().splitlines()])
    report = directory / "report.json"
    if report.exists():
        raise SystemExit("Existing attempt retained; do not regenerate a successful showcase.")
    qa.cli(["--live", "--index", str(corpus), "--question", (ROOT / "examples/sol-demo/question.txt").read_text().strip(),
            "--model", MODEL, "--max-turns", str(min(4, remaining)), "--output", str(report)])
