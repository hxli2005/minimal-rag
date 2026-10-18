"""Build the offline demo through the same SQLite index used by user corpora."""
import json
import tempfile
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path

from minimal_rag.indexed_rag import IndexedKnowledgeBase


@contextmanager
def fixture_index(name="retail.jsonl"):
    rows = [json.loads(line) for line in files("minimal_rag").joinpath("fixtures", name)
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    with tempfile.TemporaryDirectory(prefix="minimal-rag-demo-") as directory:
        path = Path(directory) / "corpus.sqlite"
        IndexedKnowledgeBase.build(path, rows)
        kb = IndexedKnowledgeBase(path)
        try:
            yield kb
        finally:
            kb.close()
