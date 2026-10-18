"""Build full-corpus indexes separately from benchmark questions and labels."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

from minimal_rag.indexed_rag import IndexedKnowledgeBase

from minimal_rag.paths import ROOT
BASE = ROOT / "data/benchmarks"


def finance():
    import pymupdf
    for i, path in enumerate(sorted((BASE / "financebench/source/pdfs").glob("*.pdf")), 1):
        with pymupdf.open(path) as pdf:
            for number, page in enumerate(pdf):
                # PyMuPDF sorted extraction preserves layout-oriented reading order.
                text = page.get_text("text", sort=True)
                yield {"id": f"{path.stem}:p{number}", "title": f"{path.stem.replace('_', ' ')} page {number + 1}",
                       "text": text, "source": f"financebench://{path.stem}#page={number + 1}"}
        if i % 20 == 0:
            print(json.dumps({"dataset": "financebench", "pdfs_indexed": i}), flush=True)


def enterprise():
    import pyarrow.parquet as pq
    path = BASE / "enterprise/source/data/documents/test.parquet"
    stream = pq.ParquetFile(path)
    occurrences, aliases = {}, {}
    for batch in stream.iter_batches(batch_size=1000):
        for row in batch.to_pylist():
            # Filled once the released schema is inspected; no question-dependent filtering.
            doc_id = row.get("doc_id", row.get("document_id", row.get("id")))
            text = row.get("text", row.get("content"))
            if not isinstance(doc_id, str) or not isinstance(text, str):
                raise ValueError(f"unexpected document schema: {sorted(row)}")
            occurrences[doc_id] = occurrences.get(doc_id, 0) + 1
            if occurrences[doc_id] > 1:
                original = doc_id
                doc_id = f"{original}~duplicate{occurrences[original]}"
                aliases[doc_id] = original
            title = row.get("title", doc_id)
            source = row.get("path", row.get("source", row.get("source_type", "")))
            yield {"id": doc_id, "title": str(title), "text": text, "source": f"enterprise://{source}/{doc_id}"}
    # Four released IDs are reused for different content. Preserve every record;
    # only the scoring adapter maps the duplicate IDs back to official IDs.
    (BASE / "enterprise/duplicate-id-aliases.json").write_text(json.dumps(aliases, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["financebench", "enterprise"])
    args = parser.parse_args()
    base = BASE / args.dataset
    manifest = json.loads((base / "manifest.json").read_text())
    path = base / "corpus.sqlite"
    begin = time.monotonic()
    stats = IndexedKnowledgeBase.build(path, finance() if args.dataset == "financebench" else enterprise())
    stats.update({"dataset": args.dataset, "source_manifest_sha256": hashlib.sha256((base / 'manifest.json').read_bytes()).hexdigest(),
                  "indexer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "retriever_sha256": hashlib.sha256((ROOT / 'src/minimal_rag/indexed_rag.py').read_bytes()).hexdigest(),
                  "seconds": time.monotonic() - begin, "bytes": path.stat().st_size,
                  "pymupdf_version": importlib.metadata.version("pymupdf"), "pyarrow_version": importlib.metadata.version("pyarrow")})
    (base / "index.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)
