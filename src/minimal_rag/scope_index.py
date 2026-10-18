"""Reuse a corpus-bound passage map without loading embeddings or models."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3
import threading


def signature(path):
    stat = path.stat()
    return (str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@lru_cache(maxsize=4)
def corpus_digest(identity):
    path = Path(identity[0])
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if signature(path) != identity:
        raise ValueError("corpus changed during scope-index verification")
    return digest.hexdigest()


class ScopeIndex:
    def __init__(self, directory, corpus):
        directory, corpus = Path(directory), Path(corpus)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("complete") is not True:
            raise ValueError("scope index is incomplete")
        if manifest["corpus_sha256"] != corpus_digest(signature(corpus)):
            raise ValueError("scope index belongs to a different corpus")
        self.db = sqlite3.connect((directory / "metadata.sqlite").resolve().as_uri() + "?mode=ro",
                                  uri=True, check_same_thread=False)
        self.lock = threading.Lock()

    def bounds(self, document_id):
        with self.lock:
            low, high = self.db.execute(
                "SELECT min(chunk_id),max(chunk_id) FROM passages WHERE document_id=?", (document_id,)).fetchone()
            # Empty documents have no passages. The source document predicate is
            # retained by the caller, including when rows are not contiguous.
            return (low, high) if low is not None else (1, 0)

    def close(self):
        with self.lock:
            self.db.close()
