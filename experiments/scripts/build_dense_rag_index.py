"""Embed every passage in an immutable SQLite corpus; atomic, resumable publication.

No questions or relevance labels are read by this script.
"""
import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

from minimal_rag_experiments.hybrid_rag import DEFAULT_MODELS, LocalModels, file_sha256, read_only


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def build(corpus, destination, models, batch_size=64, resume=False):
    corpus, destination = Path(corpus).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    working = destination.with_name(destination.name + ".building")
    if working.exists() and not resume:
        raise FileExistsError(f"{working}; pass --resume to continue the same configuration")
    working.mkdir(parents=True, exist_ok=True)
    source = read_only(corpus)
    n = source.execute("SELECT count(*) FROM chunks").fetchone()[0]
    if not n:
        raise ValueError("empty corpus")
    print(json.dumps({"stage": "fingerprint", "corpus": str(corpus), "chunks": n}), flush=True)
    config = {"corpus_sha256": file_sha256(corpus), "chunks": n, "dimension": models.dimension,
              "embedding": models.identity, "input_format": "title + newline + original chunk text",
              "dtype": "float32", "similarity": "normalized dot product; exact search"}
    config_path = working / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("partial index configuration differs")
    else:
        save(config_path, config)
    meta_path = working / "metadata.sqlite"
    if not meta_path.exists():
        temporary = working / "metadata.building.sqlite"
        if temporary.exists():
            temporary.unlink()  # Only this script's unfinished derived metadata, not the source corpus.
        meta = sqlite3.connect(temporary)
        meta.execute("ATTACH DATABASE ? AS source", (str(corpus),))
        meta.execute("CREATE TABLE documents(id TEXT PRIMARY KEY,title TEXT,source TEXT)")
        meta.execute("INSERT INTO documents SELECT id,title,source FROM source.documents")
        meta.execute("CREATE TABLE passages(position INTEGER PRIMARY KEY,chunk_id INTEGER UNIQUE,document_id TEXT)")
        meta.execute("INSERT INTO passages SELECT row_number() OVER (ORDER BY rowid)-1,rowid,document_id FROM source.chunks")
        meta.execute("CREATE INDEX passages_document ON passages(document_id)")
        meta.commit()
        meta.close()
        temporary.replace(meta_path)
    meta = read_only(meta_path)
    progress_path = working / "progress.json"
    done = json.loads(progress_path.read_text())["completed"] if progress_path.exists() else 0
    vectors_path = working / "vectors.npy"
    vectors = np.lib.format.open_memmap(vectors_path, mode="r+" if vectors_path.exists() else "w+",
                                       dtype=np.float32, shape=(n, models.dimension))
    if list(vectors.shape) != [n, models.dimension]:
        raise ValueError("partial embedding matrix has wrong shape")
    last_id = meta.execute("SELECT chunk_id FROM passages WHERE position=?", (done - 1,)).fetchone()[0] if done else 0
    cursor = source.execute("SELECT rowid,title,text FROM chunks WHERE rowid>? ORDER BY rowid", (last_id,))
    begin, initial, logged = time.monotonic(), done, 0.0
    while rows := cursor.fetchmany(2048):
        block = models.encode([r["title"] + "\n" + r["text"] for r in rows], batch_size=batch_size)
        vectors[done:done + len(rows)] = block
        done += len(rows)
        vectors.flush()
        elapsed = time.monotonic() - begin
        progress = {"completed": done, "total": n, "fraction": done / n,
                    "seconds_this_session": elapsed, "chunks_per_second": (done - initial) / max(elapsed, .001)}
        save(progress_path, progress)
        if elapsed - logged >= 30 or done == n:
            print(json.dumps(progress), flush=True)
            logged = elapsed
    if done != n or meta.execute("SELECT count(*) FROM passages").fetchone()[0] != n:
        raise ValueError("corpus coverage mismatch")
    if file_sha256(corpus) != config["corpus_sha256"]:
        raise ValueError("source corpus changed during indexing; refusing publication")
    meta.close()
    source.close()
    del vectors
    save(working / "manifest.json", {**config, "complete": True, "source_path": str(corpus),
                                     "device": models.device, "builder_sha256": file_sha256(__file__)})
    working.replace(destination)
    print(json.dumps({"stage": "complete", "index": str(destination), "chunks": done}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--models", default=str(DEFAULT_MODELS))
    p.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if a.batch_size < 1:
        p.error("batch size must be positive")
    build(a.corpus, a.output, LocalModels(a.models, a.device), a.batch_size, a.resume)
