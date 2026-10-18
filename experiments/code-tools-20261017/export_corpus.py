"""Export every original document for real full-corpus ripgrep; no labels read."""
import argparse
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from urllib.parse import quote, urlsplit


def export(index, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    db = sqlite3.connect(Path(index).resolve().as_uri() + "?mode=ro", uri=True)
    catalog = sqlite3.connect(output / "catalog.sqlite")
    catalog.execute("CREATE TABLE files(id TEXT PRIMARY KEY, path TEXT UNIQUE)")
    started, total_chars, n = time.monotonic(), 0, 0
    try:
        for n, (docid, title, text, source) in enumerate(db.execute("SELECT id,title,text,source FROM documents"), 1):
            title = re.sub(r"[^\w ()-]+", "_", title).strip() or "untitled"
            title = title.encode("utf-8")[:140].decode("utf-8", errors="ignore")
            kind = re.sub(r"[^\w-]", "_", urlsplit(source).netloc) or "documents"
            safe_id = quote(docid, safe="")
            relative = Path(kind) / safe_id[-2:] / f"{safe_id}--{title}.txt"
            path = output / "text" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode("utf-8"))
            os.utime(path, (0, 0))
            catalog.execute("INSERT INTO files VALUES (?,?)", (docid, relative.as_posix()))
            total_chars += len(text)
            if n % 10000 == 0:
                catalog.commit()
                print(json.dumps({"documents": n, "seconds": round(time.monotonic()-started)}), flush=True)
        catalog.commit()
        result = {"index": str(Path(index).resolve()), "documents": n, "text_chars": total_chars,
                  "seconds": time.monotonic()-started, "text": "Exact UTF-8 source text, no title/header added.",
                  "paths": "Source type / ID suffix shard / ID--sanitized title.txt. All mtimes equal; no source dates inferred."}
        (output / "manifest.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result), flush=True)
    finally:
        catalog.close(); db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export(args.index, args.output)
