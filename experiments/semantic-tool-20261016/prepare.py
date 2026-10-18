"""Prepare public corpus inputs and download the official precomputed index."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import httpx


def download(out):
    repo = "Tevatron/browsecomp-plus-indexes"
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        meta = client.get(f"https://huggingface.co/api/datasets/{repo}")
        meta.raise_for_status()
        revision = meta.json()["sha"]
        listing = client.get(f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/qwen3-embedding-8b")
        listing.raise_for_status()
        files = [f for f in listing.json() if f["type"] == "file"]

    def one(info):
        target = out / "browsecomp" / Path(info["path"]).name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            tmp = target.with_suffix(".partial")
            failures = 0
            while not tmp.exists() or tmp.stat().st_size < info["size"]:
                start = tmp.stat().st_size if tmp.exists() else 0
                end = min(start + 16*1024*1024, info["size"]) - 1
                try:
                    with httpx.stream("GET", f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{info['path']}",
                                      headers={"Range": f"bytes={start}-{end}"},
                                      follow_redirects=True, timeout=120) as response:
                        response.raise_for_status()
                        if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                            raise ValueError("download server did not honor the resume range")
                        with tmp.open("ab") as stream:
                            for block in response.iter_bytes(1024 * 1024): stream.write(block)
                except httpx.HTTPError:
                    failures += 1
                    if failures >= 5: raise
                    time.sleep(2)
            if tmp.stat().st_size != info["size"]:
                raise ValueError("incomplete official index download")
            tmp.replace(target)
        digest = hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
        print(json.dumps({"downloaded": target.name, "bytes": target.stat().st_size}), flush=True)
        return {"path": info["path"], "bytes": target.stat().st_size, "sha256": digest}

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, files))
    (out / "browsecomp-source.json").write_text(json.dumps({"repo": repo, "revision": revision, "files": results}, indent=2)+"\n")


def export(corpus, out):
    db = sqlite3.connect(Path(corpus).resolve().as_uri() + "?mode=ro", uri=True)
    with gzip.open(out / "enterprise.jsonl.gz", "wt", compresslevel=1) as stream:
        for n, row in enumerate(db.execute("SELECT id,title,text FROM documents ORDER BY rowid"), 1):
            stream.write(json.dumps(dict(zip(("id", "title", "text"), row)), ensure_ascii=False)+"\n")
    print(json.dumps({"exported": n}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("download", "export"))
    parser.add_argument("--output", default="runs/semantic-tool-20261016")
    parser.add_argument("--corpus", default="data/benchmarks/enterprise/corpus.sqlite")
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    download(out) if args.action == "download" else export(args.corpus, out)
