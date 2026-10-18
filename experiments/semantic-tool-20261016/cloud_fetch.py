"""Fetch frozen public inputs on the GPU host without uploading the local corpus."""
import argparse
from email.utils import parsedate_to_datetime
import gzip
import hashlib
import json
from pathlib import Path
import time

import requests

ENTERPRISE = {"repo": "onyx-dot-app/EnterpriseRAG-Bench", "revision": "69916e31c68aa5963c00248fd7f0bc12d04fd235",
              "path": "data/documents/test.parquet", "bytes": 1409893131,
              "sha256": "6b0747bf160af9427b12101537d53056ac592ada9831c1a98ae01fa50a8d2a9f"}


def fetch(item, target):
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".partial")
    failures = 0
    while not target.exists() and (not tmp.exists() or tmp.stat().st_size < item["bytes"]):
        start = tmp.stat().st_size if tmp.exists() else 0
        try:
            url = f"https://hf-mirror.com/datasets/{item['repo']}/resolve/{item['revision']}/{item['path']}"
            with requests.get(url, headers={"Range": f"bytes={start}-"}, stream=True, timeout=120) as response:
                response.raise_for_status()
                if ((response.status_code == 206 and not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"))
                        or (start and response.status_code != 206)):
                    raise ValueError("download server did not honor requested range")
                print(json.dumps({"streaming": str(target), "resumed_from": start}), flush=True)
                with tmp.open("ab") as stream:
                    for block in response.iter_content(1024*1024): stream.write(block)
                failures = 0
        except requests.RequestException as error:
            failures = 0 if tmp.exists() and tmp.stat().st_size > start else failures + 1
            if failures >= 5: raise
            status = error.response.status_code if error.response is not None else None
            delay = 60 * 2**max(0, failures-1) if status == 429 else 10
            retry_after = error.response.headers.get("Retry-After") if error.response is not None else None
            if retry_after:
                try: delay = max(delay, float(retry_after))
                except ValueError: delay = max(delay, parsedate_to_datetime(retry_after).timestamp()-time.time())
            print(json.dumps({"retry": str(target), "http_status": status, "wait_seconds": delay}), flush=True)
            time.sleep(delay)
    source = target if target.exists() else tmp
    if source.stat().st_size != item["bytes"] or hashlib.file_digest(source.open("rb"), "sha256").hexdigest() != item["sha256"]:
        raise ValueError(f"Frozen source mismatch: {target.name}")
    if not target.exists(): tmp.replace(target)
    print(json.dumps({"downloaded": str(target), "verified": True}), flush=True)


def enterprise():
    import pyarrow.parquet as pq
    fetch(ENTERPRISE, "input/enterprise.parquet")
    occurrences, n = {}, 0
    with gzip.open("enterprise.jsonl.gz", "wt", compresslevel=1) as stream:
        for batch in pq.ParquetFile("input/enterprise.parquet").iter_batches(batch_size=1000):
            for row in batch.to_pylist():
                doc_id = row.get("doc_id", row.get("document_id", row.get("id")))
                text = row.get("text", row.get("content"))
                occurrences[doc_id] = occurrences.get(doc_id, 0) + 1
                if occurrences[doc_id] > 1: doc_id = f"{doc_id}~duplicate{occurrences[doc_id]}"
                title = str(row.get("title", doc_id))
                stream.write(json.dumps({"id": doc_id, "title": title, "text": text}, ensure_ascii=False)+"\n")
                n += 1
    assert n == 511962
    Path("enterprise-source.json").write_text(json.dumps({**ENTERPRISE, "documents": n}, indent=2)+"\n")
    print(json.dumps({"documents_exported": n}), flush=True)


def browsecomp():
    source = json.loads(Path("browsecomp-source.json").read_text())
    items = [{**item, "repo": source["repo"], "revision": source["revision"]} for item in source["files"]]
    for item in items:
        fetch(item, Path("browsecomp") / Path(item["path"]).name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("enterprise", "browsecomp"))
    args = parser.parse_args()
    enterprise() if args.dataset == "enterprise" else browsecomp()
