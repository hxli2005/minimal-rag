"""Download immutable public benchmark sources; never select the corpus using gold answers."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import time
from pathlib import Path

import httpx

from minimal_rag.paths import ROOT
BASE = ROOT / "data/benchmarks"
SEED = 20261012


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, path, expected_size=None, git_sha=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (expected_size is None or path.stat().st_size == expected_size):
        if git_sha is None or hashlib.sha1(f"blob {path.stat().st_size}\0".encode() + path.read_bytes()).hexdigest() == git_sha:
            return
    part = path.with_suffix(path.suffix + ".part")
    for attempt in range(4):
        try:
            start = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={start}-"} if start else {}
            with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=120) as response:
                response.raise_for_status()
                mode = "ab" if start and response.status_code == 206 else "wb"
                with part.open(mode) as stream:
                    for block in response.iter_bytes(1 << 20):
                        stream.write(block)
            if expected_size is not None and part.stat().st_size != expected_size:
                raise ValueError(f"size mismatch: {path.name}")
            if git_sha and hashlib.sha1(f"blob {part.stat().st_size}\0".encode() + part.read_bytes()).hexdigest() != git_sha:
                raise ValueError(f"Git blob mismatch: {path.name}")
            part.replace(path)
            return
        except Exception as error:
            # Do not print redirect URLs containing signed query credentials.
            print(json.dumps({"file": path.name, "attempt": attempt + 1, "error": type(error).__name__,
                              "http_status": getattr(getattr(error, "response", None), "status_code", None)}), flush=True)
            if attempt == 3:
                raise RuntimeError(f"download failed: {path.name}") from None
            time.sleep(1 + attempt)


def get_json(url):
    response = httpx.get(url, timeout=90, follow_redirects=True)
    response.raise_for_status()
    return response.json()


def stratify(rows, category, n_dev):
    groups = {}
    for row in rows:
        groups.setdefault(category(row), []).append(row)
    rng = random.Random(SEED)
    for group in groups.values():
        rng.shuffle(group)
    exact = {key: len(group) * n_dev / len(rows) for key, group in groups.items()}
    counts = {key: int(value) for key, value in exact.items()}
    for key in sorted(groups, key=lambda key: (-(exact[key] - counts[key]), key))[:n_dev - sum(counts.values())]:
        counts[key] += 1
    dev = [groups[key].pop() for key in sorted(groups) for _ in range(counts[key])]
    test = [row for key in sorted(groups) for row in groups[key]]
    rng.shuffle(dev)
    rng.shuffle(test)
    return dev, test


def finance():
    base, repo = BASE / "financebench", "patronus-ai/financebench"
    manifest = base / "source/tree.json"
    if not manifest.exists():
        save(manifest, get_json(f"https://api.github.com/repos/{repo}/git/trees/main?recursive=1"))
    tree = json.loads(manifest.read_text())
    if tree.get("truncated"):
        raise ValueError("truncated repository listing")
    files = [x for x in tree["tree"] if x["type"] == "blob" and
             (x["path"].startswith("pdfs/") or x["path"].startswith("data/") or x["path"] == "README.md")]
    def download(item):
        fetch(f"https://raw.githubusercontent.com/{repo}/{tree['sha']}/{item['path']}",
              base / "source" / item["path"], item.get("size"), item["sha"])
        return item["path"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i, name in enumerate(pool.map(download, files), 1):
            if i % 20 == 0 or i == len(files):
                print(json.dumps({"dataset": "financebench", "downloaded": i, "total": len(files)}), flush=True)
    rows = [json.loads(line) for line in (base / "source/data/financebench_open_source.jsonl").read_text().splitlines() if line]
    dev, test = stratify(rows, lambda r: r["question_type"], 50)
    save(base / "dev.json", dev)
    save(base / "test.json", test)
    save(base / "manifest.json", {"repo": repo, "revision": tree["sha"], "seed": SEED,
         "dev_n": len(dev), "test_n": len(test), "corpus": "ALL PDFs in repository; not gold-selected",
         "files": [{"path": x["path"], "bytes": x["size"], "git_blob_sha1": x["sha"]} for x in files],
         "questions_sha256": digest(base / "source/data/financebench_open_source.jsonl")})


def enterprise():
    base, dataset = BASE / "enterprise", "onyx-dot-app/EnterpriseRAG-Bench"
    manifest = base / "source/hf-info.json"
    if not manifest.exists():
        save(manifest, get_json(f"https://huggingface.co/api/datasets/{dataset}"))
    info = json.loads(manifest.read_text())
    tree_path = base / "source/hf-tree.json"
    if not tree_path.exists():
        save(tree_path, get_json(f"https://huggingface.co/api/datasets/{dataset}/tree/{info['sha']}?recursive=true&expand=false"))
    files = [x for x in json.loads(tree_path.read_text()) if x["path"].endswith((".parquet", "README.md"))]
    for item in files:
        print(json.dumps({"dataset": "enterprise", "downloading": item["path"], "bytes": item["size"]}), flush=True)
        path = base / "source" / item["path"]
        fetch(f"https://huggingface.co/datasets/{dataset}/resolve/{info['sha']}/{item['path']}", path, item["size"])
        actual = digest(path)
        expected = item.get("lfs", {}).get("oid")
        if expected and actual != expected:
            raise ValueError("LFS SHA256 mismatch")
    import pyarrow.parquet as pq
    table = pq.read_table(base / "source/data/questions/test.parquet")
    rows = table.to_pylist()
    print(json.dumps({"dataset": "enterprise", "question_schema": str(table.schema), "document_schema": str(pq.read_schema(base / 'source/data/documents/test.parquet'))}), flush=True)
    cat = lambda r: str(r.get("category", r.get("question_type", "unknown")))
    dev, test = stratify(rows, cat, 100)
    save(base / "dev.json", dev)
    save(base / "test.json", test)
    save(base / "manifest.json", {"dataset": dataset, "revision": info["sha"], "seed": SEED,
         "dev_n": len(dev), "test_n": len(test), "corpus": "ALL released documents",
         "files": [{"path": x["path"], "bytes": x["size"], "sha256": digest(base / 'source' / x['path'])} for x in files]})


def hotpot():
    base = BASE / "hotpotqa"
    # Prior 100 questions are development-only; new final holdout never overlaps them.
    previous = json.loads((base / "sample-100-seed20261011.json").read_text())
    old_ids = {r["_id"] for r in previous}
    order = list(range(7405))
    random.Random(20261011).shuffle(order)
    indices = order[100:200]
    import pyarrow.parquet as pq
    listing = base / "source/hf-parquet.json"
    if not listing.exists():
        save(listing, get_json("https://datasets-server.huggingface.co/parquet?dataset=hotpotqa%2Fhotpot_qa"))
    item = next(x for x in json.loads(listing.read_text())["parquet_files"] if x["config"] == "distractor" and x["split"] == "validation")
    path = base / "source/validation.parquet"
    fetch(item["url"], path, item["size"])
    all_rows = pq.read_table(path).to_pylist()
    if len(all_rows) != 7405:
        raise ValueError("population changed")
    def convert(row):
        return {"_id": row["id"], "question": row["question"], "answer": row["answer"],
                "type": row["type"], "level": row["level"],
                "context": list(zip(row["context"]["title"], row["context"]["sentences"])),
                "supporting_facts": list(zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]))}
    if [all_rows[i]["id"] for i in order[:100]] != [row["_id"] for row in previous]:
        raise ValueError("row order changed from the frozen baseline")
    rows = [convert(all_rows[i]) for i in indices]
    if len({r["_id"] for r in rows}) != 100 or old_ids & {r["_id"] for r in rows}:
        raise ValueError("holdout overlap or duplicates")
    save(base / "dev.json", previous)
    save(base / "test.json", rows)
    save(base / "optimization-split.json", {"seed": 20261011, "dev_indices": order[:100], "test_indices": indices,
         "dev_sha256": digest(base / "dev.json"), "test_sha256": digest(base / "test.json"),
         "parquet_sha256": digest(path), "note": "dev is previously inspected; test new and must not be used for tuning"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["finance", "enterprise", "hotpot"])
    globals()[parser.parse_args().dataset]()
