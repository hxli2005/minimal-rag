"""Prepare the frozen corpus and both indexes; the host is stopped in the platform UI."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--prepared-corpus", action="store_true")
    args = parser.parse_args()
    state = {"started": time.time(), "deadline": args.deadline, "stage": "prepare", "processes": {}}
    processes = []

    def save():
        tmp = Path("full-state.json.tmp"); tmp.write_text(json.dumps(state, indent=2)+"\n")
        tmp.replace("full-state.json")

    def start(name, command):
        with open(name + ".log", "a") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        state["processes"][name] = process.pid; processes.append(process); save()
        return process

    def wait(name, process):
        state["stage"] = name; save()
        if process.wait(timeout=max(1, args.deadline-time.time())):
            raise RuntimeError(f"{name} failed; inspect {name}.log")

    try:
        models = start("models", [sys.executable, "-u", "download.py"])
        wait("data-environment", start("data-environment", [".gpu-venv/bin/python", "-m", "pip", "install",
               "--no-cache-dir", "pyarrow==20.0.0", "numpy==1.26.4"]))
        if args.prepared_corpus:
            source = json.loads(Path("enterprise-source.json").read_text())
            assert source["documents"] == 511962
            assert hashlib.file_digest(Path("enterprise.jsonl.gz").open("rb"), "sha256").hexdigest() == source["export_sha256"]
            wait("models", models)
        else:
            corpus = start("corpus", [".gpu-venv/bin/python", "-u", "cloud_fetch.py", "enterprise"])
            wait("models", models); wait("corpus", corpus)
        browse = start("browsecomp-index", [".gpu-venv/bin/python", "-u", "cloud_fetch.py", "browsecomp"])
        wait("encode", start("encode", [".gpu-venv/bin/python", "-u", "gpu.py", "encode", "--model", "models/embedding",
              "--input", "enterprise.jsonl.gz", "--output", "enterprise-index", "--resume", "--deadline", str(args.deadline)]))
        wait("browsecomp-index", browse)
        index = json.loads(Path("enterprise-index/state.json").read_text())
        assert index.get("complete") and index["documents"] == 511962
        state["stage"] = "package"; save()
        shards = [{"file": p.name, "bytes": p.stat().st_size,
                   "sha256": hashlib.file_digest(p.open("rb"), "sha256").hexdigest()}
                  for p in sorted(Path("enterprise-index").glob("shard-*.npz"))]
        manifest = {"index": index, "shards": shards,
                    "corpus": json.loads(Path("enterprise-source.json").read_text()),
                    "model": json.loads(Path("models.json").read_text())}
        Path("enterprise-index/manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        state.update(status="complete", stage="ready_for_retrieval_and_platform_shutdown",
                     index_bytes=sum(shard["bytes"] for shard in shards))
    except Exception as error:
        state.update(status="failed", error=type(error).__name__, detail=str(error))
    finally:
        for process in processes:
            if process.poll() is None: process.terminate()
        state["finished"] = time.time(); save()
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
