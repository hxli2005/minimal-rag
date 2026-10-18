"""Copy completed immutable shards while encoding; verify the final manifest."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/semantic-tool-20261016"
REMOTE = "/root/autodl-tmp/semantic-tool-20261016"
HELPER = RUN / "connection-a800/remote-scp.py"
OUT = RUN / "a800-results"
OUT.mkdir(exist_ok=True)
(OUT / "enterprise-index").mkdir(exist_ok=True)
lock = (RUN / "index-collector.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
state = {"pid": os.getpid(), "started": time.time(), "shards": 0, "status": "watching"}


def save(**fields):
    state.update(updated=time.time(), **fields)
    tmp = RUN / "index-collector-state.tmp"
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(RUN / "index-collector-state.json")


def get(relative, required=True):
    target = OUT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".receiving")
    result = subprocess.run([sys.executable, str(HELPER), "get", str(partial), REMOTE + "/" + relative],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)
    if result.returncode:
        if required:
            raise RuntimeError("Cannot retrieve " + relative + ": " + result.stdout.decode(errors="replace"))
        return None
    partial.replace(target)
    return target


save()
while time.time() < 1792241400 + 3600:
    if (RUN / "STOP").exists() or (RUN / "workflow-complete.json").exists():
        save(status="stopped"); break
    try:
        full = json.loads(get("full-state.json").read_text())
        index_path = get("enterprise-index/state.json", required=False)
        if index_path:
            index = json.loads(index_path.read_text())
            missing = []
            for name in index["shards"]:
                assert Path(name).name == name and name.startswith("shard-") and name.endswith(".npz")
                if not (OUT / "enterprise-index" / name).exists():
                    missing.append("enterprise-index/" + name)
            with ThreadPoolExecutor(max_workers=3) as downloads:
                list(downloads.map(get, missing))
            save(shards=len(index["shards"]), documents=index["documents"], stage=full["stage"], status="watching")
        if full.get("status") in ("complete", "failed"):
            for name in ("full-process.json", "full.log", "encode.log", "corpus.log", "browsecomp-index.log",
                         "enterprise-source.json", "browsecomp-source.json", "models.json",
                         "parallel-plan.json", "parallel-base.json"):
                get(name, required=False)
            for rank in range(full.get("worker_count", 0)):
                get(f"encode-{rank}.log", required=False)
                get(f"enterprise-index/workers/worker-{rank}.json", required=False)
            if full["status"] == "failed":
                save(status="pipeline_failed", pipeline=full); break
            manifest = json.loads(get("enterprise-index/manifest.json").read_text())
            for shard in manifest["shards"]:
                p = OUT / "enterprise-index" / shard["file"]
                assert p.stat().st_size == shard["bytes"]
                assert hashlib.file_digest(p.open("rb"), "sha256").hexdigest() == shard["sha256"]
            assert manifest["index"]["documents"] == 511962 and manifest["index"]["complete"]
            save(status="complete", documents=511962, final_manifest_verified=True,
                 next_action="Confirm full index validation and shut down A800 in platform console")
            break
    except Exception as error:
        save(status="retrying", error=repr(error))
        print(repr(error), flush=True)
    time.sleep(120)
else:
    save(status="deadline", next_action="Inspect platform shutdown and retrieve remaining files")
