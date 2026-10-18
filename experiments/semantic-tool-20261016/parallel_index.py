"""Encode disjoint original 4096-document shards on several GPUs, without changing embeddings."""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from gpu import Encoder, SETTINGS, batches, save

TOTAL = 511962
SHARD_SIZE = 4096


def worker(args):
    out = Path("enterprise-index")
    plan = json.loads(Path("parallel-plan.json").read_text())
    path = out / "workers" / f"worker-{args.rank}.json"
    state = json.loads(path.read_text()) if path.exists() else {
        "rank": args.rank, "world_size": plan["world_size"], "settings": SETTINGS,
        "documents": 0, "tokens": 0, "seconds": 0, "load_seconds": 0, "shards": []}
    assert state["settings"] == SETTINGS and state["world_size"] == plan["world_size"]
    if state.get("complete"):
        return
    loaded = time.monotonic()
    encoder = Encoder("models/embedding")
    state["load_seconds"] += time.monotonic() - loaded
    elapsed, started = state["seconds"], time.monotonic()
    save(path, state)
    with gzip.open("enterprise.jsonl.gz", "rt") as stream:
        for number in range(math.ceil(TOTAL / SHARD_SIZE)):
            lines = [stream.readline() for _ in range(min(SHARD_SIZE, TOTAL-number*SHARD_SIZE))]
            if number < plan["first_shard"] or (number-plan["first_shard"]) % plan["world_size"] != args.rank:
                continue
            name = f"shard-{number:05d}.npz"
            if name in state["shards"]:
                continue
            target = out / name
            if target.exists():
                raise FileExistsError(target)
            docs = [json.loads(line) for line in lines]
            tokens = encoder.tokenize([(d.get("title", "") + " " + d["text"]).strip() for d in docs], 4096)
            vectors = np.empty((len(docs), 4096), dtype=np.float32)
            for indices in batches(tokens, 64, 32768):
                if time.time() >= args.deadline:
                    raise TimeoutError("Parallel stage deadline reached")
                vectors[indices] = encoder.encode([tokens[i] for i in indices]).cpu().numpy()
            temporary = target.with_suffix(".npz.tmp")
            with temporary.open("wb") as output:
                np.savez(output, ids=np.array([d["id"] for d in docs]), vectors=vectors)
            temporary.replace(target)
            state["shards"].append(name)
            state["documents"] += len(docs)
            state["tokens"] += sum(map(len, tokens))
            state["seconds"] = elapsed + time.monotonic() - started
            save(path, state)
            print(json.dumps({"rank": args.rank, "shard": name, "documents": state["documents"]}), flush=True)
    state["complete"] = True
    save(path, state)


def coordinate(args):
    import torch
    assert torch.cuda.device_count() >= args.gpus
    out = Path("enterprise-index")
    plan_path, base_path = Path("parallel-plan.json"), Path("parallel-base.json")
    if not plan_path.exists():
        base = json.loads((out / "state.json").read_text())
        assert base["settings"] == SETTINGS and not base.get("complete")
        first = len(base["shards"])
        assert base["shards"] == [f"shard-{i:05d}.npz" for i in range(first)]
        assert base["documents"] == first * SHARD_SIZE
        plan = {"world_size": args.gpus, "first_shard": first, "documents": TOTAL,
                "assignment": "(original_shard_number-first_shard) modulo world_size", "settings": SETTINGS}
        save(base_path, base); save(plan_path, plan)
    base, plan = json.loads(base_path.read_text()), json.loads(plan_path.read_text())
    assert plan["world_size"] == args.gpus and plan["settings"] == SETTINGS
    full = {"stage": "encode_parallel", "started": time.time(), "deadline": args.deadline,
            "worker_count": args.gpus, "processes": {}, "preserved_documents": base["documents"]}
    processes = []

    def aggregate():
        workers = []
        for rank in range(args.gpus):
            path = out / "workers" / f"worker-{rank}.json"
            if path.exists():
                workers.append(json.loads(path.read_text()))
        names = base["shards"] + [name for w in workers for name in w["shards"]]
        assert len(names) == len(set(names))
        index = {**base, "shards": sorted(names), "complete": False,
                 "documents": base["documents"] + sum(w["documents"] for w in workers),
                 "tokens": base["tokens"] + sum(w["tokens"] for w in workers),
                 "seconds": base["seconds"] + time.time()-full["started"],
                 "gpu_seconds": base["seconds"] + sum(w["seconds"] for w in workers),
                 "parallel_world_size": args.gpus, "parallel_workers": workers}
        index.pop("documents_per_second", None); index.pop("estimated_full_seconds", None)
        save(out / "state.json", index)
        return index

    try:
        for rank in range(args.gpus):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank))
            with Path(f"encode-{rank}.log").open("a") as log:
                p = subprocess.Popen([sys.executable, "-u", __file__, "--worker", "--rank", str(rank),
                                      "--deadline", str(args.deadline)], env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append(p); full["processes"][f"encode-{rank}"] = p.pid
        save("full-state.json", full)
        while True:
            index = aggregate()
            codes = [p.poll() for p in processes]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f"Parallel worker failed: {codes}")
            if all(code == 0 for code in codes):
                break
            if time.time() >= args.deadline:
                raise TimeoutError("Parallel stage deadline reached")
            time.sleep(5)
        index = aggregate()
        assert index["documents"] == TOTAL
        assert index["shards"] == [f"shard-{i:05d}.npz" for i in range(math.ceil(TOTAL / SHARD_SIZE))]
        index["complete"] = True; save(out / "state.json", index)
        full["stage"] = "package"; save("full-state.json", full)
        shards = [{"file": name, "bytes": (out/name).stat().st_size,
                   "sha256": hashlib.file_digest((out/name).open("rb"), "sha256").hexdigest()}
                  for name in index["shards"]]
        save(out / "manifest.json", {"index": index, "shards": shards,
             "corpus": json.loads(Path("enterprise-source.json").read_text()),
             "model": json.loads(Path("models.json").read_text()), "parallel_plan": plan})
        full.update(status="complete", stage="ready_for_retrieval_and_platform_shutdown",
                    index_bytes=sum(s["bytes"] for s in shards))
    except Exception as error:
        full.update(status="failed", error=repr(error))
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        full["finished"] = time.time(); save("full-state.json", full)
    print(json.dumps(full), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    worker(args) if args.worker else coordinate(args)
