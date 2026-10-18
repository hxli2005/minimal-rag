"""Read-only CPU token census for scheduling the unchanged full-corpus encoding."""
import gzip
from itertools import islice
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer
from gpu import SETTINGS, save

tokenizer = AutoTokenizer.from_pretrained("models/embedding", padding_side="left", local_files_only=True)
base = json.loads(Path("parallel-base.json").read_text())
started = time.time()
state = {"started": started, "documents": 0, "tokens": 0, "shard_work": [],
         "settings": SETTINGS, "status": "counting", "cpu_threads": 4}
save("token-workload.json", state)
with gzip.open("enterprise.jsonl.gz", "rt") as stream:
    while True:
        docs = [json.loads(line) for line in islice(stream, 4096)]
        if not docs:
            break
        tokens = tokenizer([(d.get("title", "") + " " + d["text"]).strip() for d in docs],
                           padding=False, truncation=True, max_length=4096,
                           add_special_tokens=True, return_attention_mask=False)["input_ids"]
        number = len(state["shard_work"])
        item = {"file": f"shard-{number:05d}.npz", "documents": len(docs), "tokens": sum(map(len, tokens))}
        state["shard_work"].append(item)
        state["documents"] += item["documents"]
        state["tokens"] += item["tokens"]
        state["seconds"] = time.time()-started
        if len(state["shard_work"]) % 4 == 0:
            save("token-workload.json", state)
assert state["documents"] == 511962
assert sum(x["tokens"] for x in state["shard_work"][:len(base["shards"])] ) == base["tokens"]
state.update(status="complete", frozen_prefix_tokens_match=True, cuda_initialized=torch.cuda.is_initialized())
save("token-workload.json", state)
print(json.dumps({k:v for k,v in state.items() if k not in ("shard_work", "settings")}), flush=True)
