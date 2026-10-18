"""BF16 document encoding and exact GPU search for the independent-tool experiment."""
import argparse
import gzip
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import pickle
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
SETTINGS = dict(model="Qwen/Qwen3-Embedding-8B", dtype="bfloat16", dimensions=4096,
                passage_max_length=4096, query_max_length=8192, pooling="eos", normalize=True,
                normalization_dtype="float32", vector_dtype="float32",
                passage_prefix="", query_prefix=PREFIX, append_eos_token=False,
                document_input="(title + ' ' + text).strip()", search="exact float32 inner product")


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2)+"\n"); tmp.replace(path)


class Encoder:
    def __init__(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left", local_files_only=True)
        self.model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16,
                                              attn_implementation="sdpa", local_files_only=True).cuda().eval()
        assert self.model.config.hidden_size == 4096

    def tokenize(self, texts, length):
        return self.tokenizer(texts, padding=False, truncation=True, max_length=length,
                              add_special_tokens=True, return_attention_mask=False)["input_ids"]

    @torch.inference_mode()
    def encode(self, tokens):
        batch = self.tokenizer.pad({"input_ids": tokens}, padding=True, return_tensors="pt").to("cuda")
        hidden = self.model(**batch, use_cache=False).last_hidden_state[:, -1]
        # Keep inference BF16, but normalize/store in FP32 like the official prebuilt vectors.
        return torch.nn.functional.normalize(hidden.float(), p=2, dim=-1)


def batches(tokens, max_batch, token_budget):
    order = sorted(range(len(tokens)), key=lambda i: len(tokens[i]))
    batch = []
    for i in order:
        if batch and (len(batch) >= max_batch or (len(batch)+1)*len(tokens[i]) > token_budget):
            yield batch; batch = []
        batch.append(i)
    if batch:
        yield batch


def encode(args):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    previous = json.loads((out / "state.json").read_text()) if args.resume and (out / "state.json").exists() else None
    if previous and previous["settings"] != SETTINGS:
        raise ValueError("cannot resume with different embedding settings")
    if previous and previous.get("complete"):
        print(json.dumps(previous), flush=True); return
    started = time.monotonic(); encoder = Encoder(args.model)
    load_seconds = time.monotonic() - started
    state = previous or dict(settings=SETTINGS, load_seconds=0, documents=0, tokens=0,
                 batch_size=args.batch_size, token_budget=args.token_budget, shards=[])
    state["load_seconds"] += load_seconds
    elapsed = state.get("seconds", 0)
    save(out / "state.json", state)
    started = time.monotonic()
    with gzip.open(args.input, "rt") as stream:
        for _ in range(state["documents"]):
            if not stream.readline(): raise ValueError("resume input is shorter than the saved index")
        while True:
            docs = []
            for _ in range(4096):
                line = stream.readline()
                if not line: break
                docs.append(json.loads(line))
            if not docs: break
            shard = out / f"shard-{len(state['shards']):05d}.npz"
            if shard.exists():
                raise FileExistsError(shard)
            tokens = encoder.tokenize([(d.get("title", "") + " " + d["text"]).strip() for d in docs], 4096)
            vectors = np.empty((len(docs), 4096), dtype=np.float32)
            for indices in batches(tokens, args.batch_size, args.token_budget):
                if time.time() >= args.deadline:
                    raise TimeoutError("Stage deadline reached; platform shutdown still required")
                vectors[indices] = encoder.encode([tokens[i] for i in indices]).cpu().numpy()
            np.savez(shard, ids=np.array([d["id"] for d in docs]), vectors=vectors)
            state["shards"].append(shard.name)
            state["documents"] += len(docs)
            state["tokens"] += sum(map(len, tokens))
            state.update(seconds=elapsed+time.monotonic()-started,
                         peak_cuda_bytes=max(state.get("peak_cuda_bytes", 0), torch.cuda.max_memory_allocated()))
            state["documents_per_second"] = state["documents"]/state["seconds"]
            state["estimated_full_seconds"] = 511962/state["documents_per_second"]
            save(out / "state.json", state)
            print(json.dumps(state), flush=True)
    state["complete"] = True
    save(out / "state.json", state)


def load_indexes(enterprise, browsecomp):
    indexes = {}
    for corpus, folder, pattern, count in [("enterprise", enterprise, "shard-*.npz", 511962),
                                           ("browsecomp", browsecomp, "*.pkl", 100195)]:
        ids = []
        matrix = torch.empty((count, 4096), dtype=torch.float32, device="cuda")
        for path in sorted(Path(folder).glob(pattern)):
            if path.suffix == ".npz":
                with np.load(path, allow_pickle=False) as data:
                    vectors, lookup = data["vectors"], data["ids"].tolist()
            else:
                # Only the verified official prebuilt repository is accepted for this input.
                with path.open("rb") as stream:
                    vectors, lookup = pickle.load(stream)
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.shape != (len(lookup), 4096) or not np.isfinite(vectors).all():
                raise ValueError(f"Invalid vectors: {path}")
            matrix[len(ids):len(ids)+len(lookup)].copy_(torch.from_numpy(vectors))
            ids.extend(map(str, lookup))
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError(f"Incomplete or duplicated {corpus} index: {len(ids)}")
        indexes[corpus] = (ids, matrix)
    return indexes


def serve(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = Encoder(args.model)
    indexes = load_indexes(args.enterprise, args.browsecomp)
    health = dict(settings=SETTINGS, documents={k: len(v[0]) for k, v in indexes.items()})

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, value):
            data = json.dumps(value).encode()
            self.send_response(status); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            self.respond(200 if self.path == "/health" else 404, health if self.path == "/health" else {})

        def do_POST(self):
            if self.path != "/search":
                self.respond(404, {}); return
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                query, limit, offset = body["query"], body["limit"], body["offset"]
                if not isinstance(query, str) or len(query)>1000 or not 1<=limit<=20 or offset<0:
                    raise ValueError("invalid query or pagination")
                ids, vectors = indexes[body["corpus"]]
                if offset >= len(ids):
                    self.respond(200, {"hits": [], "next_offset": None}); return
                with torch.inference_mode():
                    q = encoder.encode(encoder.tokenize([PREFIX + query], 8192))[0]
                    scores = vectors @ q
                    # Stable descending order makes pagination deterministic even for tied vectors.
                    positions = torch.argsort(scores, descending=True, stable=True)[offset:offset+limit].tolist()
                self.respond(200, {"hits": [{"document_id": ids[i], "score": scores[i].item()} for i in positions],
                                   "next_offset": offset+limit if offset+limit<len(ids) else None})
            except (KeyError, ValueError, TypeError) as error:
                self.respond(400, {"error": str(error)})

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(json.dumps(health), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("encode", "serve"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--token-budget", type=int, default=32768)
    parser.add_argument("--deadline", type=float, default=float("inf"))
    parser.add_argument("--enterprise")
    parser.add_argument("--browsecomp")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    encode(args) if args.action == "encode" else serve(args)
