"""Download pinned public model assets for the two-tool RAG; no hosted inference."""
import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from minimal_rag_experiments.hybrid_rag import DEFAULT_MODELS

EMBEDDING = ("BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")
RERANKERS = {
    "mxbai": ("mixedbread-ai/mxbai-rerank-base-v1", "800f24c113213a187e65bde9db00c15a2bb12738"),
    "minilm": ("cross-encoder/ms-marco-MiniLM-L6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a"),
    "bge-m3": ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"),
    "qwen3": ("Qwen/Qwen3-Reranker-0.6B", "e61197ed45024b0ed8a2d74b80b4d909f1255473"),
}


def main(args):
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    config = {}
    roles = [("embedding", EMBEDDING), ("reranker", RERANKERS[args.reranker])]
    if args.prefilter:
        roles.append(("prefilter", RERANKERS[args.prefilter]))
    for kind, (repo, revision) in roles:
        path = snapshot_download(repo, revision=revision, token=False, cache_dir=output.parent,
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "vocab.txt", "merges.txt", "tokenizer.model", "spm.model", "sentencepiece.bpe.model", "README.md"],
            ignore_patterns=["onnx/*", "openvino/*"], max_workers=4)
        config[kind] = {"id": repo, "revision": revision, "path": path}
    config.update(query_prompt="Represent this sentence for searching relevant passages: ", max_length=512, precision="float16")
    if args.rerank_max_length is not None:
        config["rerank_max_length"] = args.rerank_max_length
    if args.prefilter:
        config.update(prefilter_count=args.prefilter_count, prefilter_max_length=512)
    if output.exists() and json.loads(output.read_text()) != config:
        raise FileExistsError("different model configuration exists; choose a new --output path")
    output.write_text(json.dumps(config, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reranker", choices=RERANKERS, default="minilm")
    p.add_argument("--output", default=str(DEFAULT_MODELS))
    p.add_argument("--rerank-max-length", type=int)
    p.add_argument("--prefilter", choices=RERANKERS)
    p.add_argument("--prefilter-count", type=int, default=100)
    args = p.parse_args()
    if args.prefilter_count < 1 or (args.rerank_max_length is not None and args.rerank_max_length < 1):
        p.error("model budgets must be positive")
    main(args)
