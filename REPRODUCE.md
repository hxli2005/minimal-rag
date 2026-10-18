# Reproduction and evaluation record

**Status: submission materials prepared; maintainer review pending.** This guide covers the frozen generation configuration and keeps it separate from later evaluation and API tooling. Hosted models are not deterministic, so a rerun can produce different answers and scores.

## Results and artifacts

| Benchmark | Available result | Status |
| --- | --- | --- |
| BrowseComp-Plus, all 830 queries | 781/830 = 94.10%; counting all eight original technical failures as incorrect: 775/830 = 93.37% | Local evaluation with official scoring logic and the engine adjustment below; awaiting maintainer review |
| EnterpriseRAG-Bench, all 500 questions | All answers exported; local references: 309/400 test and 74/100 dev | Official evaluation and document-version alignment pending |

BrowseComp: [summary and paired comparison](experiments/browsecomp-frozen-judge-20261017/result.json), [submission JSON](experiments/browsecomp-frozen-judge-20261017/submission.json), [engine diff](experiments/browsecomp-frozen-judge-20261017/engine-adjustment.json), [retry accounting](evidence/reproduction-20261017/browsecomp-retry-sensitivity.json). Enterprise: [export provenance and answer-file hash](evidence/frozen-enterprise-export-20261017.json).

Question-bearing answers, decrypted questions and raw traces are retained outside Git in `runs/` and `data/`. The public summaries are not substitutes for those files. Verification of the original run requires the corresponding private artifact package; regenerating answers is a separate, paid procedure.

## Frozen generation configuration

Generation tag: `frozen-20261017`, commit `447b15e5c97e9cbef21a4ceb55003484326970a4`. This public snapshot redacts files that would reveal BrowseComp-Plus answers; agent, adapter and retrieval code are byte-identical to the internal freeze. Both main results use the same frozen loop, Codex adapter and search/read implementation, with benchmark-specific budgets:

| Setting | Enterprise | BrowseComp |
| --- | --- | --- |
| Model / reasoning | `gpt-6.1-sol` / high | Same |
| SDK | `openai-codex==0.160.0` | Same |
| Maximum model calls per question | 12 | 30 |
| Search excerpt length | 4,000 characters | 2,000 characters |
| Request / question timeout | 180 / 900 seconds | 180 / 1,800 seconds |
| Generation concurrency | 5 | 5 |

The last permitted call is constrained to produce a final answer from existing evidence. These are model-call limits, not tool-call limits: a reply can request several tools. BM25 uses 1,600-character chunks, 200-character overlap and title/body weights 2:1. `read` defaults to 4,000 characters and supports pagination. There is no external web access, embedding model or reranker in this configuration.

The recorded BrowseComp run is `runs/semantic-tool-20261016/browsecomp-A`, the BM25 control arm of an experiment. Its three core module hashes match the frozen tag; its archived runner includes an unused semantic-tool branch. The commands below use the frozen unified runner to reproduce the configuration, not the original interrupted execution schedule.

The earlier BrowseComp result, 772/830 = 93.01%, came from an earlier internal tag, `browsecomp-submission-20261015`, which is not part of this public repository. Among `loop.py`, `codex_adapter.py` and `indexed_rag.py`, only the search-description string differs between tags. Other repository files also differ. Both results remain reported.

### Runtime disclosures

Generation used an authenticated ChatGPT account through the Codex SDK. Access to the recorded model and sufficient account allowance are prerequisites; a public model API name or equivalent access is not guaranteed.

A probe using the frozen adapter observed injected built-in tool definitions and multi-agent messages, and `text.verbosity=low`. Available benchmark traces contain no observed built-in-tool activity, but this does not rule out an effect of the injected context. Later probes found that `clock.sleep` can be removed and verbosity can be changed; no switch was found to remove the remaining injection in that tested runtime. These later settings were not applied retroactively. The 3,481-token residual measured on first requests is not an exact token count for each injected section or for every subsequent request. [Probe evidence](evidence/codex-hidden-context-20261017.json)

## Set up a separate frozen checkout

Start at the root of the repository containing this guide. Keep this shell open for the following commands. The separate checkout preserves current API code and the newer evaluation materials.

```bash
export MATERIALS="$PWD"
git worktree add --detach ../minimal-rag-frozen frozen-20261017
cd ../minimal-rag-frozen
export FROZEN="$PWD"
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[codex,benchmarks]' -e ./experiments
```

Authenticate the Codex CLI with `codex login` if needed. Commands containing `--live` consume subscription allowance; the Enterprise runner also performs local judging. Do not use the current working-tree runner to reproduce the old 12/30-call behavior.

### Pin the source data

The preparation scripts otherwise discover the latest dataset metadata on first use. Seed them with the recorded public metadata before downloading:

```bash
mkdir -p data/benchmarks/enterprise/source data/benchmarks/browsecomp-plus
cp "$MATERIALS/evidence/reproduction-20261017/enterprise-hf-info.json" data/benchmarks/enterprise/source/hf-info.json
cp "$MATERIALS/evidence/reproduction-20261017/enterprise-hf-tree.json" data/benchmarks/enterprise/source/hf-tree.json
cp "$MATERIALS/evidence/reproduction-20261017/browsecomp-download-metadata.json" data/benchmarks/browsecomp-plus/download-metadata.json
```

Recorded Hugging Face dataset revisions:

- EnterpriseRAG-Bench: `69916e31c68aa5963c00248fd7f0bc12d04fd235`.
- BrowseComp-Plus questions: `144cff8e35b5eaef7e526346aa60774a9deb941f`.
- BrowseComp-Plus corpus: `b27b02bc3e45511b8b82a13e6f90ce761df726f6`.

Use a fresh data directory; changing metadata does not replace previously downloaded files. Preparation downloads the full corpus and builds a local index, so allow sufficient disk space and time.

## Enterprise generation and export

```bash
.venv/bin/python scripts/prepare_rag_benchmarks.py enterprise
.venv/bin/python scripts/index_rag_benchmarks.py enterprise
.venv/bin/python -m minimal_rag_experiments.benchmark --dataset enterprise --split test --live --output runs/enterprise-test
.venv/bin/python -m minimal_rag_experiments.benchmark --dataset enterprise --split dev --live --output runs/enterprise-dev
```

The project splits 500 questions into dev100/test400 with seed 20261012, stratified by question type. Both sets were used during development; “test” is a local filename, not an untouched official holdout. The index contains 511,962 released document records. Four IDs have different text versions; local aliases preserve all records.

The frozen runner's `--resume` skips recorded cases; it does **not** reconstruct an interrupted conversation. Incomplete traces can remain failures. Do not delete valid answers to obtain a different result.

To export the new run above into the official row format, from the frozen checkout:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
aliases = json.loads(Path('data/benchmarks/enterprise/duplicate-id-aliases.json').read_text())
rows = []
for split in ('dev', 'test'):
    questions = json.loads(Path(f'data/benchmarks/enterprise/{split}.json').read_text())
    for question in questions:
        qid = question['question_id']
        trace = Path(f'runs/enterprise-{split}/traces/{qid}.json')
        report = json.loads(trace.read_text()) if trace.exists() else {}
        result = report.get('result', {})
        finished = result.get('status') == 'finished'
        if finished:
            assert report.get('native_answer'), qid
        ids = result.get('sources', []) if finished else []
        rows.append({'question_id': qid, 'answer': result.get('answer', '') if finished else '',
                     'document_ids': list(dict.fromkeys(aliases.get(x, x) for x in ids))})
assert len(rows) == len({r['question_id'] for r in rows}) == 500
Path('runs/enterprise-500-answers.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
PY
```

This produces the submission format, not an official score or document-version adjudication. The original exported answers came from different run folders listed in the [export evidence](evidence/frozen-enterprise-export-20261017.json). The later helper `experiments/frozen-20261017/export_official.py` is tied to those original paths and is not present at the frozen tag.

One original answer cites an alias. Its text is bound to the release Parquet record, but alignment with the upstream evaluator's UUID-to-file mapping is unresolved. Confirm this and the judge configuration with the maintainers before official evaluation. Local judge scores are not directly comparable to the leaderboard. [Official evaluation instructions](https://github.com/onyx-dot-app/EnterpriseRAG-Bench/blob/main/answer_evaluation/README.md)

## BrowseComp generation and judging

```bash
.venv/bin/python experiments/browsecomp-pilot-20261015/prepare.py
.venv/bin/python -m minimal_rag_experiments.benchmark --dataset browsecomp --split all --live --concurrency 5 --output runs/browsecomp-all
```

This uses all 830 queries and 100,195 corpus documents. Official-format answer files are written under `runs/browsecomp-all/official`. Retrieval recall counts only documents returned by `search`.

In the original run, eight technical generation failures were each rerun once with the same configuration before scoring. Six of those answers were correct, giving 781/830 after recovery versus 775/830 when those eight are counted as incorrect. The latter is a sensitivity calculation using the same grades, not a separate experimental run. The recovery procedure and failed attempts are preserved; the commands above do not automatically reproduce it.

For scoring on a CUDA machine, use the prepared official source directory and copy the disclosure-linked single-GPU script into it. The machine must have access to the frozen checkout's data and answer files; set `FROZEN` and `MATERIALS` to their paths on that machine if transferring them from another host.

```bash
export OFFICIAL="$FROZEN/data/benchmarks/browsecomp-plus/official"
cp "$MATERIALS/experiments/browsecomp-frozen-judge-20261017/evaluate_run_single_gpu.py" "$OFFICIAL/scripts_evaluation/"
cd "$OFFICIAL"
uv sync --frozen --python 3.10
uv run --frozen python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-32B', revision='9216db5781bf21249d130ec9da846c4624c16137', local_dir='models/Qwen3-32B')
PY
uv run --frozen python scripts_evaluation/evaluate_run_single_gpu.py \
  --input_dir "$FROZEN/runs/browsecomp-all/official" \
  --ground_truth "$FROZEN/data/benchmarks/browsecomp-plus/queries.jsonl" \
  --tensor_parallel_size 1 --model models/Qwen3-32B
```

Official source commit: `046949032b0328319cc9a02663a759ec601d9402`. Recorded environment: Python 3.10.22, PyTorch 2.7.0, vLLM 0.9.0.1 and Transformers 4.53.2. The run used one A800 80GB, BF16, temperature 0.7, top_p 0.8, top_k 20, output cap 4096, batch size 64 and `enable_thinking=False`.

The runtime copy differs from the official script by one engine argument: `max_model_len=8192`. The unmodified engine failed before producing a judgement because its default 40,960-token context exceeded available KV cache. The longest grading prompt before chat formatting is 1,216 tokens; the recorded run completed without input truncation. See the exact [diff and hashes](experiments/browsecomp-frozen-judge-20261017/engine-adjustment.json).

The two smoke-test grades were reused in the full run. All 830 first valid judgements were retained, with no grading backfills. GPU type, tensor parallelism and stochastic decoding can affect results; this comparison is not a controlled search-description-only ablation. Raw evaluator dates use the server timezone; submission metadata uses October 17, 2026 in America/Los_Angeles. Maintainer acceptance of the engine setting and recovery policy remains pending.

## API portability and limitations

The current checkout, rather than the frozen tag, provides `--backend api` using a Responses-compatible endpoint, native function calls and explicit task instructions. It reads `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`. Install the `api` extra and follow the [current running guide](docs/runbook.md).

Minimal RAG is an independent student project, and its budget did not cover full benchmark reruns through the API. Instead, we ran a pre-registered, fixed Enterprise dev30 migration check: API 24/30 versus Codex 22/30, with 3 wins, 1 loss and 26 ties. The paired accuracy-difference interval includes zero. This passed an engineering screening criterion; it does not establish equivalence or rule out a population-level regression. Transport, tool protocol, context and stopping behavior differ. The full API benchmarks have not been run. [Protocol and results](experiments/reports/api-validation-20261017.md)

The benchmarks have development exposure, and neither model nor judge variance is captured by a single run. Keep historical results, failed attempts and version labels when reporting new scores. No leaderboard rank or official certification is claimed before maintainer review.
