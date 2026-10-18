# Use Minimal RAG with your own documents

Start with local retrieval, then choose a live model backend. The commands below work in an empty directory on macOS or Linux with Python 3.12+.

## 1. Try the offline demo

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install https://github.com/hxli2005/minimal-rag/archive/refs/heads/main.zip
.venv/bin/python -m minimal_rag demo
```

The result contains `mode: offline_scripted_demo`, a final answer, source IDs, and the complete tool events. It builds a real SQLite index and executes real search/read calls with scripted model decisions. It makes no model requests and needs no API key.

## 2. Index two example documents

Each line in a JSONL file is one document with a stable `id`, `title`, `text`, and `source`. Replace this fictional example with your own text when you are ready.

```bash
cat > documents.jsonl <<'EOF'
{"id":"returns","title":"Example return policy","text":"Standard items can be returned within 30 days. Personalized items cannot be returned unless defective.","source":"example://returns"}
{"id":"mug","title":"Example product MUG-42","text":"MUG-42 is a personalized mug. Customers choose their name before ordering.","source":"example://catalog"}
EOF

.venv/bin/python -m minimal_rag index documents.jsonl --output my-corpus.sqlite
.venv/bin/python -m minimal_rag search --index my-corpus.sqlite --query personalized
```

This is still entirely local. Search results include original text, document IDs, character offsets, and document lengths. The builder refuses to overwrite an existing index; choose a new output filename when rebuilding.

## 3. Ask a live model through the API backend

Install the API dependency:

```bash
.venv/bin/python -m pip install 'minimal-rag[api] @ https://github.com/hxli2005/minimal-rag/archive/refs/heads/main.zip'
```

Set your provider's API key and an actual model ID available at that endpoint:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
export RAG_API_MODEL="YOUR_PROVIDER_MODEL_ID"
```

The default endpoint is `https://api.openai.com/v1`. To use another compatible provider, set `OPENAI_BASE_URL` to its base URL. The endpoint must support the Responses API, native function calls, structured answers, and the reasoning settings you select. A Chat Completions-only endpoint is not compatible.

The next command sends the question and retrieved source text to your configured provider and consumes its model allowance or API credits:

```bash
.venv/bin/python -m minimal_rag ask --live --backend api \
  --model "$RAG_API_MODEL" \
  --index my-corpus.sqlite \
  --question "Can I return MUG-42 if I changed my mind and it is not defective?" \
  --output answer-trace.json
```

The API defaults are reasoning effort `high` and verbosity `low`. The model decides when to stop; `--max-turns` defaults to `0` (no call-count cap). You can add `--max-turns N` for an explicit hard stop. Reaching that stop returns `budget_exhausted` rather than forcing a final answer.

`ask` also supports the Codex backend for an existing ChatGPT login; see the [runbook](runbook.md). The example above chooses the API backend explicitly.

## 4. Inspect what actually happened

Open `answer-trace.json` and check:

| Field | What it records |
| --- | --- |
| `events` | Model tool calls and the exact tool results returned to it |
| `model_runs` | Model request records and available token usage |
| `result.answer` | The model's final answer, if one was produced |
| `result.sources` | Source IDs to compare with your documents |
| `result.status` | Whether the model finished, hit a limit, timed out, or failed |

`finished` means the model stopped; it is not a correctness certificate. Compare the answer with the cited source text. This small example is not a benchmark reproduction; use the pinned configurations in [REPRODUCE.md](../REPRODUCE.md) for the published results.

## 中文速读

1. 先运行前三条命令：建立虚拟环境、从公开仓库安装、执行离线演示。不需要先克隆仓库，也不会调用模型。
2. 将资料整理成 JSONL，每行一篇文档，字段为 `id`、`title`、`text`、`source`。用 `index` 建库，再用 `search` 检查原文是否能被找到。
3. 真实问答需安装 `[api]` 依赖，设置自己的 `OPENAI_API_KEY` 和提供商实际支持的模型 ID，然后显式传入 `--live --backend api`。检索到的原文会发送到所选提供商并消耗用量。
4. 默认推理为 `high`、回答详细度为 `low`，模型调用次数默认不设上限；可自行使用 `--max-turns N` 硬停止。检查 `answer-trace.json` 中的工具轨迹、答案和来源，不能把 `finished` 当作答对。

本页离线安装、建库和搜索已验证；没有为编写此指南新增付费模型请求。更多运行参数见[运行指南](runbook.md)，实验归属见[复现指南](../REPRODUCE.md)。
