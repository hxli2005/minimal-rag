# 任务 2026-10-15：BrowseComp-Plus 30 题试点

你是执行者。先读 AGENTS.md 并遵守。用户不在，不要停下来提问：按本文件判断，每个决定和理由各写一行到 `experiments/browsecomp-pilot-20261015/notes.md`，然后继续。中断后重新开始时，先读 `notes.md` 和输出目录，从上次停下的地方接着做。

## 目标

回答一个问题：**sol 加本项目的极简 search/read 循环，在 BrowseComp-Plus 上大约能拿多少分？** 用这个答案决定要不要投入全量 830 题。选这个基准的理由见 `experiments/reports/benchmark-selection-20261014.md`。

决策门槛（报告第一句必须给出落在哪一档）：

| 试点准确率（Qwen3-32B 裁判） | 结论 |
| --- | --- |
| ≥ 90% | 建议跑全量 830 题 |
| 85%–<90% | 先分析错题；只有证据表明检索不足时才改检索，再试 |
| < 85% | 倾向回头主攻 EnterpriseRAG-Bench，先排除装配、裁判或预算问题 |

只有 30 题，必须报告 Wilson 95% 区间；例如 27/30=90% 的区间仍为 74.4%–96.5%，26/30 与 27/30 仅差一题。以上门槛只作投入决策的辅助，不凭一题差异或点估计直接认定系统优劣；最终建议必须结合证据召回、错题分类、失败情况与成本。低分不能直接归因为检索。这个试点只用来定方向，不能作为成绩。

## 已核实的事实（不用重新调查）

- **官方仓库：** `texttron/BrowseComp-Plus`，main 分支 `046949032b0328319cc9a02663a759ec601d9402`（2026-05-28）。用 `https://codeload.github.com/texttron/BrowseComp-Plus/tar.gz/046949032b0328319cc9a02663a759ec601d9402` 下载，解压到 `data/benchmarks/browsecomp-plus/official/`。只读取，不安装它的依赖，也不运行它的代码。
- **官方评测：** 入口是 `scripts_evaluation/evaluate_run.py`。裁判是 Qwen3-32B，经 vLLM 运行，参数为 temperature 0.7、top_p 0.8、max_tokens 4096、`enable_thinking=False`，评分提示是该文件里的 `GRADER_TEMPLATE`。准确率按 `correct: yes` 计；召回率是“agent 检索到过的全部 docid”对 `topics-qrels/qrel_evidence.txt` 的覆盖率。注意 `search_agent/prompts.py` 里另有一份措辞略有不同的 GRADER_TEMPLATE，**以 evaluate_run.py 里的为准**。
- **每题输出格式（官方）：** 每题一个 JSON，字段为 `query_id`、`tool_call_counts`（dict）、`status`（成功写 `"completed"`）、`retrieved_docids`（所有检索到过的 docid 的并集）、`result: [{"type": "output_text", "output": 最终回答}]`。
- **官方 agent 提示：** `search_agent/prompts.py` 里的 `QUERY_TEMPLATE`。它要求回答依次包含 Explanation（引用写成 `[docid]`）、Exact Answer、Confidence 三段。
- **数据：** 题目在 HF 数据集 `Tevatron/browsecomp-plus` 的 test split，共 830 题，内容经过加密。解密方法：密钥是 canary 字符串的 SHA-256，对 base64 解码后的密文逐字节异或，`query_id` 字段不加密。canary 和解密细节见官方 `scripts_build_index/decrypt_dataset.py`，用 pyarrow 直接读 parquet，不装 `datasets`。语料在 `Tevatron/browsecomp-plus-corpus`，7 个 parquet 文件，约 1.76 GB，`.venv` 里已有 `huggingface_hub` 和 `pyarrow`。
- **预建索引：** `Tevatron/browsecomp-plus-indexes` 里有 bm25（Lucene，需要 Java 21 和 pyserini）和 qwen3-embedding 0.6b/4b/8b（FAISS）。**试点不下载这些索引**，改用项目自己的 SQLite FTS5 BM25（`IndexedKnowledgeBase.build`）。理由：和项目栈一致；BM25 只是基线检索器；GPT-5.5 配 BM25 在榜上已有 90.48。dense 检索留到试点之后，看结果再决定。
- **本机：** MacBook Pro，Apple M5，24 GB 内存，没有 CUDA，所以 vLLM 跑不了。Qwen3-32B bf16 需要约 64 GB，放不下；MLX 4-bit 约 18 GB，在 24 GB 上很勉强。
- **榜单参照线**（`Tevatron/BrowseComp-Plus-results`，2026-09-24 更新）：GPT-5＋BM25 官方标准设置 57.59；GPT-5.5＋BM25＋首轮查询扩展 90.48；GPT-5＋Mixedbread＋get_doc 90.48；单 agent 最好 90.48；总榜第一 95.18（AI21，8 个 agent，平均 209 次搜索）。

## 裁判方案

**主裁判：经 OpenRouter 调用 Qwen3-32B，和官方是同一个模型。** 不在本机跑，也不用 sol 当主裁判。

- **接口：** `POST https://openrouter.ai/api/v1/chat/completions`，`model="qwen/qwen3-32b"`。裁判脚本只从环境变量 `OPENROUTER_API_KEY` 读取密钥；用户已指定由启动器从 macOS 钥匙串 `xiaobu.openrouter.api-key`（账户 `lhx`）读取并注入子进程环境，不写进文件、日志或 notes。用标准库 `urllib` 或 `.venv` 里已有的 HTTP 库，不新增依赖。
- **请求参数（对齐官方）：** user 消息为官方 GRADER_TEMPLATE 原样填入，不追加 `/no_think`。采样参数为 `temperature=0.7`、`top_p=0.8`、`top_k=20`、`max_tokens=4096`；top_k 是实际官方源码默认值，执行前核对后补齐。
- **关闭思考的传递修复：** 在 OpenRouter/DeepInfra 路由中，单独的 `reasoning.enabled=false`、`effort=none` 及透传模板开关实测均未关闭思考；仅凭响应不能确定参数在哪一层未生效。改用 Qwen 官方 `enable_thinking=False` 模板产生的 assistant 前缀 `<think>\n\n</think>\n\n`，通过最后一条 assistant 消息和 `continue_final_message=true` 传递，同时保留 `reasoning={"enabled": false, "exclude": false}`。本地模板渲染已验证序列化等价，实际两题均返回0个reasoning token；评分user文本不变，也不靠隐藏reasoning通过检查。见 `judge-template-equivalence.json`。
- **供应商锁定：** `provider={"order": ["DeepInfra"], "allow_fallbacks": false, "data_collection": "deny"}`。
  - 锁定是为了让每题用同一个供应商、同一种量化。
  - `data_collection: deny` 是为了不让第三方保留或训练这些评测题。
- **已知差异，报告里要如实写出：** 2026-10-14 查询时，OpenRouter 上 Qwen3-32B 只有 fp8 量化的供应商（DeepInfra、Tenstorrent、SiliconFlow），官方用的是 vLLM 加原始权重。所以报告口径写“Qwen3-32B（OpenRouter/DeepInfra fp8）裁判，与官方同一模型，量化不同”。
- **冒烟校验：** 两题打分后检查：实际供应商是 DeepInfra；响应没有 reasoning 内容，且返回的 reasoning token 计数为0；`parse_judge_response` 能解析出 `correct`。有任一项不满足，就停下记录原因，不要改评分提示去迁就。
- **重试规则：** 解析失败或 HTTP 出错时，同一题在冻结参数下最多重试 2 次，每次都记录。用户授权的装配修复可另建参数版本，旧失败尝试全部保留并计入调用与费用；有效判决不得重跑。最终用第一次通过协议校验的判决，不挑结果。
- **费用：** 按 DeepInfra 价格（输入 $0.08/M、输出 $0.28/M），30 题不到 1 美分，全量 830 题约 0.3 美元。
- **没有密钥时的退路：** 如果 `OPENROUTER_API_KEY` 不存在，退回 sol 代理裁判。用 `gpt-6.1-sol`、effort medium、同一个官方模板，调用方式是 `run_agent(model, {}, 评分提示, max_turns=1)`。报告每个数字都标注“sol 代理裁判，不能和榜单比较”，并在 notes 记一行原因。

本机（M5，24 GB）跑不了 vLLM，MLX 4-bit 也很勉强，所以不做本地 Qwen3 探针。

## 实现

一次性脚本全部放在 `experiments/browsecomp-pilot-20261015/`，`src/minimal_rag/` 一行都不改。

1. **`prepare.py`**
   - 下载官方仓库、题目和语料，全部放到 `data/benchmarks/browsecomp-plus/`（已 gitignore）。
   - 解密题目，写成 `queries.jsonl`。
   - 用 `IndexedKnowledgeBase.build` 建 `corpus.sqlite`，字段映射为：id=docid，title=url（没有 url 时取正文首行前 120 个字符），text=正文，source=url。
   - 构建前先打印语料字段名，按实际字段映射。不要把题目或答案写进索引。
   - 记录文档数、chunk 数和耗时。
2. **`run.py`**
   - 每题调用一次 `run_agent(CodexModel(..., model="gpt-6.1-sol", effort="high", timeout=180, max_calls=30), indexed_tools(kb), 问题, max_turns=30)`。
   - `kb = IndexedKnowledgeBase(path, excerpt_chars=2000)`。理由：一次搜索返回 5 条 4000 字符的摘录太长，30 轮下来上下文膨胀太快；read 仍可以分页读全文。
   - 问题文本用官方 `QUERY_TEMPLATE`，只把其中的 `get_document` 改成 `read`，其他一字不改。
   - 每题保存完整轨迹，另外导出官方格式 JSON：`retrieved_docids` 取 search 命中和 read 过的所有 docid 的并集；`tool_call_counts` 按工具名计数；`status` 在 run_agent 返回 finished 时写 `completed`，否则写失败原因。
   - 单题总超时 1800 秒，并发 3。支持断点续跑：仅有完整终态结果的题直接跳过；有轨迹但无终态的题属于中断，先保留原始尝试、计入已发请求，在 notes 记录恢复原因后才重跑。完整回答或预算用尽不得因结果不好而重跑；限流等基础设施失败可记录原因后恢复，保留全部尝试并使用第一次完成的回答。
   - 固定 30 题为主报告分母：失败、超时、预算用尽和未运行题不能静默剔除，单列状态且在保守主准确率中计为未答对；已完成题准确率只作补充并明确分母。裁判失败标为未判定，不冒充答案错误；仍以未答对计入保守主准确率，待首次有效判决后更新。冒烟阶段单独报告 2 题装配结果，不套用 30 题决策门槛。
3. **`judge.py`**：按上面的裁判方案打分，输出逐题判决，以及 `summary.json`。

## 运行顺序与预算

1. **样本：** 830 个 query_id 排序后，用 `random.Random(20261015).sample(ids, 30)` 抽 30 题，把 ID 列表写进 `sample.json` 并 commit。这 30 题以后就是“开发题”：将来全量成绩要公开说明这一点，不能把它们当成没见过的题。
2. **冒烟：** 先跑样本中的前 2 题，确认工具调用、官方格式导出和裁判解析都通，再跑剩下的 28 题。冒烟只修装配问题，不改提示和检索。
3. **预算：** 试点生成最多 1000 次 sol 请求，失败和重试也算；OpenRouter 裁判调用另计，并报告次数和费用。每次轮询都从轨迹汇总一次请求数，达到上限即停止发起新请求；按固定 30 题分母报告，并另列已完成题结果。应用层可观测请求与 SDK 内部不可见重试的口径必须区分，不能声称轨迹计数是底层 HTTP 请求的严格上限。
4. **限流：** 遇到 429 或 usage limit 就停掉，把并发减半后续跑，记一行到 notes。
5. **长任务：** 用 nohup 后台运行（之前的经验：用 `Popen(start_new_session=True)` 才能真正脱离终端），每 5–10 分钟轮询一次，直到全部完成。
6. **权限：** 如果 Codex SDK 被沙箱或权限拦住，立即停下并记录原因，不要绕过，也不要改 `CODEX_HOME`。

## 报告

`experiments/reports/browsecomp-pilot-20261015.md`，一屏以内：

- **第一句：** 准确率、Wilson 95% 区间，以及落在决策门槛的哪一档。
- **一张表：** 准确率、召回率（官方定义）、平均搜索次数、平均 read 次数、平均生成请求数、输入 token、sol 请求总数，和上面的榜单参照线放在一起比。
- **最多 5 条要点**，必须包括：
  - 错题分类：证据从没检索到 / 检索到了但答错 / 预算用尽；
  - 裁判实际使用的供应商、量化和费用，以及冒烟校验的结果；
  - 下一步建议。
- **口径一行：** Qwen3-32B（OpenRouter/DeepInfra fp8）裁判、30 题开发样本、非官方成绩。

## 规矩

- **数据保密：** 解密后的题目和答案只能放在 `data/` 下。不要 commit，报告里也不要引用任何一道题的原文或答案（数据集的 canary 明确要求不能以明文出现在网上）。报告只用 query_id。
- **git：** 在 `codex/standalone-rag` 分支上提交，不推送，不碰 main。第一次 commit 时，把尚未提交的 `experiments/reports/benchmark-selection-20261014.md` 一起带上。
- **commit 节点：** 每完成一个阶段 commit 一次：脚本加 `sample.json`、冒烟结果、试点摘要、报告。
- **不新建测试套件**，不改核心循环、`indexed_rag.py` 和 `codex_adapter.py`。
- **issues：** 发现但不阻塞的问题，记入 `docs/issues.md`，一行一个。
- **收尾：** 在终端输出不超过 10 行的总结。
