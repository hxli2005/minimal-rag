# 任务 2026-10-15：BrowseComp-Plus 对照组、降本与全量 830 题

你是执行者。先读 AGENTS.md 并遵守。用户不在，不要停下来提问：按本文件判断，每个决定和理由各写一行到 `experiments/browsecomp-full-20261015/notes.md`，然后继续。中断后重新开始时，先读 `notes.md` 和输出目录，从上次停下的地方接着做。

## 目标

试点 30 题拿到 93.3%，但这个分数主要说明 sol 强，说明不了本项目的贡献。本轮要拿到三样东西：

1. **本项目贡献了多少：** 同一个 sol、同一个检索库，只把工具换成官方“标准设置”（只能 search，每次返回前 5 篇文档开头 512 个 token，不能读全文）。两组的准确率差，就是本项目工具和循环设计的贡献。
2. **又准又省：** 试点每题输入约 59.1 万 token，其中 51.5 万是缓存命中，未缓存约 7.6 万。主要开销来自每题约 21 次 search，每次 5 条 2000 字符的摘录。本轮在开发题上验证一个降本版本：准确率不降，就用它跑全量，不再另跑一遍。
3. **全量成绩和提交文件：** 冻结的配置跑完 830 题，产出符合官方格式、可以提交维护方核实的结果文件。

**为什么先降本再跑全量：** 如果先用现在的配置跑全量，降本之后要拿“又准又省”的结论，就得再跑一遍全量。先在开发题上定好配置，全量只跑一次，也省额度。

## 共同的冻结部分（不许改）

- **生成：** 和试点相同，以 `experiments/browsecomp-pilot-20261015/run.py` 写出的 `config.json` 为准。参数为 gpt-6.1-sol、effort high、request_timeout 180、case_timeout 1800、max_turns/max_calls 30。语料用 `data/benchmarks/browsecomp-plus/corpus.sqlite`，不重建。
- **裁判：** 和试点相同，以 `experiments/browsecomp-pilot-20261015/judge.py` 为准。
  - Qwen3-32B，经 OpenRouter 调用，锁定 DeepInfra fp8，`data_collection: deny`。
  - 参数 temperature 0.7、top_p 0.8、top_k 20、max_tokens 4096；关闭思考的方式是在回复开头预先填入空的 think 段；评分提示用官方 GRADER_TEMPLATE。
  - 每题最多 3 次尝试，用第一次有效的判决。逐条校验：提供方是 DeepInfra，推理 token 为 0。
  - 密钥只从环境变量 `OPENROUTER_API_KEY` 读取。
- **代码：**
  - 新脚本放在 `experiments/browsecomp-full-20261015/`，从试点脚本 import，或做最小改动后复制过来。`src/` 一行不改。
  - 每组启动前，把它的 config 和试点 config 逐字段比对，只允许本文件列出的差异。

## 三个系统

| 名称 | 工具 | 唯一差异 |
| --- | --- | --- |
| **A**（试点配置） | search＋read，`excerpt_chars=2000` | 无，和试点完全相同 |
| **A-lean**（降本候选） | search＋read，`excerpt_chars=800` | 只改 search 摘录长度，read 不变，模型需要时自己去读原文 |
| **B**（官方标准设置对照） | 只有 search | 见下文 |

**A-lean 只准试这一个候选，不许多版本挑最好的。** 理由：未缓存输入大头是 search 摘录；800 字符仍然覆盖命中块的一半左右，缺的内容模型可以用 read 补。

**B 组的定义：** 对齐官方 `search_agent/openai_client.py` 的默认参数（固定版本 `0469490`，源码在 `data/benchmarks/browsecomp-plus/official/`）。

- **只有一个 `search(query)` 工具。** 返回前 5 篇不同的文档，每篇给 `docid` 和 `snippet`，snippet 是**文档全文开头**的 512 个 token。用 `Qwen/Qwen3-0.6B` 的 tokenizer 先编码，截断到 512 个 token 后再解码，和官方实现一致。
- **没有 read 工具，也不返回命中位置附近的摘录。**
- **检索：** 用 `IndexedKnowledgeBase.search(query, limit=5)` 取文档排序，再从 `documents` 表取全文截断。理由：检索库和 A 组一致，两组只差工具设计。
- **已知差异：** 官方用的是 Lucene 文档级 BM25，我们是 FTS5 分块 BM25。报告写明 B 组是“官方标准工具＋本项目检索”，不是官方 BM25 基线。
- **提示：** 官方 `QUERY_TEMPLATE_NO_GET_DOCUMENT` 原样使用。
- 工具定义写在实验目录里。

## 样本（先冻结，再运行）

所有抽样都在按字符串排序的 query_id 上进行，ID 列表写入文件，在第一次模型调用前 commit。

- **开发集 dev100：** 试点的 30 题，加上从其余 800 题里用 `random.Random(20261016).sample(ids, 70)` 抽的 70 题。写入 `sample-dev100.json`。
- **对照集 b200：** 从剩下的 730 题里用 `random.Random(20261017).sample(ids, 200)` 抽取。写入 `sample-b200.json`。
- **全量：** 830 题全部计入最终成绩。报告要写明其中 dev100 是开发样本，另外单独报一个口径：剩余 730 题的准确率。

## 运行顺序与预算

sol 请求按应用层计数（同试点），失败和重试都算。每次轮询都从账本汇总，超过上限就停掉那一阶段，按已完成的题报告。

1. **阶段 1，B 组跑试点 30 题（上限 900 次）。** 先跑 2 题冒烟，检查 snippet 长度、工具调用、官方格式导出和裁判；冒烟只修装配问题。这一步最便宜，最早能看出项目贡献有多大，跑完就 commit 摘要。
2. **阶段 2，降本验证（上限 2200 次）。**
   - A 跑 dev100 里新抽的 70 题；试点 30 题复用已有轨迹，先核对 config 并确认逐字节一致。
   - A-lean 跑 dev100 全部 100 题。
   - **判定规则（预先写死）：** A-lean 同时满足下面两条就采用，否则保留 A：
     - 每题平均未缓存输入 token 比 A 至少低 40%；
     - 和 A 配对比较，负题数减胜题数 ≤ 2（100 题内准确率最多掉 2 个点）。
   - 判定结果、两组的配对表和成本表 commit 后，**选中的配置就是最终冻结配置**，此后不再改。
3. **阶段 3，全量（上限 7500 次）：** 用冻结配置跑 dev100 以外的 730 题；dev100 复用选中配置的已有轨迹。并发 3。
   - 跑完前 100 题后，在 notes 里记录实际的平均调用次数、未缓存 token 和耗时，并估算剩余部分的消耗。
   - 估算会超过上限，就停下来，按已完成的题报告。不得为了省预算改配置。
4. **阶段 4，B 组跑 b200（上限 5000 次），并发 3。**
5. **裁判：** 每题生成完就打分，可以和生成并行。裁判调用不占 sol 预算，但次数和费用单独记录。
6. **限流和额度：**
   - 遇到 429 或 rate limit：把并发减半后续跑，记一行到 notes。
   - 遇到 usage limit（额度用完）：停掉所有阶段，保存状态，记一行到 notes。下次启动时从断点续跑，不重跑已完成的题。
7. **长任务：** 用 nohup 后台运行（之前的经验：用 `Popen(start_new_session=True)` 才能真正脱离终端），每 5–10 分钟轮询一次，直到全部完成。
8. **权限：** 如果 Codex SDK 被沙箱或权限拦住，立即停下并记录原因，不要绕过，也不要改 `CODEX_HOME`。

## 统计

- **全量（冻结配置）：**
  - 准确率和 Wilson 95% 区间；剩余 730 题的准确率单独报。
  - 官方定义的召回率：检索到过的 docid 对 `qrel_evidence` 的覆盖率。
  - 官方定义的校准误差：从裁判输出解析 confidence，用官方 `calib_err`，beta=100。
  - 每题平均的 search 次数、read 次数、生成请求数、输入 token、缓存 token、未缓存 token。
- **配对比较：** 统一按题 bootstrap 10,000 次，seed 20261015，给出 95% 区间和胜/负/平题数。
  - 冻结配置对 B：分别报 30 题、200 题和合计 230 题，比较准确率、召回率、平均搜索次数、未缓存 token。
  - A-lean 对 A：报 dev100 上的准确率差和成本差。
- **榜单定位：** 和这几条参照线放在一起：GPT-5＋BM25 标准设置 57.59；单 agent 最好 90.48；总榜第一 95.18。
  - 只有 Wilson 区间下界 > 90.48 时，才能写“单 agent 设置下高于现有最好成绩”。
  - 只有下界 > 95.18 时，才能写“高于总榜第一”。
  - 榜单提交时会报平均搜索次数，所以单独对比我们和同档系统的搜索次数。
- **错题分类：** 证据从没检索到 / 部分检索到 / 全部检索到但答错 / 预算用尽 / 基础设施失败。

## 提交文件（只准备，不发送）

按官方 README 的提交格式，生成 `data/benchmarks/browsecomp-plus/full-20261015/submission.json`：

- 字段：`LLM`、`Retriever`、`Accuracy (%)`、`Recall (%)`、`Search Calls`、`Calibration Error (%)`、`Link`、`Evaluation Date`、`per_query_metrics`。
- `LLM` 写 `gpt-6.1-sol (effort high)`；`Retriever` 写 `BM25 (SQLite FTS5, 1600-char chunks) + search/read tools`；`Link` 留空。
- 在 notes 里写一段 scaffold 说明，供用户邮件提交时附上。

**不要发送邮件，也不要上传到任何地方。** 是否提交由用户决定。`submission.json` 里没有题目和答案原文，可以 commit 一份到实验目录。

## 报告

`experiments/reports/browsecomp-full-20261015.md`，一屏以内：

- **第一句：** 全量准确率和区间；对照组测出的本项目贡献（配对差和区间）；降本结果（采用了哪个配置，每题未缓存 token 降了多少）。
- **一张主表：** 冻结配置、B 组（230 题）、A 与 A-lean（dev100）、榜单参照线。列为准确率、召回率、平均搜索次数、平均 read 次数、每题未缓存输入 token。
- **最多 5 条要点**，必须包括：
  - 榜单定位，严格按上面的区间规则写；
  - 本项目贡献有多大；
  - 降本是否成功，以及成本主要花在哪里；
  - 错题分类；
  - 下一步建议。
- **口径一行：** Qwen3-32B（OpenRouter/DeepInfra fp8）裁判、dev100 开发样本已含在全量中、B 组是官方标准工具配本项目检索、未经维护方核实。

## 规矩

- **数据保密：** 解密后的题目和答案只能放在 `data/` 下。不要 commit，报告也不要引用任何一道题的原文或答案，只用 query_id。
- **git：** 在 `codex/standalone-rag` 分支上提交，不推送，不碰 main。
- **commit 节点：** 每完成一个阶段 commit 一次：样本加脚本、B 组 30 题、降本判定、全量摘要、B 组 200 题、报告。
- **不新建测试套件**，不改 `src/`。
- **issues：** 发现但不阻塞的问题，记入 `docs/issues.md`，一行一个。
- **收尾：** 在终端输出不超过 10 行的总结。
