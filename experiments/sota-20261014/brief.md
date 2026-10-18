# 夜间任务 2026-10-14：一套冻结系统，三个基准

你是执行者。先读 AGENTS.md 并遵守。用户在睡觉，不要停下提问；按本文件判断，把每个决定和理由各写一行到 `experiments/sota-20261014/notes.md`，然后继续。中断后重新开始时，先读 `notes.md` 和各输出目录，从上次停下的地方接着做。

## 目标与判断

**今晚只跑仓库已经接好的三个基准。三者用同一核心循环、同一生成模型、同一提示；检索沿用各语料在 dev 上验证过的配置。** 不新增 MultiHop-RAG、FRAMES、CRAG 等：一晚之内要下载数据、建库、对齐裁判，这样得到的成绩不可信。

“准 SOTA”按基准分别定义，达不到就如实写：

| 基准 | 外部参照 | 历史成绩（astra v6） | 今晚准 SOTA 标准 |
| --- | --- | --- | --- |
| EnterpriseRAG-Bench test400 | CDL 86.5%（346/400） | 77.75%（311/400），配对差 −8.75，95% 区间 −12.5～−5.0 | 与 CDL 的配对差，95% 区间包含 0 |
| FinanceBench test100 | Dewey 74.0% | 81.0% | 不低于 Dewey，且配对差区间下界 ≥ −5 pp |
| HotpotQA test100（distractor） | 官方榜 answer EM/F1（查出数值并附链接；测试集不同，只作量级参照） | EM 78.0 / F1 87.1 | EM ≥ 78 且 F1 ≥ 87 |

差距最大的是 Enterprise。dev 上 cascade 栈 85/100，CDL 88/100，旧 fixed 栈 75/100。test400 的封存成绩来自 fixed 栈，cascade 从未在 test400 上跑过。今晚只有这一处改动有可能把企业差距拉进区间。

## 冻结配置

公共部分：用 `experiments/src/minimal_rag_experiments/rag_optimization_eval.py`，variant `v6`（effort high，原中文系统提示），sol 负责生成和裁判，`--max-turns 12 --timeout 180 --case-timeout 900`，裁判提示沿用 runner 内置的历史版本。核心 `src/minimal_rag/loop.py` 不改。

**和历史成绩相比只换了一处：生成模型 astra → sol。** read 仍默认 4000 字符，“读全文”不在今晚做：它未经验证，和换检索栈一起改就无法归因。

命令模板（在仓库根目录运行）：

```bash
PYTHONPATH=src:experiments/src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 \
nohup .venv/bin/python -m minimal_rag_experiments.rag_optimization_eval \
  --dataset <D> --split <S> --variant v6 --output data/benchmarks/sota-20261014/<NAME> \
  --model gpt-6.1-sol --judge-model gpt-6.1-sol --max-turns 12 --timeout 180 --case-timeout 900 \
  --concurrency <C> --retrieval hybrid --candidates 100 --rerank 200 --models <M> <额外参数> \
  > data/benchmarks/sota-20261014/<NAME>.log 2>&1 &
```

| 基准 | `<M>` | 额外参数 | 对照的历史 config |
| --- | --- | --- | --- |
| Enterprise | `models/hybrid-rag/models-bge-m3-cascade100.json`（BGE-small＋BM25 RRF，MiniLM 预筛 100 → bge-reranker-v2-m3） | `--preview-count 15` | `data/benchmarks/optimization-20261012/enterprise-dev100-bge-cascade-preview/config.json` |
| FinanceBench | `models/hybrid-rag/models-minilm.json` | 无 | `data/benchmarks/optimization-20261011/holdout/financebench/config.json` |
| HotpotQA | `models/hybrid-rag/models-minilm.json` | `--short-style extractive` | `data/benchmarks/optimization-20261011/holdout/hotpotqa/config.json` |

test 集额外加 `--known-regression`；崩溃后续跑加 `--resume`。历史 config 里的绝对路径指向旧目录 `…/ChatGPT/即兴学习/…`，对应本仓库的 `data/benchmarks/` 和 `models/hybrid-rag/`。逐项核对新 `config.json` 与历史 config 的 retrieval 段（模型 id／revision、candidates、rerank、preview、prefilter）、提示和裁判提示，只允许 model 和路径不同。有其他差异时：能改回就改回，改不回就停下这一项并记录。

已知情况：上一次中断的冒烟里，Enterprise dev 和 HotpotQA dev 各跑通 2 题，sol 能生成、HotpotQA 能算 EM，Enterprise 检索召回 1.0；FinanceBench 还没冒烟。正式运行前补跑冒烟：Enterprise 3 题、FinanceBench 2 题、HotpotQA 2 题，都取 dev 文件前几题（`--limit`），三项都要完成裁判或 EM。冒烟时只修装配问题，不改检索和提示。冒烟通过后，把三份完整命令和核对结果写入 `experiments/sota-20261014/frozen.json` 并 commit，此后不再调参。

## 运行顺序与预算

sol 请求数＝各题生成次数之和＋裁判次数；失败和重试也要计入。每次轮询时从输出目录汇总请求数。超过上限就停掉该项，按已完成的题报告，不补跑。不另写预算包装层。

1. **E-dev**：Enterprise dev100，并发 3，上限 900 次请求。同时启动 **H-test**：HotpotQA test100，并发 2，上限 600 次（无裁判）。
2. H-test 完成后，启动 **F-test**：FinanceBench test100，并发 2，上限 900 次。
3. **门槛**：E-dev ≥ 82/100，且基础设施失败 ≤ 5 题，才启动 **E-test**：Enterprise test400，并发 3，上限 3200 次。82 不是质量线，只用来挡住装配错误；单次 100 题的波动约 ±4，所以不拿 85 当硬线。
   E-dev < 82 时不跑 E-test：用保存的轨迹逐题对比 sol 与历史 astra cascade 的错题，判断是模型差异、装配错误还是随机波动，写进报告。
4. 遇到限流（429 / rate limit / usage limit）：停掉该项，并发减半后 `--resume`，记一行到 notes。单题超时计为未完成，保留在分母中。

长任务一律用 nohup 后台运行，每 5–10 分钟轮询一次，直到全部完成，不要提前结束。

## 配对与统计

用现有的 `experiments/scripts/compare_*` 或 runner 自带的配对逻辑：按题目 bootstrap 10,000 次，seed 20261012。

- Enterprise dev100：对比历史 astra cascade（`data/benchmarks/optimization-20261012/enterprise-dev100-bge-cascade-preview`）和 CDL（`data/benchmarks/optimization-20261011/judged/enterprise-dev100-cdl-lean`），按题型拆分。
- Enterprise test400：对比 CDL 和 v6，报告正确率、完整度、overall、document recall、95% 区间、胜／负／平，并按题型拆分。CDL／Dewey 的已判分逐题结果在 `data/benchmarks/optimization-20261011/holdout/public-comparisons/{cdl,dewey}`，历史 v6 在同级 `holdout/{enterprise,financebench,hotpotqa}`。
- FinanceBench：对比 Dewey 和历史 v6。HotpotQA：对比历史 v6 的逐题 EM/F1。
- 每项都报告平均生成次数、输入 token 和 sol 请求总数。

## 规矩

不改 gold、历史成绩和证据。在 `codex/standalone-rag` 分支上提交，不推送，不碰 main。不新建测试套件；一次性脚本放在 `experiments/sota-20261014/`。每完成一个阶段 commit 一次（frozen.json、各项结果摘要 JSON、报告）。原始结果留在 `data/`（已 gitignore），仓库只提交摘要。

## 交付

1. `experiments/reports/sota-20261014.md`，一屏以内：首句给结论（几个基准达到准 SOTA）；一张表三行，列为基准、历史 v6、今晚 sol、外部参照、配对差与区间、是否达标；最多 5 条要点。用一行说明口径：本地 sol 裁判、已知回归集、非官方榜单；正式 SOTA 还需要官方协议和全部题目。
2. `docs/scoreboard.md` 新增一节“统一 sol 冻结系统”；README 记分板加一行或一列，不覆盖历史 v6 成绩。
3. 发现但不阻塞的问题记入 `docs/issues.md`，一行一个。
4. 最后在终端输出不超过 10 行的总结。
