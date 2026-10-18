# 基准筛选：首选 BrowseComp-Plus，EnterpriseRAG-Bench 次之

**结论：先打 BrowseComp-Plus，EnterpriseRAG-Bench 保留为第二目标；HotpotQA distractor 退出头条，FinanceBench 降为领域佐证。** 两个保留项都有正在更新、接受外部提交的官方榜单，且都直接考“模型自己 search/read”的检索能力，正好对应本项目的极简循环。其余候选要么没有活跃榜单，要么不考检索，要么各论文口径不一、无从比较。

## 唯一首选：BrowseComp-Plus

**两项里最有希望、也最有说服力的是 BrowseComp-Plus，先打它；Enterprise 排第二。**

| | BrowseComp-Plus | EnterpriseRAG-Bench |
| --- | --- | --- |
| 拿第一要过的线 | 总榜 95.18（8 agent、平均 209 次搜索）；单 agent 最好 90.48 | CDL 官方 overall 90.91 |
| 我们离这条线多远 | 还没跑过。但这个榜主要拼模型：同样用 BM25 标准设置，GPT-5 57.6、Opus 4 15.5；GPT-5.5 做到 90.48。榜上还没有 GPT-5.5 以后的模型配单 agent＋get_doc 的成绩，sol 正好补这个空白 | 本地口径 overall 79.16，CDL 86.42，落后 7.3 pp；问题出在检索，需要改系统 |
| 夺第一的把握（估计） | 单 agent 第一（> 90.48）约 60%；总榜第一（> 95.18）约 20%，因为接近标注上限：gpt-4.1 拿到全部证据时也只有 93.49 | 官方第一约 10%；挤进前三（> 82.74）约 50% |
| 说服力 | ACL 2026 主会，学界加 AI21、Mixedbread、LightOn 等公司在刷；语料固定，提交时附逐题结果，别人可以复核 | 业界相关性强，但只有企业 RAG 圈子认 |

主要风险是会被质疑“只是换了个更强的模型”。所以提交时同时报同一个 sol 在官方标准设置（只有 search，返回 top-5）下的成绩，用来证明 search/read 循环和检索器本身的贡献。

**决策门槛：先跑 30 题试点。** 试点 ≥ 90% 就跑全量 830 题；85–90% 先改检索；< 85% 就回头主攻 Enterprise。

## 筛选标准

按项目三个目标定：**主流**（论文/业界在用、榜单近期有新提交）、**考检索**（固定语料、端到端、不直接给金标准段落）、**可比**（官方评分协议＋可提交）、**可行**（本机能建库、请求预算可控）、**能争第一**、**求职相关**。

## 候选评估

| 基准 | 主流 | 考检索 | 可比 | 可行 | 能争第一 | 判断 |
| --- | --- | --- | --- | --- | --- | --- |
| **EnterpriseRAG-Bench** | ◐ 学术新（arXiv 2605.05253），业界活跃：29 个参赛系统，含 Mixedbread、多家 RAG 创业公司，榜单 2026-10-12 仍在更新 | ● 51 万文档 | ● 官方脚本＋GPT-5.4 裁判，邮件提交、Onyx 复核 | ● 已建库 | ◐ CDL overall 90.91；本地口径我们落后约 5.5 pp | **保留，主攻** |
| **BrowseComp-Plus** | ● ACL 2026 主会；84 条榜单记录，最近一条 2026-09-19；AI21、Mixedbread、LightOn 等在刷 | ● 固定 10 万网页，单独计 evidence recall | ● 官方评测脚本；Qwen3-32B 裁判；邮件提交含逐题结果 | ● 语料 1.76 GB，官方提供预建索引 | ◐ 见下文 | **新增** |
| FinanceBench | ◐ 业界知名 | ● 本地 PDF 库 | ○ 公开仅 150 题样本，无活跃官方榜 | ● 已完成 | — 无榜可争 | 降为领域佐证 |
| HotpotQA distractor | ● 名字响 | ○ 每题直接给 10 段含金标准段落，不考检索 | ◐ 官方榜多年无更新 | ● | — | **退出头条** |
| HotpotQA fullwiki / MuSiQue / 2Wiki（Search-R1、FlashRAG 设置） | ● 论文最常用 | ● | ○ 语料、切分、指标（EM/F1/LLM 判）各论文不同；对手多为训练过的小模型 | ◐ 需 Wikipedia 索引 | ○ 没有公认的第一 | 不做 |
| FRAMES | ● 常被引用 | ● | ○ 无榜单；依赖 Wikipedia 版本 | ◐ | ○ | 不做 |
| CRAG | ◐ KDD Cup 2024 | ◐ 模拟 API | ○ 比赛已结束，2026 无活跃榜 | ◐ | ○ | 不做 |
| BrowseComp（实时网页） | ● 前沿模型常报 | ◐ 考网页搜索，不是自有语料 | ○ 依赖外部搜索 API，不可复现 | ○ | ○ | 不做 |
| BEIR / MTEB / BRIGHT | ● | ◐ 只评检索器，不评端到端回答 | ● | ● | — | 不做；以后单独评检索器时再考虑 |

## BrowseComp-Plus：目标和代价

榜单（[results CSV](https://huggingface.co/datasets/Tevatron/BrowseComp-Plus-results)，数据集最后更新 2026-09-24，10-14 拉取）里和我们最接近的是“search＋get_doc”设置，相当于本项目的 search/read：

| 系统 | 准确率 | 检索召回 | 平均搜索次数 |
| --- | ---: | ---: | ---: |
| AI21：GPT-5，最多 8 个 agent 协同 | **95.18** | 93.17 | 209.3 |
| Sail：GLM-5.1，多 agent | 90.72 | 84.31 | 12.5 |
| Mixedbread：GPT-5＋Mixedbread Search v3＋get_doc | 90.48 | 83.03 | 11.5 |
| GPT-5.5＋BM25＋首轮查询扩展 | 90.48 | 96.58 | 33.7 |
| LightOn：GPT-5＋Reason-ModernColBERT＋get_doc | 87.59 | 81.55 | 13.3 |
| 官方基线：GPT-5＋Qwen3-Embed-8B | 71.69 | 78.98 | 21.7 |

- **能争什么：** 第一名靠 8 个 agent、平均 209 次搜索，这和本项目“极简单循环”的主张冲突，不值得去追。现实目标是**单 agent search/read 设置达到 90+**。GPT-5.5 配纯 BM25 已经做到 90.48，所以 sol 加我们现有的 BM25/混合检索有希望，把握中等。
- **代价估算：** 830 题，按每题约 15 次工具调用算，大约需要 1.3 万次 sol 生成请求，加 830 次裁判。先跑 30 题试点（约 500 次请求），用来校准单题成本和准确率。
- **待定的阻塞点：** 官方裁判是 Qwen3-32B，开源模型，需要本地或第三方推理；榜单另有一列是 GPT-4.1 裁判的成绩。用 sol 当裁判不符合提交口径。
- **没有 dev 集：** 830 题全部是测试题。调参只能在一个固定的小子集上做，最终报全量成绩，并公开调参用了哪些题。

## EnterpriseRAG-Bench 的新情况

官方榜单（[CSV](https://huggingface.co/spaces/onyx-dot-app/EnterpriseRAG-Bench-Leaderboard/resolve/main/data/final_display_data/leaderboard.csv)，2026-10-12 更新）：CDL 90.91，Mixedbread（+Opus 5）86.58，Neurond 82.74，之后有 6 家在 76–80。提交流程是：跑完 500 题后发邮件，开源系统附可复跑脚本，由 Onyx 复核。Codos 另外声称在 42k 文件子集上拿到 94%，但不是全量语料，也不在官方榜上，不可比。

## 下一步

1. 改 `AGENTS.md` 目标 2：从“EnterpriseRAG-Bench SOTA”改为“EnterpriseRAG-Bench 冲官方第一＋BrowseComp-Plus 单 agent 设置同口径可比”。
2. 先定 BrowseComp-Plus 的裁判方案，再下载语料和预建索引，接上现有 search/read，跑 30 题试点。
3. Enterprise 按上一轮离线诊断，先做零成本的检索回放。

来源：[BrowseComp-Plus 论文（ACL 2026）](https://aclanthology.org/2026.acl-long.1023/)、[BrowseComp-Plus 榜单 Space](https://huggingface.co/spaces/Tevatron/BrowseComp-Plus)、[EnterpriseRAG-Bench 页面](https://onyx.app/enterpriserag-bench)、[EnterpriseRAG-Bench 论文](https://arxiv.org/abs/2605.05253)、[FRAMES](https://arxiv.org/abs/2409.12941)、[CRAG](https://arxiv.org/abs/2406.04744)、[FinanceBench（Patronus）](https://docs.patronus.ai/docs/research_and_differentiators/financebench)、[Codos Aethos 声明](https://www.codos.ai/blog/aethos-enterpriserag-benchmark)。各榜单数字于 2026-10-14 拉取。
