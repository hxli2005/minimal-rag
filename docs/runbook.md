# 运行指南

Python 3.12＋。核心命令只有 `demo`、`index`、`search`、`ask`。`ask` 和所有 benchmark 的当前入口共用纯 BM25 的 `IndexedKnowledgeBase`、`search/read` 工具、模型适配器和核心循环。

## 冻结与官方提交（2026-10-17）

用户决定结束调优，冻结当前无 Grep 主路径。源码包：[source.tar.gz](../evidence/frozen-bm25-20261017/source.tar.gz)；版本、参数、SHA256 和文件清单：[manifest.json](../evidence/frozen-bm25-20261017/manifest.json)。这是工作区源码快照，包含尚未提交的改动，不能仅用清单中的 Git HEAD 复现；源码包不含语料、索引、权重或运行结果。实验源码一并存档不代表启用这些候选。此后只补复现与提交材料，停止优化实验。

冻结模型为 gpt-6.1-sol high，工具为 search/read。Enterprise 摘录4000字符、12轮、单调用180秒、每题900秒；BrowseComp 摘录2000字符、30轮、单调用180秒、每题1800秒。当前 indexed_rag.py 指纹与309/400那轮记录一致；这个核查不等于对全部环境重新验证。当前 Enterprise 77.25% 是本地 test400 结果，不是官方分数；历史81%和 BrowseComp 93.01%不移植到此版本。

### 完成标准与顺序

完成标准：维护方明确复核认可或官方榜单出现可公开核验的系统条目，条目能关联真实模型、版本与复现材料。自行运行官方脚本、发出邮件、维护方收到邮件均不等同于完成。

1. **优先整理历史 BrowseComp 93.01% 的独立提交包。** 已有830题官方格式结果及Qwen3-32B BF16本地判分；本次离线核对 submission.json 有830个唯一题号、772题正确。官方README要求将评分摘要JSON发送给 s42chen@uwaterloo.ca，并填写检索方案和项目链接。现有提交JSON的 Link 为空，当前仓库没有配置 Git remote；先准备可访问的项目/复现页面、准确的历史运行说明和模型访问说明。历史 scaffold 中的选型叙述应改清：93.01%对应原A/v0，后续确有用户采用v3的决定，不能暗示A一直是主版本。披露开发集接触及多轮实验，保留4道原生成失败。历史源码复现完整性仍需核验，不能用本次冻结源码替代。维护方是否接受该模型访问方式与历史运行由其确认。
2. **Enterprise 保持为当前冻结版本的主目标。** 官方主集500题，提交格式为 question_id、answer、document_ids；现有309/400及此前历史500题导出不能直接冒充本冻结版本的官方成绩。先离线核对现有当前版本dev100与test400的配置及题号覆盖，仅在证明配置一致后合并复用；缺失部分再明确必要生成量。对齐引用UUID与物理文档版本，补全官方评测流程或按维护方要求验证，不继续使用本地代理裁判来宣称官方分数。
3. **优先确认验证方式，避免无效花费。** Enterprise官方要求联系 joachim@onyx.app，开源提交提供复现指南，闭源提交提供可验证的sandbox/endpoint。与维护方确认生成模型可访问性、指定裁判/环境、既有答案复用和已接触题集的披露要求，再执行必要的正式评测。此阶段不要求先冲到某个分数。

本次只冻结、核对公开规则和整理路线；没有新增模型生成/判分、租用GPU、公开上传或发送邮件。对外邮件内容及发布材料准备完整后再交用户确认发送。

2026-10-17核对的官方来源：[Enterprise提交要求](https://github.com/onyx-dot-app/EnterpriseRAG-Bench#leaderboard)、[Enterprise答案格式与评分](https://github.com/onyx-dot-app/EnterpriseRAG-Bench/blob/main/answer_evaluation/README.md)、[BrowseComp提交要求](https://github.com/texttron/BrowseComp-Plus#submitting-to-the-leaderboard)。入口可能更新，实际提交前再核对。

## 安装与离线演示

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m minimal_rag demo
```

`demo` 从 `fixtures/retail.jsonl` 建 SQLite 索引，回放预先写好的模型决策，生成调用为 0，无需账号。

## 自己的资料与真实问答

JSONL 每行包含 `id`、`title`、`text`、`source`。SQLite FTS5 索引保留原文和字符位置；构建器拒绝覆盖已有文件。

```bash
.venv/bin/python -m minimal_rag index examples/documents.jsonl --output data/example.sqlite
.venv/bin/python -m minimal_rag search --index data/example.sqlite --query "connection timeout"
.venv/bin/python -m pip install -e '.[codex]'
.venv/bin/python -m minimal_rag ask --live --index data/example.sqlite \
  --question "How should I investigate a connection timeout?" --max-turns 4
```

`ask --live` 使用现有 ChatGPT 登录，会发送检索到的原文并消耗用量；`--max-turns` 限制外层调用次数。模型自主决定搜索词、文档内搜索、翻页和继续读取；主实现不加载向量模型、重排或标题预览。5 篇原文的小例子见 [examples/sol-demo](../examples/sol-demo/README.md)。

## 当前 benchmark

```bash
.venv/bin/python -m pip install -e '.[codex,benchmarks]' -e ./experiments
.venv/bin/python -m minimal_rag_experiments.benchmark --dataset enterprise \
  --split test --live --output runs/bm25-enterprise
```

`--dataset` 支持 `enterprise`、`financebench`、`hotpotqa`、`browsecomp`。前三者使用现有 `data/benchmarks/<dataset>/{dev,test}.json`，BrowseComp 使用冻结的 dev100／剩余730；完整830可合并两个不重叠输出目录。Enterprise、FinanceBench、BrowseComp 读取完整 `corpus.sqlite`；HotpotQA distractor 按官方任务提供的每题10篇文档建同一种 BM25 索引。数据准备仍用 `scripts/prepare_rag_benchmarks.py`、`scripts/index_rag_benchmarks.py`；BrowseComp 下载建库入口保留在对应日期实验目录。

生成统一用 sol high 和当前提示。前三者保留4000字符摘录／12轮，BrowseComp 保留2000字符／30轮；搜索排名和工具实现完全相同。每题只生成一次，支持同配置 `--resume`，中断记录不会自动重新抽答案。冒烟用 `--split dev --limit 1`，test 始终运行完整题集。默认生成并发5、裁判并发1，可以分别指定。

Enterprise、FinanceBench 使用原有 sol medium 裁判；HotpotQA 使用官方 EM/F1。BrowseComp 写出 `official/` 下的官方格式答案，`retrieved_docids` 只统计 search 返回；再交给固定的官方脚本＋Qwen3-32B BF16判分，生成完成本身不代表判分完成。所有输出包含逐题轨迹、配置、请求计数和汇总；历史成绩不会自动成为当前实现成绩。

历史混合检索已移入 `minimal_rag_experiments.hybrid_rag`，需要时安装 `pip install -e './experiments[historical]'`。`rag_optimization_eval`、日期目录的冻结 runner 及 `experiments/scripts/` 下的模型准备、dense 建库脚本只用于历史实验，当前 benchmark 不加载它们。历史成绩与新评测结果分栏见[实验索引](../experiments/README.md)。

2026-10-17 独立向量工具实验已按用户要求终止并[归档](../experiments/semantic-tool-20261016/archive.json)。`semantic_search`和`--semantic-url`不再是当前入口；归档代码、向量分片和云端快照仅保留作历史记录，不自动恢复运行。

## 可选文件工具开发对照

`minimal_rag.file_tools.CorpusFiles`现在只提供Grep，使用ripgrep和完整原文文件视图；`files.tools(kb)`得到search/read/Grep。独立Glob和按行Read已删除，原有按字符read保留。Grep只访问导出的`/corpus`视图，沿用成熟接口的参数名；其中`glob`仍是Grep原生的文件过滤参数，不是另一个工具。默认CLI与benchmark工具集仍保持原样。

Enterprise dev100原三工具对照74→78/100只属于历史版本，其runner使用冻结副本，见[旧协议](../experiments/code-tools-20261017/brief.md)。当前仅Grep的跨基准小样本入口见[BrowseComp dev30协议](../experiments/grep-browsecomp-20261017/brief.md)。复跑应使用新的输出目录；已有目录只允许同配置`--resume`，不会重新生成已有答案。

仅Grep的BrowseComp dev30对照已完成：A/B均28/30，输入token下降但没有正确率提升，因此仍为可选候选，详见[报告](../experiments/reports/grep-browsecomp-20261017.md)。

```bash
.venv/bin/python experiments/grep-browsecomp-20261017/run.py --live \
  --files data/benchmarks/browsecomp-plus/grep-files \
  --output runs/grep-browsecomp-new
```

## 测试

```bash
.venv/bin/python -m pip install -e '.[test,codex]' -e './experiments[test]'
.venv/bin/python -m pytest -q tests experiments/tests
```

实验包 `experiments/` 保存评测器、检索变体和各轮实验入口，核心包不导入它。各轮实验的运行命令写在对应日期目录的 brief 或 notes 中；复现历史成绩要从对应提交出发，当前代码不等于当时执行的版本。`data/`、`models/`、`runs/` 被 Git 忽略，保存语料、模型和原始运行材料。
