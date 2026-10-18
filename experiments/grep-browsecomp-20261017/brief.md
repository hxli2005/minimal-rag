# Grep-only：BrowseComp-Plus dev30 跨基准小样本

用户要求删除未起作用的新Read与Glob，只保留Grep，在另一个benchmark小范围对照；后续明确实验无需严格限制裁判，重点看对比效果。因此使用BrowseComp-Plus与现有sol代理裁判，不租GPU，不恢复已归档向量实验，也不复用其答案。

## 固定设计

- 样本：既有BrowseComp dev100固定顺序的前30题，见sample.json。不是挑错题或按结果筛选；30题曾用于开发，不作为独立test成绩。
- A：当前BM25 search/read；B：同样工具加Grep与路径/原文ID桥接。无新Read、Glob或其他工具。Grep原参数、扫描与排序、输出预算不调优；只将文件名前缀的URL编码说明写清，保持原文ID引用。
- 完整100,195篇原文，BM25索引、chunk、排名和摘录参数不变。每组新生成30题，模型自行选择工具、参数、次数和结束。
- 相同gpt-6.1-sol high、原系统提示、官方QUERY_TEMPLATE（仅get_document改为read）、search摘录2000字符、最多30次模型调用、单调用180秒、单题1800秒；共用生成并发5，按题交错A/B先后。
- 相同gpt-6.1-sol medium盲裁判，使用冻结官方0469490的evaluate_run.py中的GRADER_TEMPLATE与解析函数；裁判看相同问题、标准答案和各自回答，不看组名、轨迹或工具。每个判决一轮，无工具，裁判并发1。
- 代理裁判只支持本轮配对开发比较，不与历史Qwen3-32B BF16官方判分或公开榜单直接横比。
- 首次有效回答与判决保留；生成失败计错，留在每组30题分母。仅裁判传输/解析失败最多3次尝试；resume不重新生成已经开始的题。不开新付费服务、不充值、不消费重置券。
- 两组轮数/超时相同，实际token与工具时间如实记录，不宣称严格等token。结果目录runs/grep-browsecomp-20261017。

## 核验与指标

原文导出逐篇与SQLite文本对比，不读取题目标签选文档。Grep输入为真实全库文件视图，模型无法访问问题/标准答案文件。路径结果是发现，只有返回正文才计正文曝光。生成后再用qrels计算证据覆盖，标签不传给模型。

主要看正确数与配对胜/负/平，报告配对bootstrap 95%区间（100,000次，seed20261017）。同时看模型调用、输入token（含缓存与不含缓存）、输出token、实际耗时、Grep调用/使用题数、截断/空返回与错误。30题仅作迁移诊断，不以单题胜负宣布效果，也不自动跑完整830题。

## 运行

```bash
.venv/bin/python experiments/code-tools-20261017/export_corpus.py \
  --index data/benchmarks/browsecomp-plus/corpus.sqlite \
  --output data/benchmarks/browsecomp-plus/grep-files
.venv/bin/python experiments/grep-browsecomp-20261017/run.py --live \
  --files data/benchmarks/browsecomp-plus/grep-files \
  --output runs/grep-browsecomp-20261017
```

Enterprise原三工具版本在experiments/code-tools-20261017/frozen_file_tools.py保留，其字节与原结果记录的源码摘要一致；历史runner显式使用该副本。历史结果不改写，主包只提供Grep。

开跑前142项测试通过；100,195篇导出原文逐篇字节核对一致，见export-validation.json。全库Grep预检约5.1秒，工具集合核对为search/read/Grep，官方裁判模板与解析函数可用。正式对照于UTC 2026-10-17 11:27启动。
