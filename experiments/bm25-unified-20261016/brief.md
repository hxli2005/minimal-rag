# 单一纯 BM25 主实现与评测

用户决定所有 benchmark 统一使用纯 BM25，保留模型自主搜索和阅读；历史实验与成绩单独保留。当前入口为 `minimal_rag_experiments.benchmark`，检索共用 `minimal_rag.indexed_rag`，不加载历史快照、embedding、RRF、cascade、reranker 或标题预览。核心 loop、模型适配器、BM25 排名与原文读取未修改。

## 配置与验证

- Enterprise：完整 test400，4000字符摘录、12轮；由历史 cascade＋预览改为当前 BM25。
- FinanceBench：完整 test100，4000字符摘录、12轮；由历史 MiniLM 混合检索改为当前 BM25。
- HotpotQA distractor：完整 test100，4000字符摘录、12轮；每题提供的10篇文档使用同一 BM25，替换历史 MiniLM 混合检索。
- BrowseComp：保持2000字符、30轮；对 dev100 中固定随机抽取12题的前两次搜索做离线重放，逐字段对比历史返回，不新增模型调用。
- 生成 sol high，Enterprise／Finance 裁判 sol medium，提示和格式要求沿用原约定；Hotpot 使用官方 EM/F1。统一当前适配器，所有结果归属新实现；不把历史混合检索成绩迁移为新成绩。
- 生成并发 Enterprise 5、Finance 2、Hotpot 2；Enterprise／Finance 各自独立裁判并发1。各题生成调用上限12，裁判最多3次传输／格式重试；保留首次有效判决。600题生成最多7200次，500题裁判最多1500次，外层请求上限8700；另有3道开发冒烟实际12次。SDK内部网络重试不可观测。
- 所有轨迹写入 `runs/bm25-unified-20261016/`，主评测分别使用 `<dataset>-test` 子目录；历史输出与分数不覆盖。

## 命令

```bash
.venv/bin/python -m pytest -q tests experiments/tests
.venv/bin/python -m minimal_rag_experiments.benchmark --dataset enterprise --split test --live --concurrency 5 --output runs/bm25-unified-20261016/enterprise-test
.venv/bin/python -m minimal_rag_experiments.benchmark --dataset financebench --split test --live --concurrency 2 --output runs/bm25-unified-20261016/financebench-test
.venv/bin/python -m minimal_rag_experiments.benchmark --dataset hotpotqa --split test --live --concurrency 2 --output runs/bm25-unified-20261016/hotpotqa-test
```

判分公式、完整配置、应用层调用计数和逐题结果见各目录 `summary.json`。当前代码测试138项通过，三个真实冒烟均完成且正确；三个完整test题集共600题全部生成并判分，无运行／工具错误。Enterprise 298/400（74.50%，比历史v0低6.50pp）；FinanceBench 83/100；HotpotQA EM 77.00、F1 87.25。BrowseComp 24次重放完全一致。完整结果见[报告](../reports/bm25-unified-20261016.md)。
