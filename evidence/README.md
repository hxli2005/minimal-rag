# 结果证据

可提交的结果摘要。逐题 JSON、查询、评分和答案导出在被 Git 忽略的 `runs/` 与 `data/` 中；每份摘要的 `raw_detail` 给出原始文件位置与 SHA-256。结论和当前口径以[实验索引](../experiments/README.md)为准。

| 文件 | 内容 |
| --- | --- |
| [portfolio-summary.json](portfolio-summary.json) | 历史 v6（gpt-6-astra）600 题回归与 CDL／Dewey 公开答案配对 |
| [transport-summary.json](transport-summary.json) | 会话复用与增量观察的 token 对照 |
| [validation.json](validation.json) | 元验证记录索引 |
| [real-sol-run-20261012.json](real-sol-run-20261012.json) · [showcase-attempt-20261012.json](showcase-attempt-20261012.json) | 单例真实运行；一次成功、一次失败 |
| [failure-analysis-20261012.json](failure-analysis-20261012.json) | 22 道开发错题、48 道 CDL 配对负题的人工错误分类 |
| [exposure-audit-20261012.json](exposure-audit-20261012.json) | 70 道错题的原文暴露区间 |
| [reference-audit-20261012.json](reference-audit-20261012.json) | 参考答案与原文版本争议 |
| [offline-encoder-granularity-20261012.json](offline-encoder-granularity-20261012.json) | 4,646 篇小库的编码器与切分粒度召回 |
| [bm25-title-preview-20261012.json](bm25-title-preview-20261012.json) · [bm25-replay-20261012.json](bm25-replay-20261012.json) | 标题预览与 BM25 参数离线重放 |
| [sentence-capacity-probe-20261012.json](sentence-capacity-probe-20261012.json) | 分句存储容量估算，非检索成绩 |
| [qwen3-chunk-scale-20261012.json](qwen3-chunk-scale-20261012.json) · [qwen-cpu-rerank-20261013.json](qwen-cpu-rerank-20261013.json) · [rrf-grid-20261013.json](rrf-grid-20261013.json) · [full-bm25-calibration-20261013.json](full-bm25-calibration-20261013.json) | Qwen 抽样扩库、CPU 重排、融合参数与全库 BM25 校准 |
| [semantic-replay-20261013.json](semantic-replay-20261013.json) · [loss-query-replay-20261013.json](loss-query-replay-20261013.json) | 语义漏检与负题查询重放 |
| [enterprise-sol-dev24-20261013.json](enterprise-sol-dev24-20261013.json) · [sol-prefix12-20261013.json](sol-prefix12-20261013.json) · [cross-oracle-sol-20261013.json](cross-oracle-sol-20261013.json) · [oracle-effort-20261013.json](oracle-effort-20261013.json) · [current-errors-20261013.json](current-errors-20261013.json) | sol 小样本对照与 gold 干预 |
| [complete-answer-20261013.json](complete-answer-20261013.json) · [tool-capabilities-20261013.json](tool-capabilities-20261013.json) · [fixed-evidence-v3-20261015.json](fixed-evidence-v3-20261015.json) | 未采用的回答提示与工具候选 |
| [failure-layer-audit-20261015.json](failure-layer-audit-20261015.json) | v0 test400 错题分层与公开答案长度统计 |
| [hotpot-output-audit-20261013.json](hotpot-output-audit-20261013.json) | Hotpot EM 失分核对 |
| [official-prompt-compatibility-20261013.json](official-prompt-compatibility-20261013.json) · [official-history-export-20261013.json](official-history-export-20261013.json) | Enterprise 官方提示兼容检查与历史 500 题官方格式导出 |
| [overnight-summary-20261013.json](overnight-summary-20261013.json) · [overnight-final-checks-20261013.json](overnight-final-checks-20261013.json) · [request-reconciliation-20261013.json](request-reconciliation-20261013.json) | 2026-10-12 夜间汇总与请求核账 |
