# 仅优化搜索工具说明：Enterprise 同口径对比

本轮只修改 `indexed_tools` 中 `search` 的说明文本。完整前后文案见 `change.json`。说明允许完整问题或保留关键限定条件的查询，解释宽泛词和过严匹配的风险，并提示由模型自行判断改写或探索已有分页。没有规定固定搜索顺序、次数、必跑查询，也没有新增执行分支。

AST核对确认移除这一段说明字符串后源码完全相同；BM25排名／分词／分块、工具schema与默认值、返回结果、read、核心循环、系统提示、模型适配器、评测器和裁判都未变。

## 评测

- 对照：`runs/bm25-unified-20261016/enterprise-test/`，298/400（74.50%），search 3342次，read 452次，模型调用2285次。
- 候选：`runs/search-description-20261016/enterprise-test/`。
- 完整同一test400、同一全文索引，生成 `gpt-6.1-sol` high，4000字符摘录、最多12轮，单调用180秒／每题900秒，生成并发5。
- 原本地语义裁判 `gpt-6.1-sol` medium，独立并发1；保留首次有效判决，最多3次传输／格式重试。
- 每题只生成一次，400题生成最多4800次、裁判最多1200次，外层调用总上限6000；SDK内部网络重试不可观测。
- 主要指标：正确数／准确率、search总次数／每题均值。辅助记录read、模型调用、输入token（含缓存）、工具错误与分类结果。对准确率做逐题配对比较，计算胜负和bootstrap区间。
- 只运行这一版说明；评测中途不修改文本，不挑题重跑。历史结果和判决保持原样。

```bash
.venv/bin/python -m pytest -q tests experiments/tests
.venv/bin/python -m minimal_rag_experiments.benchmark --dataset enterprise --split test --live --concurrency 5 --judge-concurrency 1 --output runs/search-description-20261016/enterprise-test
```

## 手动停止后的恢复

用户停止后，本轮保留64份完整答案、46份有效判决；18份完整答案尚待判分，5题在关闭SDK时中断、没有完整答案。用户随后要求继续跑完：只将这5份中断轨迹移入 `interrupted-20261016/` 保存，再按原配置重新完成；64份答案不重新生成，46份有效判决不重判。qst_0308保留原失败裁判尝试并继续补判。中断的17次生成调用及工具开销单列计入实验总开销，见 `recovery.json`。

```bash
.venv/bin/python -m minimal_rag_experiments.benchmark --dataset enterprise --split test --live --concurrency 5 --judge-concurrency 1 --output runs/search-description-20261016/enterprise-test --resume
```

## 完成结果

400/400全部完成并判分：309题正确（77.25%），比原说明298题多11题；33胜22负，正确率差95%配对区间[−0.75,+6.50]pp。search 3342→2469（−26.12%），生成调用2285→1977（−13.48%）。保留新说明，提分稳定性不作过度推断。中断额外17次生成／20次搜索／1次阅读；全部实际生成调用1994、裁判402。结果与逐题胜负见[报告](../reports/search-description-20261016.md)。
