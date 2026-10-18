准确率 **28/30 = 93.33%**，Wilson 95% 区间 **78.7%–98.2%**，落在 **≥90%，建议全量验证** 档；Qwen3-32B（OpenRouter/DeepInfra fp8）裁判，仅用于方向判断。

| 系统 / 口径 | 准确率 | 证据召回 | 平均 search | 平均 read | 平均生成请求 | 输入 token | sol 请求总数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 本次固定 30 题 | 93.33% | 88.43% | 21.27 | 3.77 | 9.63 | 17,731,031 | 289 |
| GPT-5＋BM25（brief 榜单参照） | 57.59% | — | — | — | — | — | — |
| GPT-5.5＋BM25＋首轮扩展（全量参照） | 90.48% | 96.58% | 33.7 | — | — | — | — |
| GPT-5＋Mixedbread＋get_doc（全量参照） | 90.48% | 83.03% | 11.5 | — | — | — | — |
| AI21，8 agents（全量参照） | 95.18% | 93.17% | 209.3 | — | — | — | — |

- 失败分组：未检到证据0题、部分证据答错2题、全部证据答错0题、预算/超时0题、未完成0题、未判定0题。错题ID：1234、775。检索到文档不代表读到关键正文，证据缺口也不等于唯一错因。
- 运行：完成及判分 30/30 题，固定分母30；生成用量缺失 0 次。请求为应用层计数，SDK内部重试不可见；输入含缓存token 15,440,768。
- 裁判：实际供应商 DeepInfra，调用 42 次（含修复失败），已返回费用 $0.003153；8 次无用量，费用未知。官方空think前缀续写通过双题冒烟，评分正文不变，30/30 个有效判决均为0 reasoning token且内容为空。
- 可比性：SQLite分块BM25、30轮预算、search/read与表中全量系统不同；表格只作参照，不能证明模型或工具的独立贡献。
- 下一步：建议冻结当前配置跑全量830题，检验90%水平能否保持；按本次均值粗估约7,996次sol生成请求。先验证整体水平，再决定是否改检索，公开说明这30题已作为开发样本。

口径：Qwen3-32B（OpenRouter/DeepInfra fp8）裁判、固定 30 题开发样本、非官方成绩。样本与逐题数字见 [summary](../browsecomp-pilot-20261015/summary.json)，[样本 ID](../browsecomp-pilot-20261015/sample.json)；原始轨迹仅存 `data/benchmarks/browsecomp-plus/pilot-20261015/`。外部参照沿用 [基准选型报告](benchmark-selection-20261014.md)。关闭思考依据：[Qwen模板等价校验](../browsecomp-pilot-20261015/judge-template-equivalence.json)、[Qwen官方说明](https://github.com/QwenLM/Qwen3/blob/main/docs/source/inference/transformers.md)、[OpenRouter续写接口](https://openrouter.ai/docs/api-reference/overview#assistant-prefill)。
