# 跨基准与证据干预：提升召回仍不够

**结论：完整参考文档不是当前 sol 的可靠答对保证，后续必须同时验证取证和证据利用。** 新增 FinanceBench、HotpotQA 的各 12 道分层开发补测，以及 12 道已知企业错题的证据干预。模型统一 sol，检索只用 CPU BM25，无 GPU。[逐题、用量与限制](../../evidence/cross-oracle-sol-20261013.json)

| 实验 | 结果 | 条件 |
| --- | --- | --- |
| FinanceBench | 9/12，75.0% | 同本地 PDF 库，agent 最多 5 次生成，37 次生成＋12 次评分 |
| HotpotQA | EM 10/12，83.33%；F1 90.74% | 题内给定 distractor 文档，27 次生成，官方答案公式，无模型评分 |
| 企业旧错题：历史工具观察 | 0/12 | 同 sol high 一次回答，不继续取证 |
| 同题：完整 gold 文档隔离 | 5/12 | 同 sol high 一次回答，仅用参考文档标签选资料，答案／事实不传生成 |

跨基准是小样本开发诊断，不能替代历史 100 题封存分数，也不能宣称金融成绩下跌或 Hotpot 整体提升。没有按成功状态移除题目，没有多次生成挑最佳。

Hotpot的12题都实际暴露了全部标注支持句。两个EM失败分别是输出`National Hockey League`而gold为`NHL`，以及在`Arthur Ignatius Conan Doyle`前增加`Sir`；官方标准归一化不扩展缩写、不删除称号，因此前者F1=0，后者F1=0.8889。人工检查认为实体含义相符，但**官方EM/F1原样保留**，不另报“实际100%”或加题目专用归一化。[输出约定审计](../../evidence/hotpot-output-audit-20261013.json)

gold 隔离组救回 qst_0118、qst_0143、qst_0250、qst_0220、qst_0222。干预同时补全文档、删除干扰文档、改变观察的包装与顺序；救回不能全归因于 recall。qst_0118 历史上已经完整读到全部参考原文，说明噪声与回答利用也可能重要。

七道仍被固定参考判错的 gold-only 题已补齐针对性原文核对，真实遗漏与参考／裁判问题可以在同题共存。例如 qst_0413 的同 ID 新旧版本不同，新的 gold 输入确实含 Nina 的五工作日审批，裁判却按旧版本称它是编造。[审计原文](../../evidence/reference-audit-20261012.json)。不重写任何分数，也不把所有剩余错误都推给裁判。

另一个待审点是qst_0003：原文列“Implementation-ready／repo receives PR”和“Rollout plan”，参考要求land PR及execute rollout，规范强度存在差异。相反，qst_0351的kill-switch限制、watchdog/resync与稳定cohort key建议确实写在原文，候选仍漏掉；这是真正需要改善的证据利用，不能用参考问题掩盖它。

| 后补原文核对 | 结论 |
| --- | --- |
| qst_0240 | 候选确实漏掉host-level controllers；但原文明确写double-counting、premature eviction及evicts inflight capacity，保存裁判称这些描述错误，与原文不符。不能因为这条判因有误就把整题自动改正确。 |
| qst_0341 | 候选给出详细SLO阈值，但未显式写核查5xx和持续接近零的企业shed；原文有5xx核查、持续非零企业shed的stop-the-line。需要区分遗漏与可接受转述。 |
| qst_0369 | ADR明确retryable503 MUST、指定429 SHOULD，候选推荐语气有原文依据；hotfix又要求overload-classified归一化。gold将范围扩大到所有retryable overload/rate-limit，需要审定范围与规范强度，不能直接翻分。 |
| qst_0376 | 原文明确90天配置历史和审计导出包，候选漏掉90天且将审计导出改为可选；还有框架幂等性等程序事实未完整写出，真实遗漏存在。 |

**下一步同时冻结来源版本和评分口径，不能把修正后的判因偷偷当提分。** 工具能力的对照仍保留当前固定参考，任何参考审定或重评另列；仅凭gold-only 5/12不能推断7题都由模型阅读能力不足造成，也不能把这次人工核对换成新的正确率。

同一批 gold-only 输入的 high／max 请求已完成：两组均5/12，逐题正确性完全相同；模型请求字符数相同，提示词、一次生成和medium裁判保持原条件，只改变请求强度与超时。包含生成、裁判和提供方时延的累计用时约445秒→756秒，不能把差异全归因于模型计算，但没有观察到提分收益。[强度记录](../../evidence/oracle-effort-20261013.json)。保留high，不从两组逐题取最好答案；这不是检索系统的基准成绩。
