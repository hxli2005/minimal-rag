# 一次真实 sol 运行：找回事故缓解时限

**这次 agent 找回了一次自然问题检索漏掉的政策，但单例不能证明总体优势。** 2026-10-12 实际调用 `gpt-6.1-sol`，CPU SQLite BM25，模型自己选择查询和阅读；不是脚本化模型决策。

问题：What are the proposed internal time to mitigate targets by severity for incident response, regardless of customer deployment tier?

## 实际过程

1. 查询 `"time to mitigate" "severity"`，找到内部事故政策。
2. 在文档 `dsid_c83811391709468fa9534e6495c205e3` 内查询 `TTM targets proposed`，检查 TTM 目标。
3. 根据实际返回原文给出四档目标、统一适用范围和计时定义。

模型生成调用 3 次，盲裁判 1 次；两种请求都计入整晚 500 次共享预算。输入 token 28,880，其中缓存 0；输出 token 261。累计用量每会话取最后一次，缓存包含在输入里，不重复相加。

## 最终原始回答

The proposed internal time-to-mitigate (TTM) targets are the same across Hosted, Dedicated, and Private deployments:

- Sev0: 30 minutes
- Sev1: 2 hours
- Sev2: 1 business day
- Sev3: 5 business days, or the next planned release train

TTM starts when the incident is declared and severity assigned, and ends when service is restored to its SLO or a documented acceptable degradation mode. These are draft internal targets for breach alerting and leadership escalation.

来源：`dsid_c83811391709468fa9534e6495c205e3`。

## 验证和限制

本地 sol 裁判判为正确，5/5 参考事实支持。相同题、相同 sol、相同正文库，一次自然问题检索的答案判为错误；agent 多用了 2 次生成，不能把这个单例解释为相同成本或总体提升。这里是已知开发题，政策来自公开的模拟企业语料，原文称 **draft internal targets**，不是当前真实公司生产承诺。

完整本地轨迹：`runs/overnight-20261012/results/enterprise/agent/qst_0105.json`；可携带的实际观察与调用记录见 [证据](../../evidence/real-sol-run-20261012.json)。本轮企业24题开发对照已全部完成，但不是完整500题的新模型重跑；历史600题成绩不因本例改变。
