# OpenRouter API 固定小样本验证（v2）

原Codex全量成绩保持原配置归属。本报告仅覆盖已固定的开发样本；本地裁判不是官方榜单判分。

已发出 192 次API生成请求，已核对费用 $7.510916；未结算请求 0。
专项Key为$20 TOTAL，执行器$14停止线、$6在途预留，生成并发1。

| 阶段 | 组 | 已生成/计划 | 已判分 | 正确 | 费用USD |
| --- | --- | --- | --- | --- | --- |
| 0 | low | 2/2 | 2 | 1 | 0.106968 |
| 1 | low | 30/30 | 30 | 24 | 3.660014 |
| 2 | low | 30/30 | 30 | 23 | 1.847662 |
| 2 | high | 30/30 | 30 | 22 | 1.896273 |

## 结果与判断

**API迁移筛查通过：24/30，对照Codex 22/30。** 配对3胜1负26平；正确率差+6.67pp，配对bootstrap95%区间[-6.67,20.00]pp。这是已知开发样本，不证明两后端等价，也不是API版全量成绩。
宏平均事实完整度：Codex 84.73% → API 92.96%。迁移同时改变运行时、消息包装、原生工具调用和停止策略，不能将差异单独归因于某个改动。

**high没有通过预先设定的收益门槛。** 固定证据下low 23/30、high 22/30；high相对low为1胜2负27平。正确率差95%区间[-13.33,6.67]pp，不能认定high稳定变差。
宏平均事实完整度 86.88% → 88.57%；答案字符数中位数 371.5 → 701.0。篇幅增长没有转化为本轮明确的正确率收益。两组没有输出上限截断，实际请求逐题比较仅text.verbosity不同。

**已核对总费用$7.5109163**，与专项Key累计用量一致；按预算口径8元/美元并计5.5%充值费约63.39元，不是实际银行换汇账单。总额低于200元，未提高任何额度或追加充值。
保留API后端作为可迁移实现；本轮不追加high全量实验，不改变原Codex全量成绩的版本归属，也不自动发送官方提交。

未完成的组不能作为完整准确率发布。阶段1固定原基线22/30、筛查门槛19/30；这不是统计等价证明。
阶段2是同一证据下的low/high回答层对照，两组输出上限均32768，不能解释为完整检索agent成绩。
模型没有返回缓存写入用量时报告缺失，不当作0；实际费用以OpenRouter回执为准。

冒烟门槛：{"passed": true, "cost_usd": 0.1069676, "reasoning_roundtrip_verified": true, "cache_is_not_a_functional_gate": true}
迁移门槛：{"passed": true, "correct": 24, "finished": 30, "n": 30, "baseline_correct": 22, "screen_not_equivalence": true, "amendment": "User-authorized replacement of key-query failures with first actual model runs; old records retained."}

技术恢复：请求80在连接中断前未交付完整响应，供应商标记取消；原收费$0.079958计入总额。原样重发该轮，不重做前面的检索、不替换有效答案。状态：recovered_original_question_without_restart。

执行中用户明确追加规则：读取额度信息失败且未发出任何模型请求的题，不按答错统计，原题补跑。原记录保留；补跑前标pending_retry，不替换任何已有有效答案。补跑可能改变实际low/high先后顺序，必须与原预定顺序区分。

[固定协议](../api-validation-20261017/protocol.json) · [结果JSON](../api-validation-20261017/result.json) · [预算核验](../api-validation-20261017/budget-preflight.json)

原始请求、加密推理与逐题轨迹仅保存在git忽略的 runs/api-validation-20261017/，不随报告公开。
