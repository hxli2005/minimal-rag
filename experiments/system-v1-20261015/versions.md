# 版本账本

**当前采用v3，后续优化从v3出发。** 2026-10-15用户明确接受v3、放宽本次采用标准；原自动门槛、参考答案和评分保持不变。v3尚无Enterprise test400成绩，历史324/400仍属于v0；本次不启动新实验。[采用决定](../system-v3-20261015/adoption.json)

每个系统版本一行，按[双主基准夜间循环](protocol.md)走panel60 → 主dev200 → Enterprise test400。每道主门槛同时比较当前基线和固定夜间anchor；失败、未完成和未采用版本都保留。整夜结束才冻结最终版本并进入GPU复核。

## 主评分与晋级

- BrowseComp-Plus：Qwen3-32B代理裁判准确率，panel30/dev100；剩余730题只给夜间最终版本另行验证。
- EnterpriseRAG-Bench：本地sol裁判正确率，panel30/dev100/test400。
- 主dev200由两项各100题组成，各50%权重；夜间test只指Enterprise400，不夹入任何dev题。每道门槛分别保存对current_best、night_anchor的配对差/区间和主成本比。
- Finance与Hotpot的分数、成本和缺判不进入上述统计或采用/淘汰条件。辅助实际调用仍计入整夜10,000请求总账。
- 所有题集都是已知开发/回归。“—”表示尚未运行；记录需关联night_id、代码版本、总请求、截止时间和晨报。

| 版本 | 当前基线／夜间anchor | 改动 | 主Enterprise dev100 / test400 | 主BrowseComp dev100 | 主门槛对两个基线的差值与区间 | 主成本比（两套） | 采用状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v0 | — | Enterprise历史cascade＋15条预览；BrowseComp BM25、2000字符（A） | 83.0 / 81.0 | 93.0 | — | — | 基线 |
| v1 | v0 / v0 | Simpler shared current core and BM25 assembly | 79.0 / — | 96.0 | dev: current_best: undetermined Δ-0.5000000000000001 CI[-4.0, 3.0]；night_anchor: undetermined Δ-0.5000000000000001 CI[-4.0, 3.0] | None / None | rejected |

| v2 | v0 / v0 | 保留v0检索/适配器，仅统一search摘录2000字符 | — / — | 93.0 | panel: current_best: fail Δ分项 CI—；night_anchor: fail Δ分项 CI— | — / — | rejected |

| v3 | v0 / v0 | 仅统一当前适配器；保留v0检索和4000字符 | 82.0 / — | 93.0 | dev: current_best: undetermined Δ-0.5 CI[-3.0, 2.0]；night_anchor: undetermined Δ-0.5 CI[-3.0, 2.0] | None / None（原门槛） | adopted：用户复核通过 |

v0主评分来源：[sota-20261014](../reports/sota-20261014.md)与[BrowseComp full](../reports/browsecomp-full-20261015.md)。A全量830题本地准确率92.65%；它是后置全量证据，不与夜间主dev200/test400混算。

## Finance单列辅助

夜间最终版本冻结后，有资源才做一次dev50辅助检查；必要基线只补一次，最多500次sol请求且占整夜总账。不跑每个候选、不额外跑test100，不凭辅助结果替换已冻结的最终版本。

| 版本 | Finance dev50 | 历史Finance test100 | 辅助差值、成本与缺失 | 对主晋级的影响 |
| --- | --- | --- | --- | --- |
| v0 | — | 82.0 | 历史sol裁判；dev基线仅在需要辅助配对时补齐 | 无 |
| 夜间最终版本 | — | 不默认新跑 | 未执行/不完整也需注明 | 无 |

## Hotpot历史资料

Hotpot本夜预算0，不进入panel、主dev、test门槛、主成本比或最终版本选择。保留[sota历史成绩](../reports/sota-20261014.md)：test100 EM75.0、F1 86.73；保留源码和原始轨迹，另行排查时再明确安排。

- 2026-10-15T10:45:53.026482-07:00 夜间20261015T102540，v1 panel：current_best=pass, Δ=见分项pp, CI=None, cost=None (panel不判成本)；night_anchor=pass, Δ=见分项pp, CI=None, cost=None (panel不判成本)；完整结果见 experiments/system-v1-20261015/v1-panel-gate.json。

- 2026-10-15T10:56:30.251951-07:00 夜间20261015T102540，v1 dev：current_best=undetermined, Δ=-0.5000000000000001pp, CI=[-4.0, 3.0], cost=0.9087843985663191 (upper_bound)；night_anchor=undetermined, Δ=-0.5000000000000001pp, CI=[-4.0, 3.0], cost=0.9087843985663191 (upper_bound)；完整结果见 experiments/system-v1-20261015/v1-dev-gate.json。

- 2026-10-15T10:58:12.489628-07:00 夜间20261015T102540，v1 dev：current_best=undetermined, Δ=-0.5000000000000001pp, CI=[-4.0, 3.0], cost=None (undetermined)；night_anchor=undetermined, Δ=-0.5000000000000001pp, CI=[-4.0, 3.0], cost=None (undetermined)；完整结果见 experiments/system-v1-20261015/v1-dev-gate.json。

- 2026-10-15T11:06:44.379158-07:00 夜间20261015T102540，v2 panel：current_best=fail, Δ=见分项pp, CI=None, cost=None (panel不判成本)；night_anchor=fail, Δ=见分项pp, CI=None, cost=None (panel不判成本)；完整结果见 experiments/system-v1-20261015/v2-panel-gate.json。

- 2026-10-15T11:10:53.092132-07:00 夜间20261015T102540，v3 panel：current_best=undetermined, Δ=见分项pp, CI=None, cost=None (panel不判成本)；night_anchor=undetermined, Δ=见分项pp, CI=None, cost=None (panel不判成本)；完整结果见 experiments/system-v1-20261015/v3-panel-gate.json。

- 2026-10-15T11:20:00.982221-07:00 夜间20261015T102540，v3 panel：current_best=pass, Δ=见分项pp, CI=None, cost=None (panel不判成本)；night_anchor=pass, Δ=见分项pp, CI=None, cost=None (panel不判成本)；完整结果见 experiments/system-v1-20261015/v3-panel-gate.json。

- 2026-10-15T11:41:02.131051-07:00 夜间20261015T102540，v3 dev：current_best=undetermined, Δ=-0.5pp, CI=[-3.0, 2.0], cost=None (undetermined)；night_anchor=undetermined, Δ=-0.5pp, CI=[-3.0, 2.0], cost=None (undetermined)；完整结果见 experiments/system-v1-20261015/v3-dev-gate.json。

## 2026-10-15夜间终态与后置官方复核

原自动流程中3候选均未通过，先冻结v0，sol新请求1340；后续用户明确接受v3，当前采用版本已更新为v3，未新增请求。后置按官方0469490/Qwen3-32B BF16复核同一批A答案：全量772/830（93.01%，Wilson95%[91.07,94.56]），dev95/100、其余730题677/730；它不替换上表用于选型的代理93/100，不回流改选。B230为194/230；同题A217/230，差+10.00pp，配对区间[5.22,14.78]。当前v3的Enterprise dev82/100，test400未运行；v0历史本地dev83/100、test324/400保持原归属。完整范围、费用估算与清理见[晨报](../reports/system-nightly-20261015T102540.md)和[官方复核](../reports/browsecomp-final-20261015.md)。

- 2026-10-15T19:12:03.967358-07:00 用户复核决定：v3通过并采用，current_best=v3，night_anchor=v0；接受当前开发集波动以统一适配器。旧dev门槛仍为未通过，不声称test400通过；新实验0，续办保持暂停。
