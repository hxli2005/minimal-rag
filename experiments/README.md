# 实验索引

2026-10-17 更新。当前统一实现与历史实验分开列出，旧成绩保留原装配归属。未解决问题见[待办](../docs/issues.md)。

**当前阶段：已冻结纯 BM25＋search/read（不启用 Grep），停止探索性调优，转入官方评测与维护方复核。** [冻结清单](../evidence/frozen-bm25-20261017/manifest.json) · [源码包](../evidence/frozen-bm25-20261017/source.tar.gz) · [提交路线与差距](../docs/runbook.md#冻结与官方提交2026-10-17)。历史 BrowseComp 93.01% 单独准备提交，不冒充当前冻结版本成绩；Enterprise 当前77.25%仍是本地结果。尚未提交或获维护方认可。

## 当前主实现：纯 BM25

2026-10-16 起，所有 benchmark 通过 `python -m minimal_rag_experiments.benchmark` 使用同一套当前核心＋纯 BM25＋模型自主 search/read。CLI `ask` 共用这套工具。向量、cascade、重排和标题预览保留在实验包，不进入主路径。命令与预算见[运行指南](../docs/runbook.md)。

纯BM25统一后的三个完整test题集已全部完成；随后只修改search工具说明，再完成Enterprise test400同口径对比。生成模型sol high，Enterprise／Finance使用同一本地sol medium裁判，Hotpot使用官方EM/F1。下表明确区分说明版本。

| 基准 | 纯 BM25结果（说明版本） | 历史实现 | 差值 |
| --- | ---: | ---: | ---: |
| Enterprise test400 | **309/400＝77.25%（新搜索说明）** | 324/400＝81.00% | −3.75pp |
| FinanceBench test100 | **83/100（原说明）** | 82/100 | +1.00pp |
| HotpotQA test100 | **EM 77.00，F1 87.25（原说明）** | EM 75.00，F1 86.73 | +2.00／+0.52pp |

**新搜索说明保留在主实现中。** 与纯BM25原说明相比：298→309题正确（+2.75pp，33胜22负），search 3342→2469（−26.12%），生成调用2285→1977（−13.48%），输入token（含缓存）85.08M→66.00M。正确率差95%区间[−0.75,+6.50]pp，提分稳定性仍未确认，省搜索的证据更明确。中断开销单列，见[新说明报告](reports/search-description-20261016.md)与[摘要](search-description-20261016/summary.json)。原说明的298/400及各项历史记录保留在[统一BM25报告](reports/bm25-unified-20261016.md)。

BrowseComp 的 BM25检索实现与参数未改，12题24次历史搜索重放逐字段一致。新搜索说明的BrowseComp A已完成830题生成，但未完成官方判分；FinanceBench和HotpotQA尚未重跑。其他成绩保留原说明归属，BrowseComp原有成绩保留在下方历史表。新入口与验证命令见[本轮记录](bm25-unified-20261016/brief.md)。

## 当前开发候选：仅Grep（2026-10-17）

按用户要求删除新Read与独立Glob，当前可选工具集为search/read/Grep。Grep参数、匹配和返回策略不调优。BrowseComp-Plus既有dev100前30题A/B对照已完成：**28/30对28/30，1胜1负28平，差值95%区间[−10,+10]pp**。两组新生成，使用同一sol medium代理裁判与官方评分提示；不租GPU，结果不与历史官方判分直接横比。

B在12题调用Grep共44次；BM25 search由563降至429，生成请求270对271，输入token含缓存−11.85%、不含缓存−15.92%，平均回答耗时108.1→111.3秒（+2.93%）。60题全部正常回答并首次有效判分，0运行/协议/工具错误，用量记录完整。保留为可选工具，暂不默认启用；30题不足以证明总体等效或稳定效率收益。142项测试通过。见[报告](reports/grep-browsecomp-20261017.md)、[协议](grep-browsecomp-20261017/brief.md)、[摘要](grep-browsecomp-20261017/summary.json)。

[一胜一负逐题审计](reports/grep-browsecomp-paired-audit-20261017.md)：418的Grep发生在BM25已取得最终证据之后，无新增参考来源；666的首个Grep实际命中了关键参考文档，但排在前面的一个130万字符长行触发输出break，挡住了后续命中，随后17次Grep积累约42.7万字符结果并未找回参考文档。只做离线核对，未改分数或工具；长行问题已记录待修。

## 历史开发对照：三种文件工具（2026-10-17）

保留框架、核心循环和BM25，按Claude Agent SDK 0.3.301公开接口实现可选Grep/Glob/Read。Enterprise dev100两组重新生成：A为当前search/read，B加文件工具及路径/来源ID映射；同sol high、12轮、超时、原sol medium裁判，按题交错执行。

**A 74/100 → B 78/100；7胜3负90平，配对95%区间[−2,+10]pp。** 改善集中在basic（28→33），semantic为16→15；单次开发集尚不能确认稳定提分。生成输入token含缓存+1.43%、不含缓存−3.99%，平均单题回答耗时28.9→36.4秒。B在28题使用Grep共107次、2题使用Glob共3次，新的Read未被调用。两组200题全部回答并首次有效判分；0生成/协议错误，1次可恢复Grep正则错误。

后续已裁剪为仅Grep，原三工具实现保存在本实验的frozen_file_tools.py供旧runner使用；默认工具不替换，未启动test。原验证143项测试通过，全部511,962篇文件逐篇核对原文相同。见[完整报告](reports/code-tools-20261017.md)、[协议与命令](code-tools-20261017/brief.md)、[结果摘要](code-tools-20261017/summary.json)。该开发集结果不能替代上方当前主实现test400成绩，也不作为当前Grep-only版本成绩。

## 已终止并归档：独立向量工具（2026-10-17）

用户已停止本轮并放弃新增向量索引。Enterprise test400：纯BM25 **309/400**，加入独立`semantic_search`后 **305/400**（18胜22负，差−1pp，配对95%区间[−4,+2]pp）；semantic类 **68/100→64/100**。93道技术错误均按授权规则恢复，最终400题有完整有效判分。BrowseComp B在234份正常答案时停止，本轮A/B均未完成官方判分，不能宣称完成双基准门槛。

主包的向量工具与统一benchmark的`--semantic-url`入口已删除，主路径继续纯BM25。完整结果和撤回前代码保留在[归档清单](semantic-tool-20261016/archive.json)，过程见[实验记录](semantic-tool-20261016/brief.md)。云实例和自动续办已停止；历史向量分片仅作归档保留，不自动恢复服务。

## 历史基准结果

生成模型均为 `gpt-6.1-sol` high。“本地裁判”指 sol medium 加历史裁判提示，比 Enterprise 官方裁判严，分数不能和公开榜单直接比较。

| 基准 | 装配 | 题集与裁判 | 结果 | 对照 | 来源 |
| --- | --- | --- | --- | --- | --- |
| BrowseComp-Plus | A：当前核心＋BM25，search 摘录 2000 字符，最多 30 轮 | 全部 830 题；官方脚本 0469490＋Qwen3-32B BF16，本机运行，未经维护方核实 | **772/830＝93.01%**，Wilson95% [91.07, 94.56] | 同题 230 题，B（标准 search 工具）194/230；A−B＝+10.00pp [5.22, 14.78]，28 胜 5 负 | [报告](reports/browsecomp-final-20261015.md) · [摘要](browsecomp-final-20261015/summary.json) |
| EnterpriseRAG-Bench（v0） | 历史 cascade 检索＋15 条标题预览，4000 字符，最多 12 轮 | 本地 test400；本地裁判 | 正确 **324/400＝81.00%**，overall 79.16%；2 题超时计错 | CDL 公开答案同裁判 86.50%，差 −5.50pp [−8.75, −2.25]；历史 v6（gpt-6-astra）77.75% | [报告](reports/sota-20261014.md) · [摘要](sota-20261014/enterprise-test-summary.json) |
| EnterpriseRAG-Bench（历史 v3） | v0 检索＋当前模型适配器 | 本地 dev100；本地裁判 | 82/100；**test 未运行** | v0 dev 83/100 | [门槛](system-v1-20261015/v3-dev-gate.json) · [采用记录](system-v3-20261015/adoption.json) |
| FinanceBench | 历史 MiniLM 检索配置 | 本地 test100；本地裁判 | 82/100 | Dewey 公开答案同裁判 74/100，差 +8.00pp [0.00, 16.00] | [摘要](sota-20261014/financebench-test-summary.json) |
| HotpotQA distractor | 每题给定 10 段文档 | 本地 test100（公开 dev 划分）；官方 EM/F1 公式 | EM 75.00，F1 86.73 | 历史 v6 EM 78.00，F1 87.10 | [摘要](sota-20261014/hotpotqa-test-summary.json) |

区间均为按题配对 bootstrap 10,000 次的 95% 区间，只反映题目差异，不含重复生成或裁判波动。BrowseComp 早先用 OpenRouter fp8 代理裁判得到 769/830，已由同一批答案的 BF16 官方脚本复核结果取代（[原记录](reports/browsecomp-full-20261015.md)）。

## 2026-10-15 夜间候选

主评分只用 BrowseComp＋Enterprise；三个候选都没有通过当时的自动门槛，之后用户决定采用 v3。协议见 [protocol](system-v1-20261015/protocol.md)，逐轮记录见 [versions](system-v1-20261015/versions.md) 和[晨报](reports/system-nightly-20261015T102540.md)。

| 候选 | 改动 | Enterprise | BrowseComp | 结论 |
| --- | --- | --- | --- | --- |
| v1 | 两个基准统一为当前核心＋BM25＋800 字符 | dev 79 对 83（2 胜 6 负） | dev 96 对 93（4 胜 1 负） | 合并 −0.5pp [−4, 3]；拒绝 |
| v2 | 保留 v0 检索，摘录统一为 2000 字符 | panel 23 对 26 | panel 28 对 28 | 拒绝 |
| v3 | 只换当前模型适配器 | dev 82 对 83（3 胜 4 负） | dev 93 对 93 | 自动门槛未过，用户采用 |

## 诊断实验

| 实验 | 结果 | 报告 |
| --- | --- | --- |
| 仅修改search说明（Enterprise test400） | 74.50%→77.25%，search减少26.12%，模型调用减少13.48%；纯BM25和核心循环不变 | [完整对比](reports/search-description-20261016.md) |
| Enterprise纯BM25回退诊断（离线） | 43退化题：30检索、2检索＋回答、1阅读＋回答、7已见证据回答缺口、3判分敏感；32检索相关回退题原问题BM25 Top5命中8，实际首query为0 | [逐题诊断](reports/enterprise-bm25-regression-audit-20261016.md) |
| Qwen dev24瓶颈审计（离线） | 24题内参考ID Recall@5仅67.33%→77.33%；10道Qwen原判错中4道候选消费/读取、1道检索、2道混合、1道覆盖待定位、2道裁判争议；原分数未改 | [24题审计](reports/qwen8b4b-dev24-audit-20261016.md) |
| Qwen 8B＋4B 对纯 BM25 | 含参考文档的小库：整体 Recall@5 61.05%→86.31%，semantic 32%→92%；固定 dev24 回答 12/24→14/24，4 胜 2 负；项目覆盖有回退，未晋级主实现 | [报告](reports/qwen8b4b-bm25-20261016.md) |
| Enterprise dev24：闭卷／一次 BM25／agent | 2/24 ／ 12/24 ／ 20/24；agent 生成调用 98 次，一次检索 24 次 | [报告](reports/enterprise-sol-dev24-20261013.md) |
| 完整回答提示（dev24） | 19/24 对 19/24，事实完整度 87.10%→85.87%；未采用 | [报告](reports/complete-answer-20261013.md) |
| 文档与章节工具（dev24） | 21/24 对 21/24，章节调用 0；未采用 | [报告](reports/tool-capabilities-20261013.md) |
| 固定证据回答要求（dev24） | 21/24 对 21/24，事实完整度 91.90%→94.51%；未采用 | [报告](reports/fixed-evidence-v3-20261015.md) |
| v0 test400 错题分层 | 76 错，56 部分答对；30 题全部参考 ID 曾返回，其中 22 题参考正文全部暴露 | [报告](reports/failure-layer-audit-20261015.md) |
| Qwen 10% 抽样扩库 | semantic recall@20：BM25 80% → 混合 84%，未过 +10pp 门槛，未建全量索引 | [报告](reports/qwen-scale-gate-20261013.md) |
| 会话复用与增量观察 | 完整 RAG 输入 token（含缓存）：Enterprise 20 题 −49.9%，Finance 15 题 −67.3% | [摘要](../evidence/transport-summary.json) |
| BrowseComp pilot | 28/30，开发样本 | [报告](reports/browsecomp-pilot-20261015.md) |

更早的检索与错题诊断：[失败分析](reports/failure-analysis-20261012.md) · [暴露审计](reports/exposure-audit-20261012.md) · [标题预览](reports/bm25-title-preview-20261012.md) · [BM25 重放](reports/bm25-replay-20261012.md) · [编码器比较](reports/offline-encoder-granularity-20261012.md) · [语义回放](reports/semantic-replay-20261013.md) · [窗口与一次检索](reports/sol-prefix12-20261013.md) · [跨基准与 gold 干预](reports/cross-oracle-sol-20261013.md) · [当前错题](reports/current-errors-20261013.md) · [官方协议核对](reports/official-protocol-20261013.md) · [基准选择](reports/benchmark-selection-20261014.md) · [单例真实运行](reports/real-sol-run-20261012.md) · [2026-10-13 晨报](reports/morning-20261013.md)。日期报告是当时的记录，后来的结论以本页和[待办](../docs/issues.md)为准。

## 目录

- `src/minimal_rag_experiments/`：评测器、检索变体和领域应用；核心包不依赖这里。
- 日期目录：各轮的 brief、冻结配置、脚本、摘要和 notes。
- `reports/`：各轮人读报告。
- 原始轨迹和逐题数据在被 Git 忽略的 `data/`、`runs/` 中；[evidence/](../evidence/README.md) 保存可提交的摘要。
