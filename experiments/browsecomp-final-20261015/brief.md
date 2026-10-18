# 任务 2026-10-15：BrowseComp-Plus 最终版本的官方复核

> 本任务在[BrowseComp＋Enterprise夜间双主基准优化](../system-v1-20261015/brief.md)完全结束后另行启动，不放进每个候选的循环。提交对象由两项主评分决定的current_best预先固定；Finance辅助结果和Hotpot历史结果不参与选择，没有新候选采用就复核原基线A。

先读AGENTS.md；决定与理由写入本目录notes.md，恢复时先读记录和输出目录。保持模型、工具、提示、语料、样本和评分的冻结口径，不根据官方判分结果换版本或继续本夜调参。

## 目标与选中的系统

用官方 `evaluate_run.py` 和Qwen3-32B原版BF16复核最终系统；OpenRouter fp8只用于监控和交叉核对，不进入官方主表。95.18、90.48、57.59是既有报告采用的历史参照值，发布结论前核对最新榜单及对应系统设置，不把历史值直接称为当前最高。

| 夜间最终版本 | 使用的运行结果 | 是否新生成730题 |
| --- | --- | --- |
| v0，或没有新候选晋级 | 已完成的A830 | 否 |
| v1且BrowseComp配置与A-lean一致 | 复用A-lean dev100，冻结同一配置 | 是 |
| v2及后续候选 | 该最终版本自己的冻结dev100；有效配置一致时可复用原结果 | 缺少完整同配置结果时才跑 |

全文以 `selected` 表示最终选中的系统，不能默认等于A-lean。记录选中的版本、git提交、配置、dev结果目录及夜间决策来源。A830是原基线，B230是官方标准工具配本项目检索的对照；若selected就是A，只复用A的评分，不重复启动同一答案的裁判。

## 开始条件和独立预算

- 上一轮BrowseComp full已经真实收尾：workflow终态为completed或completed_with_budget_or_evaluation_limitations，包含b200/report，最终backfill-state覆盖A/A-lean/B且completed，报告已提交，父进程和阶段子进程已退出。
- 夜间优化已停止，晨报、night状态与current_best已冻结并提交；不能仍在提出新候选时开始官方选型。
- 本任务另行启动并记录明确截止时间。需要新生成时保留**7,000次sol应用层请求**上限，不占用或重置已结束的夜间10,000次总账，也不在夜间额度耗尽后自动启动；本阶段730题不混入夜间主dev200或Enterprise test400。
- 本任务不自行租机、开通账号或恢复用户暂停的巡检。没有GPU连接时保存准备材料和操作说明；这不会阻断此前夜间循环的完成。

## 两块H800的分工

建议同一实例的2×H800 80GB，优先NVLink互联，到机后检查实际显存和拓扑。两卡共同运行一个Qwen3-32B BF16裁判，固定 `CUDA_VISIBLE_DEVICES=0,1`、`TP=2`，每次只开一个双卡判分进程。

| 资源 | 任务 |
| --- | --- |
| Mac | search-only导出、冻结配置下的最终730题生成与检索（需要时并发5）、统计与报告 |
| OpenRouter | 独立并发1监控和交叉核对；无模型响应的失败统一补判 |
| 同机双H800 | 官方BF16判分，TP=2、官方batch_size=64及其他默认参数不变 |

先在Mac准备包再租GPU计算。GPU做两题装配冒烟，通过后按A830、B230、selected已有dev100顺序判分；同一时间Mac可生成selected缺少的730题。冒烟有效评分保留，不重判挑分。GPU若先完成，取回evals后停止空闲计算；缓存保留与否以租赁平台实际规则为准。最后只补判新增答案，汇总、取回结果、清理敏感数据并停止GPU计费。

记录首个64题批次耗时后再估算剩余租期；下载、安装、加载和判分分开计时，不预先承诺完成时长。GPU仍在整个夜间版本筛选之后使用。

## 第1步：修正retrieved_docids（不重跑）

- 官方 `search_agent/utils.py::extract_retrieved_docids_from_result`只收集search/retrieval工具返回的docid，不计get_document单独读过的文档。
- 在本目录实现 `export_v2`，对已有轨迹只汇总search hits的docid，其余字段不变；不修改原轨迹和历史导出。
- 导出到 `data/benchmarks/browsecomp-plus/final-20261015/runs/{A,selected,B}/`。selected=A时直接引用A目录，不重复复制/评分；另保留selected版本映射。
- 记录每组修正前后的召回率，差异超过0.1pp时解释。先前部分样本“召回不变”只是历史检查，不能代替本次全量核对。

## 第2步：只补最终系统缺少的730题

- selected=A时跳过。其他情况逐项对照该最终候选的冻结配置、代码和dev100，不借用其他版本的结果。
- v1有效配置为BM25、search/read、excerpt_chars=800；后续候选沿用它自己的已验证装配，不自动覆盖成800字符。模型gpt-6.1-sol/high，max_calls/max_turns30、request_timeout180、case_timeout1800，样本为 `browsecomp-full-20261015/sample-remaining730.json`。
- 只有配置、提示与工具行为一致才复用dev100；原始文件逐字节核对，已完成答案不重跑。
- 请求上限7,000，生成并发5；前100题完成后记录实际请求、未缓存token和耗时，估算全程。预计超预算或截止则停止并保留部分结果，不改配置、不缩分母冒充830题完成。
- OpenRouter监控裁判沿用上一轮冻结参数与首次有效判决。3次只计模型响应，连接失败/HTTP错误另记并统一补判；429只降低出问题侧并发，usage limit或权限阻塞则停止并保存断点，不绕过权限、不改CODEX_HOME。

## 第3步：GPU判分包与官方命令

本地目录 `data/benchmarks/browsecomp-plus/final-20261015/gpu-bundle/`：

- 将现有official目录固定版本 `046949032b0328319cc9a02663a759ec601d9402` 的内容原样复制到bundle根目录，直接包含pyproject.toml、uv.lock、scripts_evaluation和topics-qrels，不修改官方代码。
- 初始包包含A830、B230及selected已有dev100；最终selected必须有830个终态run JSON才可作为全量提交。selected=A时只需A830和B230。失败题保留原状态/空答案，不补造completed；缺少未运行题时不宣称全量完成。
- 从本地queries.jsonl复制 `data/browsecomp_plus_decrypted.jsonl`，字段保持query_id/query/answer。语料和SQLite索引留在Mac，GPU无需它们。
- bundle根目录执行 `uv sync --frozen --python 3.10`，后续始终用项目环境，不额外安装flash-attn或Java。
- 逐组执行 `uv run --frozen python scripts_evaluation/evaluate_run.py --input_dir runs/<组名> --tensor_parallel_size 2 --model "${MODEL:-Qwen/Qwen3-32B}"`；固定两卡可见，不改变temperature0.7、top_p0.8、top_k20、max_output_tokens4096、batch_size64、官方提示和关闭思考方式。
- MODEL可指向从HF或ModelScope下载的同一官方BF16原版权重本地目录；判分前记录实际权重来源、revision、精度、vLLM/torch/CUDA版本至 `evals/model-source.txt`。镜像只影响下载，不替换模型或依赖版本。
- 正式组间共用冻结环境。增量补入selected730题后再次运行官方脚本，利用已有eval缓存跳过有效判决，不使用 `--force` 挑分。
- 取回 `evals.tar.gz`，保留逐题结果；检查A/selected/B的实际数量与计划一致。官方失败题仍留在准确率分母，校准误差按官方原实现，不自行补置信度或修算法。
- README-gpu.md不超过5行：同机双H800/TP=2、数据盘位置、锁定环境命令、先冒烟实测耗时、取回文件与清理要求。

如果已提供且可连通的GPU_SSH指向用户租用机器，按授权上传私有bundle、运行、取回evals；完成后删除远端bundle、ground truth及含答案的判分副本，记录删除命令和结果。不得上传到公开链接、HF或对象存储。连接未提供时停在已准备好的操作说明，不启动或续办夜间优化去等待它。

## 第4步：汇总、边界与提交文件

- 主结果取官方 `evaluation_summary.json` 的Accuracy (%)、Recall (%)、Calibration Error (%)；平均搜索次数映射 `avg_tool_stats.search`，不重新计算官方主分。
- 报selected全量准确率、Wilson95%区间，以及剩余730题单独结果；selected=A时原有dev100仍明确标为开发样本。
- 对最终730题同时报告相对A的配对差及95%区间；只有点估计和区间下界都≥−2pp，才可连同Enterprise test结论表述两个主基准质量未退步。否则如实披露BrowseComp全量差距，不换提交对象、不回到本夜调参。Finance和Hotpot不能补偿这项差距。
- selected≠A时按830题比较selected对A，按230题比较selected对B；selected=A时只需A对B。逐题bootstrap10,000次、seed20261015，报胜/负/平；成本包含全部尝试并标注缺失用量。
- 官方与OpenRouter判决按共同有有效判决的题比较一致率，列出不一致的query_id；缺判决/生成失败另列，不冒充裁判分歧。
- 本轮选型发生在已有基线结果之后，不能描述成上一轮已预先选定A-lean。后续结果无论好坏都保留，不换提交对象，也不补跑另一个版本挑最高分。

| 结果 | 可用表述 |
| --- | --- |
| 官方脚本点估计高于核实后的公开参照值 | 本地官方脚本点估计超过该公开值，待维护方核实 |
| Wilson下界高于公开参照点估计 | 本系统区间下界高于该公开值；不是两系统配对显著性检验 |
| 未超过参照或结果不齐 | 报实际成绩和不足，不写榜单第一或SOTA |

以选中系统的官方summary生成 `final-20261015/submission.json`：LLM写 `gpt-6.1-sol (effort high)`，Retriever如实写冻结检索实现，Link留空，Evaluation Date为判分日期，per_query_metrics原样保留。英文scaffold不超过5行，写实际工具/摘录长度、单agent/调用上限、dev100及夜间多候选选择经历；不能对所有版本写死800字符。只准备提交文件，不发送邮件、不上传榜单。

报告 `experiments/reports/browsecomp-final-20261015.md` 一屏以内：第一句写selected版本、官方准确率与区间；表格列selected、A、B及已核实参照（selected=A不重复行），展示准确率、召回、搜索次数、未缓存token、校准误差；最多5点说明是否可提交、对照贡献、成本变化、裁判一致率、限制。口径明确官方0469490、Qwen3-32B BF16、vLLM，未经维护方核实。

## 规矩

- 题目、答案、裁判正文只留data或私有GPU机器，报告只写query_id，不提交Git。
- 在codex/standalone-rag分阶段提交口径修正、最终系统全量摘要、官方判分/提交文件和报告；不推送、不碰main。
- 不改src或官方仓库代码，不新建测试套件。发现但不阻塞的问题写docs/issues.md，一行一条。
- 终端收尾总结不超过10行，明确实际完成范围、遗留问题和是否已产生可提交文件。
