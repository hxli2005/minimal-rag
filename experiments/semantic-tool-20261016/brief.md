# 已归档：独立 semantic_search 对照实验

**2026-10-17 用户决定停止实验并放弃新增向量索引。** 主实现保持纯 BM25；`semantic_search` 模块和统一 benchmark 的 `--semantic-url` 入口已撤回。所有云实例、后台进程和自动续办已停止，禁止按下方历史运行指令恢复任务。

Enterprise 已完整评测：A 309/400，B 305/400；semantic 类 68/100→64/100；总体配对95%区间为[−4,+2]个百分点。BrowseComp A已生成830题、B有234份正常答案，均未完成本轮官方判分，不提供完整双基准结论。停止和撤回依据用户明确决定，不冒充完整1230题采用门槛已执行。

[归档清单](archive.json)列出代码快照、完整评测记录包及保留在原目录的向量分片。归档包不含SSH凭据。原始结果、重试轨迹、索引清单及关机证据保留；大索引仅作归档材料，不再接入产品。下面保留历史协议和过程记录，撤回前原文另见[协议快照](archive/brief-before-closure.md)。

## 原运行前冻结说明

只增加独立工具，由模型决定调用；不融合、不重排、不换模型。不改 BM25、原索引、切块、read、核心循环、系统提示、裁判或结果长度。

## 样本、配置和判分

- Enterprise A：复用 `runs/search-description-20261016/enterprise-test/`，309/400；B 使用完全相同 test400、sol high、4000字符、12轮、并发5，sol medium 原裁判并发1。
- BrowseComp A/B：均重新生成完整830题，按既有 dev100 后 remaining730 的固定顺序；sol high、2000字符、30轮、并发5、单调用180秒、单题1800秒；相同原始语料及官方回答格式。A 不调用GPU。
- BrowseComp 评分沿用官方固定版本046949032b0328319cc9a02663a759ec601d9402、Qwen3-32B BF16及原参数，A/B同口径；先生成，再集中判分，不替换裁判。判分租赁也在120元内。
- 每题只生成一次，首次有效判决永久保留。失败与中断保留在完整分母，不挑题重跑；仅缺少有效裁判输出可按原规则恢复。工具说明从开跑起固定。
- 完整 traces 保存每次工具名字、参数、返回内容/文档ID、最终答案、调用用量、裁判输入和全部尝试。BrowseComp判决另存judgements；不发布题目或标准答案。

## 向量工具

模型Qwen3-Embedding-8B，BF16模型推理、不量化，FP32归一化和保存；每篇文档一个4096维向量，文档最长4096 token、eos/last-token pooling、L2归一化、无文档指令前缀。查询前缀严格为 `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:`（冒号后不额外插入空格）。查询长度沿用官方searcher默认8192 token。

BrowseComp直接下载Tevatron/browsecomp-plus-indexes的qwen3-embedding-8b预建索引。Enterprise全库511,962篇在云端编码，不用参考题/答案选择文档。固定随机种子20261016做4096篇均匀样本吞吐测试；先验证模型和设置，再据实测含余量估计全量费用。精确检索遍历全部向量，无ANN、融合或重排。GPU同时装载模型与两个索引，服务仅监听127.0.0.1，经SSH隧道使用。

semantic_search参数query/limit/offset，与search共享参数默认值和范围；返回相同字段、文档开头原文及同样摘录长度。工具文案固定在 `src/minimal_rag/semantic_search.py::DESCRIPTION`。

## 预先确定的采用规则

仅当两项正确数分别B≥A，且合并1230题的逐题正确性差值(B−A)的双侧95%配对bootstrap区间下界严格>0，才保留工具。按基准分层重采样（400/830权重固定），100,000次，随机种子20261016，百分位2.5%/97.5%，每题等权。否则删除semantic_search产品入口和实现，保留实验记录。未完成、缺有效裁判或费用门槛失败时不声称满足采用规则，不默认启用。

记录各基准胜/负/平、逐题差值、分类表现、semantic_search调用总数/每题均值/调用过的题数。参考文档仅由semantic_search观察到：同一次B轨迹中semantic命中参考ID集合减去BM25命中集合非空；另报semantic首次发现参考文档及全部参考文档覆盖。这是轨迹归因，不宣称因果必需性。BrowseComp参考ID来自官方qrels，Enterprise来自原gold_ids。

## 预算和云端

总云端费用上限120元，不充值。按平台实际单价记录每次开机时间、预计上界、结束账单；吞吐门槛必须覆盖准备/建库/B服务/官方判分/取回数据，加20%时间余量。预算不足则停止并取回结果，不改模型/语料/精度挤预算。

每阶段取回结果后，在平台控制台关闭实例并确认已关机；进程退出或容器shutdown不等于停止计费。尽量利用已保存的官方8B模型；启用平台定时关机作为费用兜底。

## 命令

```bash
.venv/bin/python -m minimal_rag_experiments.benchmark --dataset browsecomp --split all --live --concurrency 5 --output runs/semantic-tool-20261016/browsecomp-A
.venv/bin/python -m minimal_rag_experiments.benchmark --dataset enterprise --split test --live --concurrency 5 --judge-concurrency 1 --semantic-url http://127.0.0.1:8765 --output runs/semantic-tool-20261016/enterprise-B
.venv/bin/python -m minimal_rag_experiments.benchmark --dataset browsecomp --split all --live --concurrency 5 --semantic-url http://127.0.0.1:8765 --output runs/semantic-tool-20261016/browsecomp-B
```

## 准备阶段记录

- 全部139项测试通过；原BM25/read、loop、codex_adapter与Enterprise A的源码摘要一致。独立工具默认不启用。
- BrowseComp A已于洛杉矶2026-10-16上午启动，PID和命令见 `runs/semantic-tool-20261016/browsecomp-A-process.json`；原始轨迹持续写入独立目录，未判分。
- 官方预建向量4片已下载；100,195个ID与本地语料完全一致，float32/4096维、抽样单位范数校验通过，来源和文件摘要见runs下browsecomp-source.json与browsecomp-index-validation.json。
- 云端首个实例因原生桌面/剪贴板阻塞，未计算就保存并关机，余额差0.35元；用户回复ok后恢复。原机恢复延迟完成，随后创建的重复实例已立即关闭、未运行计算。所有费用计入120元上限。
- 当前测试实例 gafpysbycqmrrlkj-obey，L40S 48GB，2.98元/时；开始北京时间2026-10-17 00:14:44。平台确认01:20自动关机并保存（洛杉矶10-16 10:20）。SSH凭据只在git忽略的本地connection目录，勿打印。
- 实测环境torch2.6.0+cu124 / transformers4.53.2，BF16可用。4096篇样本已上传。第一次计时在归一化改为FP32时中止，原日志保留probe-initial；正式probe已重新启动，尚未通过全量费用门槛。
- 云端目录 `/data/minimal-rag/semantic-tool-20261016`，Python为 `/data/minimal-rag/qwen-bm25-20261015/.gpu-venv/bin/python`，模型在同目录的models/embedding。实验进程及日志见probe-process.json/probe.log，完成统计见probe/state.json。

## 用户后续指令（优先于上述原预算门槛）

用户于2026-10-16要求：“直接a800开跑，不管预算了”。取消120元总预算门槛和第二轮吞吐门槛；A800权重/语料就绪后直接建511,962篇全量索引。保留原模型/向量设置、A/B配置、采用规则、首次有效判分及每阶段取回结果/平台关机要求。未收到充值授权，使用现有余额。

L40S完整4096篇实测561.568秒、5,131,178 token、7.29386篇/秒；外推全量19.497小时（58.102元，20%余量69.723元）。完整向量和日志已取回，所有向量维度/ID/单位范数校验通过，FunHPC已确认没有运行实例，本轮余额减少1.70元。

当前A800实例 `cb894da57f-8b71d0d4`，AutoDL A800专区008机，单卡5.59元/时。路径 `/root/autodl-tmp/semantic-tool-20261016`；base Python `/root/miniconda3/bin/python`，PyTorch2.8.0+cu128，BF16可用。SSH helper在 `runs/semantic-tool-20261016/connection-a800/remote.py`。当前准备进程与日志在preparation-process.json/preparation-state.json，模型下载使用同一官方冻结manifest，准备脚本正切换为全量流程。

## 持续推进与额度授权

用户进一步要求睡觉期间持续推进，并明确授权Codex额度用尽时使用重置券。账户读取确认有2次免费Full reset。`quota_guard.py --watch`通过官方app-server的account/rateLimits/read与account/rateLimitResetCredit/consume操作；只有ordinaryUsageAllowed=false且有券时重置，使用持久化幂等键，观察恢复后才允许下一次。只读校验成功，未提前消耗券。状态在runs下quota-state.json，守护/保持唤醒PID在background-processes.json；完成后写workflow-complete.json并停止守护与保持唤醒，用户要求停止时写STOP。

本聊天每10分钟自动续办已创建，automation ID为`automation`；状态不变时保持安静，阶段完成/失败/需要用户动作才通知。完成实验后停用。

A800已启动full_index.py全量流程，PID2266。旧preparation进程组已结束，模型下载被新的全量流程接管（权重已完整就绪）。完整流程并行下载Enterprise原始parquet和官方BrowseComp索引，校验冻结来源，再编码Enterprise511,962篇；没有A800预算门槛或第二轮pilot等待。远端full-state.json是阶段状态，enterprise-index/state.json是编码进度，完成包为enterprise-index.tar.gz。GPU encode支持--resume续接已完成文档分片；不要重复启动活进程。

AutoDL平台定时关机现改为洛杉矶/页面服务器时间2026-10-17 05:00，控制进程截止UTC2026-10-17 11:50（LA04:50）。续办应核实实际状态/余额/剩余时间，及时取回结果并关机；这只是无人值守兜底，并非容器退出即停计费。

全量流程第一次在数据依赖导入阶段失败：pyarrow26需要NumPy2，而推理环境固定NumPy1.26.4。已将数据读取依赖改为pyarrow20.0.0（冻结Parquet由Arrow19.0.1写出），保持推理环境不变；此前尚未开始编码。第一次失败状态保存在远端full-failed-pyarrow.json。全量流程已重启，当前PID应以full-process.json为准（重启时为2509）；不要依赖上文旧PID2266。

额度守护所用机制见[OpenAI官方App Server文档](https://learn.chatgpt.com/docs/app-server#auth-endpoints)。完整流程的NumPy/PyArrow冲突已解决，当前远端Enterprise原始文件及BrowseComp分片持续增长，full-state阶段为corpus；旧日志中保留的ImportError不代表当前进程失败，检查full-state和实际PID。

2026-10-16 17:27 UTC：下载准备阶段因hf-mirror HTTP429停止，尚未开始任何Enterprise向量编码。已保留234,881,024字节Enterprise原文件断点和全部BrowseComp部分分片；旧失败状态见远端full-failed-429.json。修复为完整文件流式续传，避免每16MB重新请求解析URL；HTTP429尊重Retry-After并退避，BrowseComp改为顺序下载且在Enterprise语料就绪后与GPU编码并行。全量流程已从原断点恢复，PID仍以远端full-process.json为准。BM25、模型、工具说明、评测代码/配置都未改。

2026-10-16 17:42 UTC：镜像持续约0.1MB/s，本地SSH直传16MiB实测1.1MB/s，改用已从现有完整SQLite导出的511,962篇语料（1,062,695,982字节gzip）。本地已校验条数和导出文件摘要，来源沿用同一冻结Parquet。`upload_corpus.py`在本地后台上传到`.uploading`临时名，远端摘要相符后原子改名并自动启动`full_index.py --prepared-corpus`，不再等待镜像原始Parquet；这只改变传输路径。旧PID3036进程组已结束，状态留存full-before-direct-upload.json。当前传输PID12299，状态/日志在runs下corpus-transfer-{state,process}.json和corpus-transfer.log；传输仍存活时不要重复启动全量进程。

`collect_index.py`已作为单实例后台回收器启动，PID12618、flock防重复；每120秒只取回远端state列出的已完成分片，到本地runs/semantic-tool-20261016/a800-results/enterprise-index。最后核对远端manifest的全部文件大小和SHA256，取回流程日志/来源。为避免额外8GB打包和结束后长时间空等，不再生成重复的enterprise-index.tar.gz；完整分片加manifest就是最终索引。回收器complete仍不代表已关机，必须检查数据校验后在平台控制台关机。其状态见index-collector-state.json。17:44 UTC平台余额111.15元，A800仍是唯一运行实例，定时关机仍为LA10-17 05:00，其他两个历史实例已关机。额度守护保持单实例且尚未使用任何重置券。

2026-10-16 17:50 UTC：完整语料直传及远端SHA256核对完成，传输进程正常结束，新的全量流程PID3770、编码PID3782已启动。17:56 UTC确认full-state为encode、GPU利用率100%、显存约24GB，首个4096篇分片尚未结束。官方BrowseComp分片也已继续下载校验，末两片日志显示verified。应等实际多个分片吞吐再调整阶段定时关机/余额安排，不增加任何开跑门槛。BrowseComp A此时432/830，全部finished、零错误；后续只依据实时summary续办。

18:16 UTC：首个A800分片4096篇已取回并校验：ID顺序与完整输入一致、4096维FP32、全部有限、最大单位范数误差1.19e-7，证据见first-a800-shard-validation.json。首片9,296,601 token用956.28秒，文本较长，不能用首片篇/秒直接外推全库。四个官方BrowseComp分片均已在远端SHA256校验完成。A组题426、434出现固定180秒模型请求超时，已保留runtime_error和完整分母，不重跑；题342的两次无效JSON工具参数由模型自行恢复，最终finished。其他在途题仍持续写入新工具/模型事件，没有持续服务中断证据。额度89%、守护存活、尚未用券。

18:36 UTC平台复核：唯一A800运行，余额105.56元，平台关机仍为LA10-17 05:00。两片8192篇已编码并取回，20,682,448 token用2154.23秒（约9601 token/s）。固定4096篇随机样本推算全库约6.41亿token，按当前token吞吐粗估总编码约18.5小时，可能超过现有UTC11:50控制期限；后续根据更多分片和实际余额，在完成分片边界续接相同配置并调整控制期限/平台定时关机。不能拿按顺序的前两片篇/秒外推37小时，也不能把这项安排变成新的吞吐或预算开跑门槛。

19:03 UTC普通Codex额度耗尽，单实例quota_guard已成功消费1次授权免费Full reset（outcome=reset）；额度恢复，尚余1券。A组题772、773在恢复前触发usage limit，按原规则永久保留失败。原A进程93395等其他在途题结束后正常退出，共701个终态trace，无未完成trace；19:07 UTC按完全相同配置加--resume续跑剩余129个未启动题，没有重生成前701题。当前PID以browsecomp-A-process.json为准，旧PID勿再使用。既有2题超时加2题额度失败均留在830分母。GPU全量编码继续；19:06 UTC已有16,384篇、42,784,239 token，约9552 token/s，GPU100%。

19:47 UTC：A组尾段新增题983超时及969、979、985的SDK `no active turn to interrupt`，共8题runtime_error；保持原配置和一次生成规则，不重生成。全部原始错误另存browsecomp-A-runtime-failures.json，SDK取消问题记入docs/issues.md供本轮之后处理。官方判分源文件已逐字节对照冻结official.tar.gz核验55个文件，并准备runs下judge-bundle、judge-support和judge-bundle-manifest.json；尚未加入答案组、未启动GPU判分。来源版本仍为046949032b0328319cc9a02663a759ec601d9402，Qwen3-32B BF16、TP2及原采样参数不变。历史同环境A830判分约274秒，下载/环境准备另算；仅用于安排资源，不能当成本轮实测。

19:56 UTC确认BrowseComp A生成阶段完整结束，进程27624已退出。830个终态trace及830个官方导出逐题核对一致；822题finished、8题runtime_error（3次TimeoutError、2次额度错误、3次SDK取消错误），全部保留在830分母，禁止再启动A生成。共6003次生成请求、10441次search和2843次read，未调用semantic_search。最终核验见browsecomp-A-generation-validation.json；A导出已原样加入judge-bundle/runs/A。尚未判分，822不是答对数；待B完成后统一使用冻结的官方裁判。报告会另列涉及生成失败的配对胜负，采用规则仍按全部1230题，不改分母或置信区间规则。云端此时32,768篇向量已完成并取回，GPU持续100%，额度守护与回收器存活；最近平台余额99.97元，定时关机LA10-17 05:00未变。

20:19 UTC保护窗口调整：前10片40,960篇、85,095,702 token持续编码正常，但按固定样本全库token量估算，结束时间接近旧控制期限。平台已确认自动关机改为LA/页面服务器时间10-17 06:00（UTC13:00），截图a800-timer-extended.png；页面预计至该时刻93.35元、余额99.97元，没有充值。远端extend_deadline.py（PID9375）等待下一个完整分片后结束旧进程组，再用原模型/批次/向量设置和--resume续接，控制截止改为UTC12:50（1792241400）。查看extend-deadline-state.json及full-process.json确认完成前，不要重复启动或假定新PID已生效。回收器为匹配新窗口已单实例重启为PID35902，原PID12618已结束，停止时间现为UTC13:50；已回收分片不会重取。该窗口仅为保护上限，索引齐备并取回验证后仍立即在平台关机。

20:28 UTC已在第11个完整分片（45,056篇）完成无配置变化的续接；20:36 UTC确认新控制PID9725、编码PID9737实际运行，截止UTC12:50已生效，GPU95%、前11片已取回。旧3770/3782和保护窗口观察器9375均已退出，extend-deadline-state为complete。不要再按旧PID检查或重复启动；full-process.json仍是权威入口。

22:26 UTC为核对排期，只读扫描完整输入的字符长度（没有暂停或改变编码）：全库511,962篇平均4854字符，固定随机样本平均4819字符，仅相差0.73%；按顺序开头1万篇平均10722字符，尾部明显更短。细节见corpus-length-profile.json。该检查支持继续使用固定随机样本估计总体工作量，而不是用当前前缀文档的篇/秒外推全库；字符统计不是精确token总量，也不构成新的运行门槛。

用户于10-17约00:24 UTC新增明确指令：“能不能多开几张A800并行建库，把A/B对比结果尽快跑出来？”已开始执行，目标总计4卡并行；现有单卡继续计算至资源方案确定。保留全部已完成分片，以原始4096篇分片编号对剩余分片轮询分配，不改变模型、BF16/FP32精度、token长度、批次排序或向量设置。原实例扩卡需关机，当前正在检查同机库存；必要时使用平台同地区克隆并拷贝数据盘，复用模型和环境。parallel_index.py已准备但尚未启动，任何后续续办先核对实际平台和full-process状态，勿重复租机或启动。

00:31 UTC并行迁移中：008主机只空闲2卡，无法原地升到4卡；专区其他主机有5/6/8张空闲卡，改为同地区克隆到4卡主机。已停止9725/9737并保留114,688篇、28个完整分片，全部分片及checkpoint日志已在本地；没有未登记的NPZ残片。源实例cb894da57f-8b71d0d4已在平台确认已关机，证据a800-single-stopped-for-parallel.png。旧SSH凭据备份在connection-a800-single；回收器35902已暂停（background-processes.json注明），不要在迁移期间自动重启旧连接。已在克隆对话框勾选系统盘＋数据盘，目标选择中；须等真实新实例创建/拷贝就绪后更新当前连接、启动parallel_index.py、重启回收器，并把本续办的旧单卡指令更新为实际新实例。

## 用户追加的运行错误重试规则（覆盖原一次生成约束中的技术失败处理）

用户要求“BrowseComp A组报错的8题重跑并加入结果，不算失败”，并明确选择“A、B使用相同的运行错误重试规则”。因此保留全部原始尝试，runtime_error/interrupted可按完全相同模型、提示、工具、轮数及超时设置重试；首次正常结束的结果即采用，不依据答案对错选择，已经正常完成的答案不重生成。模型自身轮数耗尽等非技术终态仍按原规则计分。首次有效裁判不重判；缺失裁判继续原恢复流程。Enterprise A仍复用原309/400（原400题均正常完成）。采用门槛、完整1230题分母和bootstrap规则不变，但最终报告必须注明这是用户在B开始前追加的协议修订，初始失败率及全部重试用量单列。retry_generation.py为实验调度外层，不改benchmark.py、核心循环或裁判；各轮存canonical run/generation-retries，成功后更新主trace及官方导出，原失败永存initial-failures。A组初始822正常/8失败的记录不删除，更新后成绩用重试成功的答案判定。

00:37 UTC四卡已实际开跑：新实例d6f74e95d2-148d7d42（048机），4×A800，22.34元/时，系统盘和数据盘已完整克隆，源cb894da57f-8b71d0d4在平台确认为已关机。当前connection-a800/remote.py已指向新实例。parallel_index.py控制PID1258，worker 1325/1326/1327/1328分别使用GPU0/1/2/3；现场利用率100/95/94/100%，不是仅创建实例。114,688篇/28片保留，剩余原始分片按编号轮询分配；输入gzip摘要、SETTINGS和克隆环境已验证。回收器新PID49285，状态watching/encode_parallel。平台保护关机为LA10-16 20:35（UTC10-17 03:35），控制截止UTC03:33；创建前余额69.11元，保护窗口预计67.02元，不充值。观察首批四卡分片及余额，必要时在真实状态基础上处理，完成后取回并立即平台关机；后续检索服务降为单卡。自动续办已改为每5分钟，提示也已更新到新实例与A/B共同技术重试规则。

A组8题重试已启动，进程以browsecomp-A-retry-process.json为准（启动PID49645）。benchmark.py源码摘要和所有settings逐项与原A一致；仅筛选8个runtime_error题进入单独重试目录，目前round-001。先检查进程及generation-retry-state.json，不重复启动。完成后核对830条全量主trace/官方导出并更新judge-bundle-manifest中的A摘要；原browsecomp-A-generation-validation.json保留为初次生成记录。

00:49 UTC A组8题已全部在第一轮重试中正常完成并合入主结果，现830/830 finished、0个未解决运行错误，仍未判分（不是830题全答对）。本轮额外71次生成请求，含初次尝试共6074次；8份初始失败完整保留，round-001全轨迹保留。主trace、830个官方导出和judge-bundle/runs/A已逐题核对，A的判分包摘要已更新；证据browsecomp-A-after-retry-validation.json。retry控制已complete，不再启动A生成或按初始8失败状态重跑。B组之后采用完全相同技术重试政策。

用户新增指令：建库完成后，Enterprise与BrowseComp的B组也并行跑，每组配置不变。因此在索引完整取回/校验、建库阶段平台关机确认、单卡双索引服务及隧道就绪后，同时启动两个本地benchmark进程；每组仍生成并发5，Enterprise原裁判并发1，模型/提示/轮数/超时/样本/裁判不变。start_b_groups.py已准备，启动前会核对health全库数量和各A组settings；不会在当前建库未完成时调用。命令为`.venv/bin/python experiments/semantic-tool-20261016/start_b_groups.py --url http://127.0.0.1:8765`。各组进程记录分别在enterprise-B-process.json、browsecomp-B-process.json，已有活进程不重复启动，完整初次生成后再按共同政策恢复运行错误题。

00:59 UTC四卡首批原始编号28–31已全部取回验证：16,384个ID逐条匹配原始输入对应范围，4096维FP32、无NaN/Inf、最大单位范数误差1.79e-7，见parallel-first-batch-validation.json。四个worker首片各约829–842秒；当前实际完成131,072篇/32片（约2.64亿token），继续计算。发现SFTP单流约0.69MB/s，legacy SCP的16MiB实测13.17秒（含握手，1.27MB/s）；回收已切换到同凭据目录的remote-scp.py（仅传输协议变化），单实例回收器新PID51086，旧49285已结束。首次已取回32片均保留，后续以background-processes.json为准。FunHPC余额复核44.19元、租用0；仅检查了后续资源入口，没有额外租用等待中的GPU。

01:22 UTC启动只读CPU token总量统计（count_corpus_tokens.py；远端token-count-process.json，启动PID3606），4个低优先级CPU线程、最多30分钟，不加载GPU模型，不暂停四卡。使用原tokenizer、title+text、4096截断及相同特殊token设置；完毕会对照保留前28片的token计数。token-workload.json的status=complete才是精确总量，未完成值只是部分进度。此项只校正排期/余额安排，不是新的吞吐门槛，不影响编码或评测设置。完成后取回该JSON和日志，并据各worker已完成token/剩余分片估计剩余时间。

01:31 UTC CPU全量统计已完成并取回：511,962篇、646,186,281 token，耗时400.92秒；前28片token与原向量记录完全相符，未初始化CUDA。证据a800-results/token-workload.json与token-count.log。按各worker累计token吞吐估算，四个worker约UTC03:25–03:40完成；这只是排期估计（短文尾段速度会变化），最慢估计略晚于03:33控制截止，须继续按实际吞吐和平台余额调整，不能充值。精确工作量和逐worker排期保存在parallel-schedule-estimate.json。

回传有数片积压，已在旧回收器无在途传输时单实例切换为3路并行legacy SCP（PID53337），只取已完成不可变分片；旧51086已退出，全部已回传文件保留。当前PID仍以background-processes.json为准。新增validate_index.py用于最终manifest齐备后一次性检查全部125片的摘要、511,962个ID顺序/唯一性、4096维FP32有限单位向量；尚未对不完整索引执行最终验证。完成后的命令为`.venv/bin/python experiments/semantic-tool-20261016/validate_index.py`，成功生成enterprise-index-validation.json后再平台关机和部署单卡B服务。

01:48 UTC四卡已完成229,376篇/56片，全部GPU100%。回传暂时落后数片；一次SSH认证失败已自动重试，文件持续增长，不是编码失败。对现有完整分片做纯本地无损压缩测量：67,715,578字节可压至36,022,640字节（53.2%），CPU耗时1.32秒。已为remote-scp.py增加SCP的-C传输压缩，新传输自动生效、旧传输不中断，回收器PID仍53337。它仅压缩SSH传输流，到盘NPZ字节完全保持原样；模型、向量精度和检索均不变。证据transfer-compression-probe.json；最终仍按manifest摘要核对全部文件。

01:58 UTC四卡249,856篇/61片，正常运行；SSH在重新登录时偶发Permission denied，原凭据下一次即成功，回收器保留完成分片并重试。为减少重复认证，connection-a800两个helper现共同使用ControlMaster=auto、ControlPersist=120、Compression=yes，控制socket位于权限700的/tmp/minimal-rag-ssh-<uid>/%C（按主机/端口/用户区分）。已成功建立压缩连接并取回CPU统计进程记录，未更改凭据或放宽主机密钥检查。后续确认回传恢复即可，不要重启GPU编码。

用户最新要求“减少轮询和心跳检查的频率”。常规自动续办从每5分钟降至每15分钟，每次合并一轮必要检查，不在正常推进时同一轮反复轮询；平台余额/关机保护常规至多每小时核对一次。阶段切换、预计完成或控制截止前20分钟内、真实错误时可临时加密必要检查，处理后恢复低频。建库、持续回传、免费额度守护正常继续，未暂停实验。

02:35 UTC四卡364,544篇/89片，四卡100%，回收85片并继续传输。回传连接的ControlPersist改为600秒，覆盖回收器每轮120秒等待，避免共享连接恰在下一轮前到期后重复认证；不关闭现有连接，后续新连接自动生效。编码配置与进程不变。常规心跳仍15分钟。

03:05 UTC建库进入尾段：462,848篇/113片、596,722,377 token，四卡100%；回收104片且继续传输。平台余额已结算为14.72元，唯一运行实例仍是四卡，定时关机仍LA20:35（UTC03:35），未充值。当前估计最慢worker接近03:32完成，控制截止03:33。下一轮已进入完成/关机前20分钟窗口，须重点确保最后分片回传和全量校验留有时间；必要时仅在现有余额内调整平台保护（按03:00余额计算约可支持至03:39，不可盲目延长至04:00）。检索服务阶段可使用FunHPC已确认的44.19元现有余额。先取回并确认建库平台关机，再恢复单卡双索引服务和并行B组。

03:20 UTC剩最后4片：495,616篇/121片，worker3已complete并释放GPU，其他三卡仍计算；本地113片。已在当前四卡实例的更多菜单实际看到“无卡模式开机”。若计算完成后完整回传尚未结束，先取回完成manifest/日志并平台关闭四卡，再对同一实例用无卡模式完成回传/全量验证，最后再次平台关机确认；这是阶段内数据回收安排，避免让GPU空转等下载，仍需完整取回和关机才算阶段完成。届时核对无卡价格和SSH信息，不充值，不重开原单卡源实例。当前进入截止前20分钟窗口，续办临时改为5分钟；阶段切换后恢复15分钟。

03:32 UTC全量向量计算成功完成：511,962篇、125片、646,186,281 token，最后索引清单与全流程日志已打包取回a800-results/completion-logs.tar.gz并解包，index-computation-complete.json已核对计数。取回当时本地尚缺120–124这5片。四卡实例d6f74e95d2-148d7d42已在平台确认已关机（a800-four-gpu-stopped.png），随后同一实例以无卡模式启动：0 GPU、0.5 CPU核、2GB内存、0.10元/小时；SSH复核/dev/nvidia0不存在且远端完整索引仍在。平台无卡保护关机改为LA10-16 21:00（UTC10-17 04:00），截图a800-cpu-retrieval.png。原单卡源实例保持关机。回收器53337继续使用当前connection-a800取剩余文件；完整取回后运行validate_index.py，再平台关机确认，随后恢复单GPU服务并同时启动两组B。旧编码PID和03:33截止已不再适用，切勿重启计算。private helper已增加tunnel子命令，固定仅转发本地127.0.0.1:8765，尚未启动服务隧道。

03:41 UTC完整125片已全部取回，回收器53337正常complete退出。validate_index.py用14.29秒完成全量校验：511,962个唯一ID与原始输入逐条一致、4096维FP32、全部有限且最大单位范数误差1.79e-7、所有文件大小和SHA256匹配最终manifest。证据enterprise-index-validation.json；manifest摘要465afb668cbb56167758dbee69d3191e96003421e6f10620a6cb6d2ebe484f48。无卡实例随后已在平台确认关机，index-stage-all-stopped.png显示所有AutoDL实例均已关机，建库/回收阶段完整结束。当前已向FunHPC提交semantic-wait-1016快照的原机恢复请求并确认，须先核对“我的租用”的真实状态，避免重复恢复/创建；尚未确认新服务实例ID与SSH，尚未启动B。

FunHPC原机恢复请求在03:42:42 UTC返回HTTP400，实际租用列表为空；未再次确认同一请求。改选rag-qwen8b4b-result快照的换机恢复入口，尚待选择目标和确认实例。建库完整结束后，续办频率已恢复15分钟。

04:11 UTC服务部署已实际运行：FunHPC原机恢复HTTP400；L40S/A800售空，且A100可用MAKE集群无法挂载OBEY/SNOW快照，因此创建单卡A100-PCIE-40GB实例loygrhucqytximef-make，2.86元/小时，开始北京时间11:47:56，保护释放并保存/data为北京时间10-17 19:30（UTC11:30，LA04:30），证据a100-service-timer.png。40GB足够8B BF16模型和两套FP32矩阵，模型/检索/评测配置未改。当前connection/remote.py已私下更新新实例凭据，旧L40S连接备份connection-previous-l40s。FunHPC密码复制必须确认“已知悉并复制密码”提示，初次误复制旧剪贴板后已纠正，实际SSH登录已成功。不要输出凭据。

真实PyTorch解释器为/data/miniconda/envs/torch/bin/python：Python3.10.16、torch2.6.0+cu124、CUDA12.4、A10040GB、BF16=true。/opt/setEnv.sh不是激活脚本，而是平台环境和密码初始化脚本，不要再执行；之前一次试调用因缺少平台变量发生下载/初始化错误，密码初始化也报错，后续现有平台密码仍能正常登录，日志保留activation.log。现在直接用正确解释器创建/data/minimal-rag/semantic-tool-20261016/.gpu-venv。固定transformers4.53.2/numpy1.26.4加传输用paramiko3.5.1已通过阿里云PyPI镜像完成安装；初始慢源日志保留*.initial。系统apt镜像403，未放宽验签或修改apt源，改用Paramiko传输。

官方冻结模型从相同models.json用download.py下载，PID495，models-ready.json为完成凭据，下载中已超过10GB；download.py仅改用等价流式SHA函数兼容Python3.10，权重来源/文件摘要未变。为快速部署，当前d6f74e95d2索引源再次以无卡模式用于云间直传，0 GPU、0.10元/小时，保护关机LA21:30（UTC04:30），截图a800-cpu-direct-transfer.png；这是新服务部署阶段，之前建库阶段完整回收与全部关机记录仍保留。旧原单卡cb894da57f保持关闭。源凭据只临时放在目标/tmp，保留原known_hosts严格校验，传输完成/失败会清理，不进/data保存快照；本地一次性凭据压缩包已清除。

目标远端部署控制deploy_query_service.py PID697，截止UTC05:25；先等依赖，再启动transfer_indexes.py（当前PID763），三条复用SSH/SFTP会话直传125个Enterprise分片和4个官方BrowseComp分片并逐文件校验原摘要；状态index-transfer-state.json，完成index-transfer-ready.json，失败保留日志。随后等待模型ready并自动加载单卡双索引服务，query-service-process.json和service-stage-state.json记录实际服务PID与health。根目录/data/minimal-rag/semantic-tool-20261016，helper仍是本地connection/remote.py。

本地localhost隧道守护PID62375（query-tunnel-process.json），仅127.0.0.1:8765，掉线5秒后重连；该隧道独立SSH连接，避免复用控制连接退出而丢失转发。就绪启动器await_query_service.py PID62376（service-launch-process.json）每30秒检查本地health；随后用两个普通annual report查询检查接口和ID存在性，不使用任何基准题，结果保存query-service-preflight.json；通过后自动调用原start_b_groups.py同时启动两组B。不要重复手动启动两组或守护。B实际开始必须以各B-process.json和实际进程/结果确认，目前尚未开始。源CPU直传一完成就必须在AutoDL平台关机；如果传输预计超过UTC04:30，在现有余额内调整CPU保护，不充值。常规续办仍15分钟。

04:16 UTC云间直传已核对前16/129文件约1.08GB，约6.4MB/s，随后文件持续写入。根据实际速度将无卡源平台保护延长为LA22:00（UTC05:00），页面估计只增加0.07元以内，仍是0GPU/0.10元每小时，传完即关机。模型和索引齐备后的自动服务启动、普通查询预检及两组B自动并行启动已经部署；启动前仍以实际ready/preflight/process记录为准，未声称B已开始。续办指令已更新新A100、正确运行时、所有守护PID和CPU直传源清理要求，正常频率15分钟。

04:28 UTC两套索引已在目标完成129个文件的全部原SHA校验，临时源凭据已清理。单卡服务PID910真实ready（Enterprise511962/Browse100195），普通annual report查询预检通过且ID均在各本地库，query-service-preflight.json未使用任何基准题。启动器于UTC04:28:53同时启动两组B（最初PID63140/63141）。部署证据已取回service-deployment-evidence.tar.gz并解包。AutoDL无卡直传源已随后在平台确认关机，direct-transfer-source-stopped.png显示所有AutoDL实例均关闭；只剩Fun单A100提供检索服务。

B首轮刚启动遇到生成服务集中HTTP429限流。周额度仍ordinaryUsageAllowed=true、已用62%、还剩1张免费重置，不能提前消耗券来处理429。已对两组发SIGINT停止新增题目，停止时Enterprise6正常/70技术失败，Browse4正常/49技术失败；停止时在途任务的取消/TransportClosed也全部保留，不删除原轨迹。原因来自Codex生成上游，向量服务的数量/设置/真实查询预检均正常。完整事件记录B-runtime-incident.json；旧进程记录保存在*-B-process-history.jsonl。

为恢复用户要求的两组并行且每组配置不变，新增仅实验外层的transport_control.py与rate_limited_run.py：两个进程共享最多5个正在执行的模型请求，开始时间至少间隔2秒；HTTP429触发120秒冷却并让本轮调度停止接收新题，之后只resume未启动题。原BM25/read/核心循环/codex_adapter/benchmark/系统提示/模型/裁判源码及参数均未改；各组concurrency仍5、Enterprise judge_concurrency仍1。该操作只对上游请求排队，不在失败会话内偷偷续生成。模型调用超时仍180秒，原case_timeout与轮数不变；排队时间计入题目墙钟时间，因此最终报告必须披露该基础设施恢复措施，不能将A/B耗时当作完全同口径效率对比。transport-policy.json与transport-events.jsonl记录策略/429冷却。

两组当前由新包装进程恢复：Enterprise PID66986、Browse PID66987，以各*-B-process.json及实际进程为准。包装器先完成初始全量（只补未启动题、保留正常答案和首次有效裁判），再自动调用retry_generation.py只恢复runtime_error/interrupted，原错误永存generation-retries，最后补齐正常答案缺少的首次有效裁判。不重复启动包装器或手动并跑重试。生成失败对应的“Generation failed”占位判决随该技术失败尝试一起保留，恢复正常答案后用原裁判获得首次有效判决；已经正常完成的答案与有效判决不改。首次恢复仍见少量429，已由共享冷却吸收，未再出现原先的快速失败连锁；下一轮重点比较正常终态的增长和新增transport-events。若持续无有效进展，处理上游阻塞并避免GPU长期空等，不能靠提前重置或购买额度。继续保持15分钟心跳；报告须包含该运行事故、重试用量和原固定采用规则，不改1230题分母或统计门槛。

用户要求查明429并加快、避免GPU空等。05:07 UTC诊断：已找到原始Codex订阅端点HTTP429及Retry-After=1–11秒；ordinaryUsageAllowed=true、周额度68%、1张免费券未动。保存的114次生成429全部在新会话第一轮，后续轮无429，独立非基准探测成功；不能据此声称已知RPM或并发阈值，具体错误子类尚待捕获。证据429-http-response-metadata.json、429-diagnosis.json、429-probe-result.json。实验transport_control.py新增结构化TurnError捕获（SDK原来仅抛message），全局5个调用/2秒间隔保持，仅新会话第一轮串行且至少间隔10秒；冷却从固定120秒改为30秒，短期再次失败递增至120秒并尊重捕获的Retry-After。模型/提示/工具/轮数/超时/每组并发5/裁判并发1不改。Enterprise原66986已在无子进程的自然批次边界退出，没有SIGINT取消在途题，新PID71807已续跑；Browse原66987仍在执行已启动题，boundary watcher71060等待其自然批次边界，若退出须运行start_b_groups.py补启动缺失组（不会重复启动活组）。不要把磁盘新policy当成旧Browse进程已经热加载。原策略存transport_control.initial.py，所有原尝试保留。GPU查询日志仍持续更新，无需关掉正在服务的卡；若上游持续无有效请求/正常答案推进，应取回查询日志并平台关机，不能让GPU空等。

05:11 UTC新策略初步验证：17次模型调用成功（涵盖high生成和medium原裁判），0个新429；正常答案Enterprise35、Browse39，不能据短窗口宣称429彻底消失。旧Browse仍正常产出，未强行中断。边界迁移观察器已替换为PID72312，文件diagnostic-boundary-watch-process.json，最多900秒；找到旧66987无子进程的自然边界时会终止旧包装器并自动调用start_b_groups.py，新Enterprise71807会被识别为活进程而跳过。若观察器到期，旧Browse会继续正常运行，下一轮按实际状态再决定，无需盲目重启。自动续办新增15分钟无成功推理且GPU查询停滞即保全日志、停止B并平台释放保存/data的处理规则；常规仍每15分钟。report.py已纳入事故/诊断/策略与墙钟比较限制。

05:28 UTC一轮合并检查：Enterprise71807已57正常/74技术失败，旧Browse66987已84正常/51技术失败。新策略累计155次模型调用成功、0个429；UTC05:26旧Browse出现一次429，其旧120秒冷却暂影响共享队列，但服务查询最近约100秒内更新，并非长期空转。边界观察器72312在05:24超时正常退出，旧Browse继续运行；趁当前失败批次将自然结束，已单实例重新启动同一边界迁移脚本，PID76814（以diagnostic-boundary-watch-process.json为准），自动安全停止旧66987后start_b_groups.py，切勿另起重复B。Fun控制台余额39.52元，唯一租用仍1个，semantic-B-single-gpu定时释放保存/data仍为北京时间19:30/UTC11:30；未充值。额度75%、ordinaryUsageAllowed=true，守护6972与隧道62375存活，无提前重置。

05:43 UTC调度迁移完成：旧Browse66987已于UTC05:30:21在自然无子进程边界退出，新Browse77438启动成功；Enterprise71807保持运行，没有人为取消在途题。边界观察器76814已结束，不要重启迁移脚本或旧PID。最近15分钟206次成功模型调用、0个429；当前Enterprise74正常/76技术失败，Browse99正常/54技术失败。新增5次均为原180秒模型请求TimeoutError（非case排队超时），保留原尝试，由全量后的共同技术重试流程处理；未改变超时或模型。quota守护一次InternalRpcError已自行恢复watching，78%且ordinaryUsageAllowed=true、余1张券，guard6972存活；历史error字段可能残留，须看status和checked_at。GPU日志约1秒内更新，不应关机。429-diagnosis.json已更新实际PID和验证结果。

06:33 UTC修复一个真实实验外层排队错误：发现若干Enterprise题等待900秒却model_runs为空，说明新会话锁的轮询抢占使部分等待者一直未获调用机会。不能把这些当模型失败或加大超时。已保留原始记录（fifo-recovery-incident.json逐题列出），对71807/77438发SIGINT停止继续调度，保留所有在途尝试并按共同技术重试政策处理；两旧进程已退出。仅修改transport_control.py为新会话锁与5个活跃调用槽分别使用共享FIFO队列，取消/超时/死进程票据清理；5活跃、2秒调用间隔、第一轮串行及10秒新会话间隔、原180/900/1800秒超时均不改，原核心源码/提示不动。隔离临时目录的6等待者功能验证确认取消第2个后顺序仍0,2,3,4,5且无残留票据。两组已原配置resume：Enterprise85609、Browse85610，以各process.json为准。新model_call_started日志增加queue_seconds，下一轮重点看排队时间和是否再有零请求case超时；report.py纳入该基础设施事件。GPU服务与隧道未重启，短暂停顿后继续工作。

## Latest user priority: pause BrowseComp; complete Enterprise first

The user explicitly requested stopping BrowseComp and using all remaining model allowance on Enterprise. At UTC07:33, BrowseComp wrapper85610 received SIGINT; its parent and child processes were verified exited. Its234 normal answers,67 runtime errors,3 already-marked interrupted cases and other partial traces are retained. See enterprise-priority-change.json. The persistent browsecomp-B.PAUSED marker makes start_b_groups.py skip BrowseComp. Do not remove that marker, restart BrowseComp, retry its cases, or run its official judge until the user explicitly resumes it.

Enterprise wrapper85609 continues with unchanged settings. At the priority change it had225 normal answers of400. Finish its initial full pass, retry technical errors under the agreed policy, and obtain the first valid original judgement for every selected answer. A remains309/400. All remaining model allowance is for Enterprise; the existing quota guard may consume the final free reset only when ordinary usage is exhausted, never early or by purchasing credits.

Run report.py --enterprise-only only after all400 Enterprise answers and valid judgements are complete. It writes enterprise-comparison.json with the400-question paired interval, categories, paired wins/losses, tool use and reference-document attribution, and all-attempt accounting. This does not overwrite the full comparison.json or execute the original1230-question adoption gate: decision=deferred_pending_browsecomp. Preserve the original two-benchmark adoption rule for a later user-authorized BrowseComp continuation. Disclose the429 incident, FIFO admission repair and technical retries; wall time is not fully comparable to A.

Once Enterprise generation and technical retries no longer need retrieval, retrieve service logs and release the FunA100 through the platform with /data saved; verify billing stopped. Do not keep GPU running for local judging/report work. After the complete Enterprise report and platform shutdown, write enterprise-phase-complete.json, pause this automation, and stop this experiment's quota guard, tunnel and caffeinate. Keep BrowseComp paused. Do not write the overall workflow-complete.json or describe the original two-benchmark experiment as complete.

UTC07:53 continuation: Enterprise288/400 normal,88 runtime_error and5 interrupted pending agreed retries; ungraded144 is not a final score. Existing85609 remains active; Browse stays paused. Guard6972 consumed the final authorized free reset only after ordinary usage exhausted at epoch1792223068.4432561; official outcome=reset, current usage2%, ordinaryUsageAllowed=true, available_resets=0. Total free resets used2; no purchases. Fun console balance33.23CNY, one rental, release/save-data timer still UTC11:30. The cooldown logged at07:32:47 during the user-requested Browse SIGINT is a false429 substring match inside a numeric shell-snapshot filename in TransportClosedError, not actual HTTP429. Raw event is retained and annotated in429-diagnosis.json; nonblocking classifier issue recorded in docs/issues.md.

UTC08:13: Enterprise initial generation has terminal records for all400 questions:307 normal,88 runtime_error,5 interrupted. This is NOT a final score.62 initial valid judgements remain pending; the existing wrapper85609 is completing those and will automatically enter technical generation retries for93 cases. Do not start another retry process or apply the adoption gate. GPU retrieval remains needed for the upcoming retries; complete the existing generation/retry phase and then stop billing before final local-only report work.

UTC08:28: All307 initially normal Enterprise answers now have their first valid original judgement. The existing wrapper85609 automatically entered generation-retries/round-001 for exactly93 technical cases. Round snapshot:16/93 normal answers,5 valid judgements,0 new generation failures; canonical summary still shows307 normal plus93 technical placeholders until the whole round is merged. Do not mistake canonical complete=true/ungraded=0 for final completion. Read the retry-round summary/traces and retry-state as well. Once this round (or later recovery rounds) supplies first-normal answers for all93 pending IDs with no remaining technical failures, retrieval is finished even if round judging/merge is still running: retrieve the query-service logs and stop the FunGPU in the platform then, leaving local original judging running. Do not stop the GPU while any recovery answer still needs generation.

UTC08:43: retry round001 now72/93 normal,25 valid judgements,zero new generation failures. Remaining21 answers are close to finishing; heartbeat temporarily5min for the GPU shutdown transition, then restore15min or pause on Enterprise-phase completion. Canonical complete=true still includes original technical placeholders and is not final. When all93 recovery IDs have first-normal terminal answers and no technical generation failures remain, retrieve logs and release/save FunGPU via console even if round judging/merge is ongoing; keep local85609 running. Do not stop retrieval while any recovery generation remains.

UTC08:52 GPU phase complete: independently verified all400 first-normal answers (307 original plus93 in retry round001), no remaining technical generation failures. See enterprise-generation-complete.json. Final service logs and metadata were retrieved in query-service-final.tar.gz and extracted under query-service-final/. Tunnel62375 and its SSH process group were stopped. FunHPC rental list is empty after Save data and release; my-stopped-data lists rag-semantic-1017(loygrhucqytximef-make), saved at Beijing16:52:07, currently saving. Screenshot enterprise-gpu-stopped.png and enterprise-gpu-shutdown.json prove platform release. active_instance=null in ledger. All cloudGPU instances are now off; do not restart any for local judging/report work. Wrapper85609 is still running original judgement and will merge round001; do not interrupt it. Heartbeat restored15min. After400 normal canonical answers and400 valid original judgements, audit records, run report.py --enterprise-only, verify snapshot save complete, deliver report, stop guards6972/6973 and pause automation. Browse stays paused, overall two-benchmark adoption remains deferred.

Enterprise phase COMPLETE:400 normal answers and400 first-valid original judgements, all800 A/B traces validated. A309/400(77.25%), B305/400(76.25%), delta-1pp; paired18 wins/22 losses/360 ties,100000 bootstrap seed20261016 CI[-4,+2]pp. Semantic100:A68/B64,7 wins/11 losses, descriptive subgroupCI[-12,+4]pp. Tool used159/400 questions,276 calls,38 questions with semantic-exclusive reference IDs within the observed B trace.93 technical cases recovered in one round. Results enterprise-comparison.json, subgroup intervals and enterprise-evaluation-validation.json; final provenance enterprise-phase-complete.json. No promotion/removal decision for the original incomplete two-benchmark experiment; recommend primary BM25 based on this run. FunGPU already platform released,22GB rag-semantic-1017 snapshot now save-complete, proof enterprise-snapshot-saved.png. All cloudGPU instances off; wrapper, tunnel, quota guard and caffeinate verified exited. Automation PAUSED. BrowseComp remains paused and must not resume without explicit user instruction. Do not write overall workflow-complete.json.

FINAL ARCHIVE: User terminated the experiment and abandoned the added vector index/tool. Product semantic_search and the shared --semantic-url entry are removed. Current BM25/read, core loop, adapter, metrics and judge sources match the frozen experiment core hashes.48 targeted tests and an isolated clean-wheel build passed; archived B source hashes match the execution configuration.3334 evaluation/validation files are preserved in evaluation-records.tar.gz, with a separate63-file source snapshot. No SSH credential directories are included. Large vector shards remain historical archive material only. All experiment processes and cloud instances remain stopped; automation isPAUSED with an archive-only prompt. See archive.json; do not resume old instructions.
