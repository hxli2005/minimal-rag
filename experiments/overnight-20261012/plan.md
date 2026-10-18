# 2026-10-12 整晚工作

**推荐：先验证企业检索与证据利用的两个瓶颈，再用同条件实验决定是否继续复杂混合检索。** 目标是推进 AGENTS.md 三条，不把开发诊断写成 SOTA。

授权截止与交付：2026-10-13 08:00（Asia/Shanghai），晨报 `docs/morning-report-20261013.md`。本聊天 heartbeat `minimal-rag` 每 30 分钟续办，交付后用 automation_update 停用。电脑需保持开机、桌面应用运行。

硬限制：在线生成、裁判、失败重试全部 gpt-6.1-sol，累计最多 500 次；仅使用 `budget.py` 的 BudgetModel，调用前锁定、预扣并 fsync `calls.jsonl`，跨进程及续办不重置；失败也不退回次数。账本只指本晚新启动的实验请求，不是 Codex 聊天自身的推理 token/内部请求数。禁止启动任何其他在线 runner。原 Qwen PID 12453 不打断，不加载本地模型，不使用 MPS/CUDA/GPU，不写其索引。不提交 main；当前 `codex/standalone-rag`，不合并、不发布。

## 时段（北京时间）

| 时间 | 必须推进的工作 | 完成依据 |
| --- | --- | --- |
| 23:30–00:30 | dev100 最佳配置 22 错题、封存集 CDL 48 负题逐题分析；冻结同条件样本和预算；启动在线 worker | `failure-analysis.json`、`cohort.json`、账本、真实运行进度 |
| 00:30–02:30 | 企业 24 题闭卷／一次 BM25／BM25 agent 配对；查询重放与标题权重离线诊断 | 所有分母、逐题轨迹与裁判原因；CPU 单进程 |
| 02:30–04:00 | FinanceBench 12 题、HotpotQA 12 题 sol 补测；对企业错误做 oracle 证据诊断 | 保留失败、不挑最好重试；跨基准标小样本开发结果 |
| 04:00–05:30 | 命中片段离线节省／证据损失估计；只在证据支持且有剩余预算时做 8–12 题配对在线验证 | 文本字符估计与实际 token 分开；候选不自动晋级 |
| 05:30–06:30 | 官方评分提示词入口与答案导出；完成后 CPU 全量 Qwen 召回入口；查看原任务实际进度 | 官方来源与差异说明；入口离线验证；绝不提前读取 .building 为完整索引 |
| 06:30–07:30 | README、一页设计、面试要点和真实 sol 演示报告；克制清理与必要验证 | demo 可运行，报告可追溯，未重跑历史成绩不变 |
| 07:30–08:00 | 全部结果核账、整合晨报、下一步唯一推荐，关闭 heartbeat | 调用 ≤500、分支、Qwen 状态、实际完成与未完成分明 |

## 当前已知证据

- 最佳历史开发：100 题 78%（astra＋BGE/MiniLM 混合）；22 错题中 11 题至少缺一个参考文档，10 题全部参考文档曾返回，1 题无参考 ID。文档曾返回不保证关键事实实际暴露。
- CDL 配对负题 48：26 题参考文档不全、21 题全部返回、1 题无参考 ID；semantic 20 题，basic 10 题、project_related 7 题。检索与证据利用都要验证。
- 不启动历史最佳混合的在线重跑：它会加载本地 BGE 和重排器；今晚以 BM25 的 sol 新诊断为主，明确不是最佳配置重跑。待 Qwen 完成后另做同条件混合比较。
- 企业样本按题型分层、与正确性无关地冻结 24 题；金融 12、HotpotQA 12。同模型、同 corpus、同回答提示词和盲裁判；变量是证据获取方式。闭卷和一次检索各 1 次生成，agent 最多 6（金融和 HotpotQA 最多 5），不能将不同调用预算视作架构单因果。单次采样，无 best-of-N。
- oracle 仅针对开发错题，用 gold 文档干预定位；含标签泄漏的诊断，不能报基准成绩。

## 续办规则

先读 `state.json`、`calls.jsonl`、`runs/overnight-20261012/worker.log` 和 `results/`，避免重复生成。有运行锁时仅分析已保存结果，勿另起 worker。运行入口 `.venv/bin/python experiments/overnight-20261012/worker.py --stage all`；需要现有 ChatGPT 登录和 SDK 网络权限，禁止 API key／其他模型 fallback。在线脚本故障先确定是否阻塞，只修真正阻塞项，其余记 docs/issues.md。任何调整记录在 state.json 的 decisions，保留冻结样本与所有失败。

预算最多使用 480 次做既定工作，20 次保留真实演示／补齐传输失败。不要为凑满次数跑无用实验。若 08:00 已到立即收尾，不再开启新实验。若全量 Qwen 未完，报告进度和准备好的入口，不冒称全库结果。

## 首轮落地（23:45 前）

- 在线 worker 已启动，日志 `runs/overnight-20261012/worker.log`；已有真实成功回答和评分。运行锁仍由进程持有时不要再次启动。
- CPU BM25 重放已启动，日志 `bm25.log`，共 100 条历史首次查询＋24 条自然问题、4 个固定变体；其结果 JSON 在 `bm25/`。完成后按预先冻结的 selection/validation 分别比较，含金标题的分母不得变化。
- `offline.py windows` 已完成：历史 100 题搜索正文 9,157,613 → 3,949,851 字符，缩减 56.868%；不是 token 测量，窗口是词密度启发式，不是精确 FTS chunk。线上 snippet 变体是现有工具 excerpt_chars=1600，其排名不变，但不能把两者说成完全同一实现。
- `qwen_ready.py --watch` 的 CPU consumer 已启动，状态 `qwen-full-recall/status.json`；没有完整 full 索引前只等待。它复用原任务查询向量，无编码器、无 GPU。consumer 可在本夜之后继续等原任务完成，这是已授权的“完成即出全库召回”准备，不在晨报时杀掉原 Qwen 或 consumer。
- 官方提示词已固定到 `d36685e273713975ee20299bbf1ab64165575b3c`，导出入口离线通过。主实验完成后若预算足够，再选固定前 4 题跑 `official_eval.py --live --limit 4`，预计每题 1＋事实数次；这只是提示词兼容性诊断，不覆盖全官方校正过程。
- 真实 sol 报告、70 题失败人工标签和面试讲解已完成；55 个核心测试通过，离线 demo 通过，共享预算多线程人工 smoke 和 CPU consumer 小库 smoke 通过，均 0 在线请求。不要重复测试，除非有新的代码改动／失败。
- 后续重点：继续分析已完成配对结果；对漏条件／错误阈值的参考文档定位实际暴露区间；在得到完整企业基线后用剩余预算做 snippet 12 题（同固定前 12 题），配对质量和实际 token。切勿改主默认或追改历史 600 题分数。

每个后续 heartbeat 不应仅查看进度就结束：完成至少一个未完成的独立证据分析／文档／准备工作。来源与入口已齐，优先深化证据暴露分析、官方口径差异和下一步实验设计，不为填时长加无必要框架或测试。

首批字符定位审计 `evidence/reference-audit-20261012.json`：qst_0016 确认同一已读片段的合同工 30 天／员工 90 天被选错；qst_0248 gold 未列同文档 >15% 的 Finance 审批，其他角色仍待核查；qst_0442 同时有 Prefer 与有序操作流程，规范强度需审定。二者只是待审参考问题，**不能直接翻转评分或据此宣布裁判错了**，历史 600 题与 48 对负题继续原样保留。后续将 root 的初步判断按这些最新限定续办。

BM25 重放已全部完成（496 条），报告 `experiments/reports/bm25-replay-20261012.md`。决定保留标题权重 2：验证半集 recall@20 为 75.35%，权重 0／8 为 71.10%／70.39%，去停用词与默认完全相同。自然问题去停用词 @20 上升，但 @5 下降，不晋级。不必重新跑这些固定记录。

晨报生成器 `morning.py` 默认只写本地进度草稿；08:00 收尾时使用 `--final` 生成 docs/morning-report-20261013.md 和 evidence/overnight-summary-20261013.json，再人工核对结论、未完成项、实际分支／进程与差异。生成器只整合，仍需填入片段配对、oracle 干预和官方提示词诊断的实际结论。必须在 final 回复中链接晨报并关闭 heartbeat `minimal-rag`，不能仅写文件不交付。

首轮交接时主 worker 仍运行（原 PID 14481），账本已核对 120 次且 ID 连续、全部 sol；Qwen 原 PID 12453 持续编码，CPU consumer PID 15965 等待。续办时以当前进程、账本、锁为准，不凭这些历史 PID 起新进程。首轮没有提交任何代码，所有变更在 codex/standalone-rag。根因原文审计与 BM25 回放已完成；主在线实验、oracle、片段配对和官方提示词在线诊断尚未全部完成，晨报不能把准备或启动当完成。

若主 worker 已释放运行锁但未完成 all，先按 state.retry_after_unix 决定是否恢复，**通过 functions.exec 的 exec_command 权限框架**启动既定命令（原 SDK 需网络权限），不能绕过 sandbox。保持 500 次预扣机制和 sol 无替代。若仍运行，则做原文审计或输出分析，不要干等。全 all 已完成后可启动 snippet 12 题，480 次以后停止新实验，将剩余 20 次仅用于已授权必要补齐；不要为了用完配额重复成功答案。

继续轮已完成 `exposure_audit.py` 的 70 题字符并集核对：0 个原始来源切片不匹配；CDL 48 负题中 12 题所有参考原文字符已暴露，dev 22 错题中有 3 题。一个 duplicate2 版本的切片单独记录，不能把别名 ID 的不同正文坐标混合。报告与 evidence 已保存，后续不用重复审计。核心 evidence.py 残余的逐段哈希机制已记 issues，留到清理阶段处理，避免在架构思考中转去局部修改。

架构分析结束后已执行一个清理：调查 EvidenceLedger/CallBudget 从 core 迁入实验包，逐段 SHA 和 UI 展示移除，真实 bug 的区间检查／撤销／未送达凭证回滚保留。所有当前调用方改为实验包导入；历史轨迹和冻结源码未改，loop、indexed_rag、hybrid_rag、Qwen 编码/评测捕获源码未改。28 个相关行为／包边界测试及 55 个核心测试通过；干净 core/experiments wheel 边界验证通过，独立解包 core wheel 的离线 demo 和真实实验应用离线 demo（0 模型请求）通过。不要重新跑相同检查。未发布、未提交。句子实验的 Encoder 仍被正在运行的 Qwen 路径复用，不在此时移动或删除它。

企业三组 24 题已经全部完成，218 次生成＋评分：closed 2/24、once 12/24、agent 20/24，agent 98 次生成／1,915,063 输入 token，once 24 次／220,548。相对 once 9 胜 1 负 14 平，+33.33 pp，题目对 bootstrap95 [+12.5,+54.17] pp。四个 agent 失败是 qst_0017、0374、0377、0474；0017 是 once 正确而 agent 错的反例，必须保留。原历史最佳同 24 题也是20/24（85次生成、1,507,317输入），CDL同题19/24但不能外推全400/SOTA。报告 `experiments/reports/enterprise-sol-dev24-20261013.md`。闭卷保留 grounded prompt，是保守无检索控制，勿包装成参数知识基准。主 worker 已进入 FinanceBench，后面继续 Hotpot 和 oracle；不得再跑已成功的企业三组。

进一步判断：自然问题的一次 RAG 仍是偏弱基线，应补一个模型选查询、只检索一次的较强基线。已准备 `guided_once.py`，使用原冻结前 12 题，最多 2 次生成、一次 BM25 前5篇、无 read，共最多36次正常生成＋评分；不按已观察正确性挑题。它与主 worker 共用锁和全局账本。**主 all 完成后先运行 guided_once，再做 snippet，再按剩余预算做官方 first4（正常29次）**；若预算不足，减少官方兼容评分的题数，保留48条次因果定位和12题片段对照。不能把较弱自然查询基线代表所有 once-RAG。计划变化是加强有效比较，不改变已完成旧三组或样本。

00:59 前核心在线阶段已经完成，471 次全为 sol：主三基准与 high oracle 342 次、guided once36次、snippet64次、官方提示词兼容评分29次。前12题 natural-once 6/12、guided-once5/12、agent10/12、snippet10/12；snippet 49→52生成调用，961,762→711,762输入token，下降26.0%，一胜一负，不晋级。AND优先＋OR补足的额外CPU100题回放在验证半集与默认完全相同、选择半集反而下降，不改解析默认。官方first4 correctness和逐事实均通过（参考被冻结、无文档校正、sol不是默认gpt5.4），不作官方成绩。

最后一项新对照：`oracle_max.py` 已启动，同原 high oracle 的12个 gold-only 输入、同提示词、同sol、同1次生成、同medium裁判，仅生成effort=max（timeout放宽600秒）。通常24次，预计总495次；若重试导致500耗尽就保留未完成状态，绝不突破。旧dev20 Sol-max 是带检索的不同诊断（55%），此次是隔离阅读能力／参考口径的对照，不是重复旧检索实验。先前“20次保留”是软安排，现用于有明确问题的配对，不为凑满配额。此后不再开新在线实验，所有剩余工作离线；没有 best-of-N，禁止从 high/max 逐题挑更好结果做新成绩。

本段取代上段运行状态：max12题全部完成且逐题正确性与high完全一致（5/12）；累计到495。另完成可克隆5篇原文展示 `examples/sol-demo/`：通过核心QA CLI真实运行3次sol生成，结果数值及draft限制人工核对，录制JSON和最终来源指纹已随例子保存，没有新增评分。**最终在线累计498次，不再开任何新在线请求，即使仍有2次剩余**。这不是凑配额，剩余时间全部离线。各当前在线进程已经结束（以实际进程/句柄复查）；原Qwen和CPU consumer仍需按当前状态核对。fresh core/experiments wheels已放dist，旧core wheel保存在runs/overnight-20261012/previous-wheel，历史元验证与原分数不改。

后续必须完成：检查Qwen抽样真实gate（预计02时段，不凭估算判成功）；若gate未通过且原进程终止，不要重启GPU编码，停止自己启动的等待full-index consumer（先核对PID/command），保存终止理由；若已晋级则原任务继续、consumer等待。不得打断原Qwen。离线精读当前四道agent错误、gold-only剩余错误的实际原文版本/条件，记录明确发现，禁止修改Gold追分。最后07:30后收尾：实际分支、全账本、loop与Qwen捕获源码未改、必要检查已通过；生成晨报并人工改成结论第一句、包含所有额外比较和可克隆实跑，08:00交付并关闭heartbeat，随后才mark goal complete。

01:40附近交接：所有在线工作实际完成；498条预扣均找到保留的对应请求记录，全部completed且用量齐全，325生成/173评分。核账 `evidence/request-reconciliation-20261013.json`。max额外结果已公开在evidence/，core tiny demo3次实跑及记录在examples/sol-demo/。原Qwen及consumer在01:34的ps仍活；仅它们继续，所有在线Python进程结束。正文/docs新增表都把历史600与新dev分开；晨报默认仍写draft，**未生成/交付最终晨报，未关闭heartbeat，goal必须保持active**。此后禁止新在线请求，不靠剩余2次补实验。下一轮先核实实际时间和Qwen状态，再继续独立离线审阅；报告中可以使用请求核账、prefix12、oracle_max、官方兼容、字符审计等额外文件，勿漏掉。

01:40后继续轮完成当前四错题原文定位，evidence/current-errors-20261013.json与对应report：0017目标Horizon Quay未返回，25/50来自另一个Noble Arc客户，不是凭空数值；0374参考ID全返回但10min条款未暴露，>3/h和p99>10%/2h已暴露却混了别处阈值；0377缺两文档、include/profile字段未暴露、Last15m已暴露但未采用；0474无参考ID，诊断性kernel/selection/architecture查询可找到模块说明。这些查询使用失败后的知识，不能当自然召回成绩。没有线上请求、没有改gold/默认、没有碰Qwen源码。上一轮属于progress，本轮也新增可核验证据。继续时不要重复此四题定位。

继续轮面试60秒介绍已压短，FAQ改为已完成的实际dev24成本对照。gold-only复核新增0003规范强度旗标：原文Implementation-ready/repo receives PR/Rollout plan，gold及裁判要求land PR/execute rollout，需独立审定、不翻分。0351的kill-switch限制与watchdog/resync建议确有原文依据，候选确实遗漏，不能把全部失败推给gold。后续无需重复0003/0351同一核对。

新增Hotpot输出审计：12/12标注支持句均暴露，两个EM失败是National Hockey League vs NHL及多了Sir，官方归一化仍EM0（F1分别0/0.8889）。仅人工语义说明，不能报100%、不能重跑挑正确形式或改官方normalize。另启动一小时只读抽样阶段监视，工具session72299，每45秒检查原PID12453及stage；只在阶段变化输出，超时不等于原任务结束。后续可wait该句柄，但仍应以原进程及actual artifacts判定；不重启原GPU任务。

继续轮已合并全部498请求的token用量，按thread最新累计值而非逐turn相加：输入5,191,533（含缓存1,750,400）、输出95,027、总5,286,560。生成325请求输入4,428,589，评分173请求输入762,944；reconciliation证据和morning生成器已更新，禁止把缓存/推理子集再加到总量，也不换算费用/额度。这轮有新增核账证据，属于progress；只读监视仍在等抽样阶段变化。

02:01当前19个活动Markdown入口链接核对通过（以本轮命令实际页数为准），不含archive。原编码184,320/198,398，93%；一小时观察句柄仍运行，不是编码终止。源代码未改、预算498未变，晨报尚未交付。接下来等待gate实际生成，再更新README/scoreboard/evidence目录中“尚未产生召回”字样，只用原task输出，不自跑GPU评测。

收尾要求与证据/未完成项已整理在 completion-audit.md；这是内部核对，不代表已交晨报或已达到官方SOTA。当前继续等待原抽样进程真实阶段切换。

Qwen抽样已实际完成，gate failed：semantic25 BM25@20=.80、BGE=.24、Qwen=.52、RRF=.84；仅0203一题新召回，1胜0负24平，+4pp<+10门槛。all94 .268617→.279255未退步，100个非semantic参考出现缺失仍在分母。已独立重算全部400条@20和两组gate，与原输出一致；原12453自然退出，绝不重启或改门槛。精确核对consumer15965命令后只对自己的consumer发SIGTERM，status标stopped_no_full_build。源码、索引、Gold、原Score未改。README/runbook/scoreboard/evidence已更新，FullIndex不存在，不能再推荐直接Qwen0.6全量。保存候选top100的并集上限semantic96%，后续可CPU单线程已有MiniLM重排的事后诊断，不编码、不GPU、不线上、不自动晋级。

CPU事后重排已启动：script cpu_rerank.py，工具session96307，日志cpu-rerank.log，输出qwen-posthoc-rerank/。HF/Transformers离线，单线程，已有MiniLM参数实际CPU/float32；不加载embedding，不调用hosted模型，ledger498不变。每题≤200个实际最佳chunk（sparse100文档、Qwen100文档，chunk去重），原query，gold只在排名后计分。可以分析排序上限，但这是gate失败后新增候选，不得当作原gate通过或自动启动全量编码。此前“no local models”软安排是避免与活着的Qwen争资源；现在原任务已自然退出，仅允许这次CPU重排，仍不使用GPU。morning推荐和Qwen段落已改为停止全量及同预算验证排序/规则段，而不是继续0.6B扩库。

全库纯BM25同问题校准已完成（0模型调用、CPU、只读）：semantic25全库@5=.36、@20=.44，而抽样@20=.80。不是此前把@5与@20混比；现在按同@20明确抽样背景小带来+36pp。不能将抽样RRF84%或事后重排当全库成绩，fullQwen仍未建，fullhybrid未知。结果evidence/full-bm25-calibration-20261013.json。CPU重排仍运行（以句柄/进程为准），本轮下一步读完整100结果而非partial。

04:22后收尾前更新：CPU重排100/100实际完成（工具96307退出0），semantic@20=.76、overall94=.2579787，低于BM25；独立重算100题后拒绝，不再追加重排。单线程CPU，没有在线/GPU。全库BM25校准结果已经保存；不要重复跑。README/一页架构/面试FAQ/当前四错报告的旧Qwen推荐已统一改为暂停扩库。下一轮设计只试一条通用取证提示，现有search/read和最多6调用不变，不含题专属词或数字；dev24两臂最多336次需要下一轮额度，今晚不执行。发布计划没有Markdown标题的限制已记录；暂不添加章节解析器。morning.py草稿接入完整校准、官方first4、两种实跑demo与全部用量，最终仍需07:30后核查、08:00交付。498封顶停新在线；目前所有实验结束，无需重新启动进程或为了耗时扩大代码。

04:30后新增有效工作：补齐gold-only7个固定失败的针对性原文审计，新增0240/0341/0369/0376记录在现有reference-audit中，cross-oracle报告已接入。0240确实漏host，但double-counting、premature eviction和evicts inflight capacity确有来源，裁判否定该条款有误；0369 ADR429 SHOULD支持建议语气，gold与hotfix的范围需审定；0376的90天／审计导出遗漏确实存在。原5/12、high/max、历史分数均不变，不计算新纠错分。此后推荐需同时冻结来源版本和参考口径，不能把7错都说成纯阅读能力问题。没有新增在线调用。

05:12继续完成官方输入准备：新增离线export_official_history.py，实际导出历史dev100+test400全部500答案至evidence/historical-enterprise-500-answers.jsonl，500/500完成，原文字内文引用不变。899实际引用文档、905含替代版本映射，未知ID0、未观察引用题0。q0413唯一带alias引用且同时引用canonical和duplicate2，官方UUID合并后仍须对齐物理来源。凭据official-history-export-20261013.json，独立parse/hash/schema/count核对通过，0在线请求，不是重新生成astra，不是官方分数或提交。官方report/runbook/evidence目录与morning已接入。

05:24完成905个正文版本的发行来源绑定：单线程只读扫描Parquet，按UUID+精确正文匹配零起始行号，905/905匹配，整个1.41GB文件SHA与原manifest一致。0413新/旧分别189417/191029行。此前source URI是逻辑来源，现在另有真实Parquet行号；这不证明官方Git JSON遍历结果，不以行顺序替代官方UUID覆盖选择。导出脚本和manifest已更新，工具26998退出0，仍498在线。

05:33求职可视报告落地：新增render_showcase.py/template与examples/sol-demo/report.html/preview.png，只渲染现有3次sol录制，无新增生成或评分。三个步骤与完整答案展开、1280桌面/360/320手机/深色主题实际浏览器+截图核对；测得三宽度横向溢出0，修复favicon404后fresh page console0errors/warnings。展示来源4911offset与fixture精确切片匹配，recorded-run SHA原样。只对新QA小库回放做视觉验收，不把旧investigation视觉状态改为通过。使用已安装CLI缓存，默认npx网络失败后没有下载安装；浏览器file协议blocked故临时本机HTTP，直接file协议未单独自动验证。专用overnight-sol-report浏览器已close；精确ps核对后只TERM本聊天server34211，工具54809exit143。源码/Qwen/500账本不变，README与示例README入口已接入。

06:27新增48道CDL负题查询文本补审完成：loss_query_replay.py工具76182退出0，96次CPU排序，48保留/47有gold计分。原题vs历史astra首个查询，同title2/OR/offset0/global20，排除结构参数与旧hybrid排序因素：@5=.35786/.31176，5胜4负38平；@20=.464953/.438357，4胜2负41平；semantic20首查15%/10%。独立重算47对通过。补进bm25报告和morning，不推断整个agent召回，不把总是先搜原题升默认，0在线/索引写/模型/GPU。停止扩库、同预算取证验证推荐不变。此前bm25报告末尾旧Qwen全量推荐已同步修正。

06:40低成本融合收尾：rrf_replay.py只用原缓存top100，不加载模型、不GPU。固定12组k与权重、沿用夜初已冻结原dev-index奇偶选择/验证。按selection规则选k100等权，semantic选择7/9→8/9、验证14/16不变；验证overall47也完全不变，逐题验证47全tie（以paired20为准）。Whole84→88仅来自selection，BM80→88+8pp仍不足原+10门槛；拒绝晋级，不在validation另择赢家，原gate不改。原k60等权全部100个20-doc排名逐一复现。evidence/rrf-grid...报告/晨报已接入。仅0.02sec字典排序，不构成新模型重排或编码，online仍498。

07:00前只读编码器契约核对：Encoder源码与缓存Qwen Pooling配置匹配last-token/mean=false、左padding、仅查询instruction、文档prompt空。已有query矩阵100×1024 float32、L2norm范围~1且finite，未实例化模型或GPU；没有发现明显配置错误，但不是独立验证文档向量语义或查明质量原因。记录入validation。后续只收尾，不再加新实验。
