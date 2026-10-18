# 待办与评测边界

2026-10-15 清理。只列未解决的问题和仍然有效的口径限制，一行一条；已完成实验的结果见[实验索引](../experiments/README.md)。

## 待决定

- Grep长行可阻断后续关键命中：BrowseComp题666首次Grep真实命中5行，按路径排列的第3行是1,302,405字符的单行文档，80,000字符预算触发break后仅返回前2行/1,373字符，排在第5行的关键参考文档未传给模型；相同查询离线改为files_with_matches即返回全部5文件。需让超长行明确截断/返回匹配窗口并继续处理后续命中，本轮仅审计不改冻结实现或分数。见[脱敏审计](../experiments/reports/grep-browsecomp-paired-audit-20261017.md)；逐条原文留存于不入库的private-audit目录。

- 文件工具dev100的Grep收益仍有限：74→78/100但95%区间跨0，semantic16→15、project_related0→0；106次有效Grep中59次截断，无路径/glob/type过滤时中位返回41篇、34,108字符，路径约占返回字符30%。结果按路径而非相关性排序；这些是宽匹配/输出开销迹象，不把非参考文档直接判为噪声。见`experiments/code-tools-20261017/noise-audit.json`，暂不修改冻结工具或追加评测。

- Enterprise原文有37篇包含NUL字符：SQLite `length(text)`在NUL处停止，而Python原文长度更长，现有search的document_chars可能不准确；文件工具导出逐篇保留原文并用rg --text覆盖这些文档，本轮不修改已冻结BM25基线。

- **纯BM25仍未追平历史Enterprise成绩。** 新search说明309/400＝77.25%，原说明298/400＝74.50%，历史v0为324/400＝81.00%；新说明search减少26.12%、模型调用减少13.48%，但准确率差95%区间[−0.75,+6.50]pp，稳定提分仍待验证。只改说明，未加入固定流程。见[完整对比](../experiments/reports/search-description-20261016.md)。
- **原说明回退主要来自 semantic 检索。** 按轨迹配对：43 负中 33 题 v0 已出现全部参考文档、BM25 完全或部分没找到；按题型 semantic 78→58（−20），basic −8，其余题型持平或略升。全部参考文档出现率 92.0%→80.6%；search 1653→3342 次、read 1016→452 次，说明模型在反复换词搜索而不是读。 新说明semantic已从58升至68/100，仍低于历史78/100。

- **Qwen 8B／4B 尚缺全库与同一 agent 验证。** 含参考文档的小库整体 Recall@5 61.05%→86.31%、semantic 32%→92%；固定 dev24 回答仅 12/24→14/24，project_related 找齐率@20 5/8→3/8，仅保留为历史诊断，不进入当前纯 BM25 主实现。见[本轮报告](../experiments/reports/qwen8b4b-bm25-20261016.md)。
- **历史 v3/A 的 dev 错题分层（2026-10-16 按轨迹统计）。** Enterprise v3 dev 18 错：13 题全部参考文档已在 search／read 中出现（其中 8 题为 project_related 多条款漏写或写错），参考文档完全没找到的 2 题均为 semantic，部分找到 2 题，无参考 1 题。BrowseComp A dev 7 错：答案文档已出现 4 题，完全没找到 2 题、部分找到 1 题。检索改进的上限约 7/200 题，回答层约 17/200 题。文档级出现不等于逐条事实已读到。
- `models/hybrid-rag/models-qwen3.json` 实际只替换 reranker，embedding 仍是 BGE-small，不能称作 Qwen embedding 配置；历史 0.6B 结果也不能替代本轮或全库结论，见[编码器对照](../experiments/reports/offline-encoder-granularity-20261012.md)及[扩库实验](../experiments/reports/qwen-scale-gate-20261013.md)。
- **随模型升级受益仍缺受控验证。** 历史 Enterprise test 对比同时更换模型与检索栈，同 cascade 的 dev 对比也未显示稳定提升；尚不能证明固定工具、提示、预算和裁判后，仅升级模型即可稳定提分。见[模型切换评测](../experiments/reports/sota-20261014.md)。
- **BrowseComp 提交文件已备好、未上传**，待用户决定是否提交维护方核实：[submission.json](../experiments/browsecomp-final-20261015/submission.json)。
- **云实例停止计费须核对平台状态。** 2026-10-16 AutoDL 下载脚本完成并发出容器内关机命令后，控制台仍显示无卡运行；已从控制台关机。后续不能将进程退出或容器 `shutdown` 当作实例已停，须用平台定时关机并核实。

## 回答层

- **纯BM25回退主因已定位到取证入口。** 43道回退题中32道涉及检索缺口；仅1道确认已定位文档但关键审批段没读到，7道是已见证据的遗漏／错选／拒答，3道判分敏感。优先验证查询限定词保留与候选探索；本轮只诊断、不改实现和原分数。见[60题逐题核查](../experiments/reports/enterprise-bm25-regression-audit-20261016.md)。

- Qwen dev24一次回答实验固定Top5/4000字符且禁用工具，关闭了现有分页、文档内搜索和继续读取；4道原判错题的必要参考文档已在Qwen Top20，需用同预算真实取证闭环验证，不能直接把缺口算作生成能力上限。见[逐题审计](../experiments/reports/qwen8b4b-dev24-audit-20261016.md)。
- 本地裁判未看到实际提供的原文：`qst_0422`把已有原文支持的RBAC补充判成无依据，`qst_0446`两份几乎同义回答、相同事实向量却一对一错；旧分保留，下一轮需统一规则后对各组全量同口径评审，禁止挑题翻分。
- 原文标题有污染且未统一计入预算：本轮BM25侧5个标题超过1000字符，最大7681字符，部分实际为完整Slack正文；下一轮需要统一显示标题与真实证据预算，不能暗中改写已冻结排名。
- v0 test400 的 76 道错题中，56 道部分事实被支持；其中 30 道全部参考文档 ID 曾返回，22 道全部参考正文已暴露。主要缺口是已见证据没写对、没写全，见[失败层审计](../experiments/reports/failure-layer-audit-20261015.md)。
- 上述 30 题来自 test400，不能用来开发候选；回答层候选只在 dev 错题上开发，test 只留给最终一次评测。
- 固定证据对照（dev24）：正确数 21/24 对 21/24，事实完整度 91.90%→94.51%，候选未采用，见[报告](../experiments/reports/fixed-evidence-v3-20261015.md)。

## 口径限制

- 本地 sol 裁判比 Enterprise 官方裁判严：同 400 题上，CDL 公开答案本地判对 86.5%，官方判对 92.0%，两边判定一致 372/400（官方逐题判分取自[榜单原始数据](https://huggingface.co/spaces/onyx-dot-app/EnterpriseRAG-Bench-Leaderboard/tree/main/data/raw_data)，2026-10-15）。本地分数不能和榜单直接比较。
- 失败、超时保留在分母，首次有效判决不重跑。
- BrowseComp A/B 共用核心循环，+10.00pp 只能归因于工具、返回文本和提示的组合。
- 用量记录有缺失（BrowseComp A-dev100 1 次；Enterprise v0 test 的 qst_0050、qst_0437）：相关成本比只能给上界，不是精确节省率。
- 引用召回、观察召回、单次检索 recall@k 是三种指标；抽样库、gold 隔离、短请求 token 探测都不能外推到全库或总体成本。
- 参考答案或裁判边界有争议、分数未改：qst_0003、0240、0248、0351、0369、0373、0374、0413、0422、0442、0461。见[来源审计](../evidence/reference-audit-20261012.json)。
- qst_0413 同一 ID 对应两个正文版本；导出官方格式前要对齐实际文件。
- [sentence_retrieval.py](../experiments/src/minimal_rag_experiments/sentence_retrieval.py) 的来源快照仍引用早已不存在的 `docs/experiment-protocol.md`，该实验入口重跑会失败。
- 官方 BrowseComp 脚本 0469490 的 `calib_err` 漏算最后一箱；其 `uv.lock` 锁定的 pypdfium2 4.30.1 已被 PyPI 撤回。均按官方原样使用。
- 独立向量工具实验的 BrowseComp A（2026-10-16）题969、979、985出现SDK `no active turn to interrupt` 运行错误；本轮冻结适配器和超时设置，不补生成，原始轨迹及错误清单保存在runs/semantic-tool-20261016，后续单独检查取消请求时的错误处理。
- semantic-tool-20261016 transport_control.py matches any `429` substring: the user-requested BrowseComp SIGINT at UTC2026-10-17 07:32:47 was misclassified from digits in a shell-snapshot filename inside TransportClosedError. This caused one30-second cooldown; retain the raw event but exclude it from actual HTTP429 counts. Tighten classification in a later transport change; it does not block the active Enterprise run.
