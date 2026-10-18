# 运行指南

Python 3.12＋。核心命令只有 `demo`、`index`、`search`、`ask`。`ask` 和所有 benchmark 的当前入口共用纯 BM25 的 `IndexedKnowledgeBase`、`search/read` 工具、模型适配器和核心循环。

## 固定API小样本验证：总预算200元（2026-10-17）

用户确定路线：Codex全量结果用于正式提交；API仅做固定开发样本的迁移验证。用户随后明确授权执行v2，并确认创建专用Key；**小样本验证及技术故障补跑已全部完成，所有运行进程已结束，未发送提交邮件**。[固定协议](../experiments/api-validation-20261017/protocol.json)、[样本与顺序](../experiments/api-validation-20261017/sample.json)、[离线核对](../experiments/api-validation-20261017/preflight.json)、[执行结果](../experiments/reports/api-validation-20261017.md)。协议里的未开始状态保留为设计时快照，不是实时状态。

完成结果：API迁移24/30，对照Codex22/30（3胜1负）；固定证据low23/30、high22/30（high 1胜2负）。high的宏平均事实完整度略增1.69pp、答案字符中位数371.5→701，但未通过预设收益门槛，不自动晋级或追加全量实验。共92份完整答案和92份有效判分；实际API费用$7.5109163，与Key用量完全一致，按保守预算口径约63.39元。两项未发出模型请求的额度查询故障按用户追加授权补跑且均判对，不计原技术错误；一次取消请求收费及原样恢复费用均计入总额。完整边界和核查见[报告](../experiments/reports/api-validation-20261017.md)、[最终核验](../experiments/api-validation-20261017/validation.json)。

当前设计为v2：用户在开跑前要求阶段2改为low/high，两组统一32,768输出token上限，并补充加密推理回传验证；题号、证据、对照顺序位置和原基线判分不变。旧v1设计保存在 `experiments/api-validation-20261017/archive/v1/`，不得与v2混用。

主样本从Enterprise dev100按seed=20261017选取，只用题号和题型：basic8、semantic6，其余8类各2题，共30题。另固定2道不重叠冒烟题。选择过程不读答案、判分或成本；这些仍是已接触的开发题，稀有类别被过采样，样本准确率不代表完整题集准确率。选定后核对原Codex A为22/30；不得因为这个分数或后续API结果重新抽题。

| 阶段 | 固定任务 | 阶段费用停止线（USD） | 进入下一阶段的条件 |
| --- | --- | --- | --- |
| 0 | 两道独立冒烟，核对原生工具、加密推理回传、参数、实际模型/供应商和费用 | $1 | 两题正常完成，观测到工具反馈和加密推理条目原样进入下一请求且续轮被接受；参数/费用不明就停 |
| 1 | 同30题，API模型gpt-6.1-sol、high推理、low verbosity、完整BM25语料 | $6 | 30题全部有记录及判分，至少29题正常完成、至少19题正确，无重复的协议/映射/供应商故障 |
| 2 | 同30题，用旧Codex实际见过的固定证据，low/high各一次，共60次；两组输出上限均32,768 | $7 | 只报告详细程度的方向性影响，不自动采用配置或启动全量 |

“至少19/30”是相对原22/30最多净少3题的工程筛查门槛，不是统计非劣或无退化证明。失败计错并保留分母；完成前费用停止就报告未完成，不能换题。阶段1不固定12次调用、不强制收尾，因此同时改变了运行时、原生工具协议、提示包装与停止策略；只能比较两套完整配置，不能单独归因于工具调用。API明确传low，不沿用工作区其他会话可能改动的默认verbosity。

阶段2在抽样后离线提取旧A轨迹中实际返回的正文，验证每个字符区间，合并同一原文的重叠/相邻片段，保留首次见到文档的顺序。没有补充未见证据，没有旧答案、旧推理或gold。两组输入和全部设置相同，仅verbosity不同；15题low先、15题high先，顺序位置不变。不增设medium第三组，不把答案增长当收益；只有high净赢至少1题且事实完整度不降，或正确数不降且宏平均事实支持率增加至少3个百分点，才记为值得继续研究的方向性信号。它不是完整agent成绩。

输出上限涵盖推理与可见答案，32k也不保证不会触顶。阶段2分别报告两组的reasoning/output用量、答案长度、`incomplete_details`、触顶与超时次数；未完成仍计入原分母，但不能把触顶造成的失败直接解释成high损害答题能力，也不只给high补跑或临时增大上限。high可能更贵，$7阶段停止线不保证60次必能完成。

冒烟的推理回传检查独立于缓存检查：请求 `store=false`、`include=[reasoning.encrypted_content]` 和完整推理上下文；至少一次返回非空加密推理条目，下一次实际发送的请求逐字段原样保留该条目与原生调用/结果，随后成功续轮，模型/供应商不变且没有字段拒绝/丢弃警告。只记录数量、是否非空、原样回传比较结果、状态及usage等脱敏证据，不公开加密内容。缺失或拒绝则记兼容性未验证并停止，不删掉推理字段强行通过。HTTP成功只能证明服务接受了请求，不能单独证明上游确实使用了全部推理内容。

缓存命中率另行记录：命中不证明加密推理被使用，未命中也不证明推理断裂。零命中时复核费用是否仍符合预算，不自动判协议失败，也不为追求命中追加付费探测。显式请求 `reasoning.context=all_turns` 的支持情况属于冒烟待确认项，不支持时不静默降级。

全部新判分使用Codex订阅、原sol medium裁判和参考事实。原基线首次有效判分保持不变；新裁判不看组别、另一组答案或参数标签。只为传输/格式错误最多尝试3次，不重判有效的错误结论。额度不足暂停，不切API裁判、不购买重置。报告配对胜负、事实覆盖、工具/协议错误、全部费用与失败；配对bootstrap只描述本样本，不作迁移等价保证。

费用方案：使用OpenRouter标准档，不把Flex/Batch折扣计入预算。专项Key计划设 **$20 TOTAL、不按月重置**，本地累计费用达到 **$14** 后不发新请求，预留 **$6** 给单个在途请求。生成并发1，逐次核对OpenRouter实际费用，费用缺失/未结算则先停，不盲重试。阶段0/1单次max_output_tokens=16384，阶段2两组均32768；单调用300秒、每题3600秒。触发上限记录未完成，不强制补答案。按922,000最大输入、长上下文写入$5/M及输出$15/M计算，阶段2最坏约$5.10152，故原$5预留上调到$6。执行前重新核对费率与上下文上限，锁定OpenAI标准供应商、关闭回退并要求参数受支持；预留不足则不启动。

按保守预算换算8元/美元、另计5.5%平台充值费，$20对应168.8元，余31.2元保留作汇兑/费用缓冲；这不是实时汇率报价。阶段费用线只能阻止下一请求，不能撤回在途调用，仍受全局停止线与专项Key限制。账户现有其他余额及Key不属于本实验额度，不追加充值、不自动提高上限。所选30题按旧token重算约$1.95–2.35（含5.5%），只是成本参考，不承诺开放轮数后的实际用量。

**执行落实：** 专项Key已设$20 TOTAL、不重置、7天过期，密钥位于仓库外0600权限文件。专项执行器 `experiments/api-validation-20261017/run.py` 在每次付费调用前核对Key总额、阶段停止线、$14累计停止线和$6预留，任何未结算费用阻止下一请求；进程锁防止并发重复运行。API账单、实际供应商/模型、verbosity、reasoning参数和推理回传均检查并保留证据。UI没有启用BYOK计入限额，因此本执行器拒绝BYOK路由及非零BYOK支出，仅允许OpenRouter余额付款。按公开完整上下文窗口重算的更保守单请求上界$5.74152仍小于$6预留。[预算核验](../experiments/api-validation-20261017/budget-preflight.json)。通用benchmark CLI不具备同等美元停止线，不可代替此固定执行器。

离线准备可重复运行（已有文件不覆盖；开始请求后拒绝重建输入）：

```bash
.venv/bin/python experiments/api-validation-20261017/prepare.py
```

原始问题、固定证据和独立裁判资料保存于git忽略的 `runs/api-validation-20261017/`，仓库只记录题号、协议和统计。Enterprise500原答案已完成零生成导出，见[导出凭据](../evidence/frozen-enterprise-export-20261017.json)；官方判分和维护方接受仍待完成。小样本API结果只作为迁移佐证，不替换Codex全量分数。

官方预算依据：[OpenRouter Key总额限制](https://openrouter.ai/docs/guides/best-practices/capping-shared-agent-keys)、[供应商过滤](https://openrouter.ai/docs/guides/routing/provider-selection)。

## 直接API与开放调用预算（2026-10-17）

按用户新授权，当前入口支持 `--backend api` 和 `--backend codex`；两者共用检索和循环。原冻结包及下方77.25%/93.01%记录不变，新后端与新预算尚无真实评测成绩。API路径使用HTTP直连Responses API，不启动Codex、不需要ChatGPT订阅；端点需支持Responses原生函数调用、严格JSON Schema答案输出，以及选用推理模型时的reasoning字段。仅支持Chat Completions的端点不兼容，不会自动转换或切换付费后端。

```bash
.venv/bin/python -m pip install -e '.[api,benchmarks]' -e ./experiments
# 在本机环境设置 OPENAI_API_KEY；RAG_API_MODEL 设置为该端点实际可用的模型ID。
# 可选 OPENAI_BASE_URL（默认 https://api.openai.com/v1），不要把密钥写进URL。
.venv/bin/python -m minimal_rag ask --live --backend api --model "$RAG_API_MODEL" \
  --index data/my-corpus.sqlite --question "How should I investigate a connection timeout?"

# 使用已准备好的题集和索引，只生成，之后单独做官方判分。
.venv/bin/python -m minimal_rag_experiments.benchmark --live --backend api \
  --model "$RAG_API_MODEL" --dataset enterprise --split test --skip-judge \
  --output runs/api-open-enterprise-test
# BrowseComp完整830题：改为 --dataset browsecomp --split all，并使用另一个输出目录。
```

API模型必须显式指定，不假设订阅侧 `gpt-6.1-sol` 能在任意API账号访问。默认effort为high，不支持推理参数的模型用 `--effort none`；Enterprise/Finance本地判分另需支持medium推理设置，官方评测准备推荐 `--skip-judge`。可通过 `--base-url` 显式指定端点。每题独立上下文，后续调用保留原API输出及推理条目，`store=false`，不做静默历史截断。没有人为设置max_output_tokens，仍受模型自身上下文/输出限制。

按用户进一步要求，API提示和工具协议已清理：仅把任务要求放入 `instructions`，问题直接作为user消息；search/read的说明及参数放在原生 `tools`，返回的 `function_call` 由同一外层循环执行，结果通过对应 `call_id` 的 `function_call_output` 回传。不再发送Codex PROTOCOL、外层messages/tools包装、arguments_json输出要求、rag_observations标签或“不要嵌套协议JSON”等提示。内部Reply只是Python适配边界，不出现在模型的工具协议中。工具定义显式 `strict=false`，保留现有可选参数与默认值，本地Pydantic检查失败会作为工具结果反馈。没有供应商内置搜索或子智能体。

最终答案只保留严格结构 `answer` 与 `sources`，用于回答展示、引用核查和评测，不再要求模型输出tool_calls字段。Enterprise/Finance使用简短的语料问答任务说明；Hotpot保留最短答案要求；BrowseComp保留基准的Explanation/Exact Answer/Confidence要求。无工具的本地裁判直接返回其要求的判决JSON，无额外外层协议。原Codex路径保留其协议，API原生调用构成新的实验配置，不能称与旧传输等价。

API生成默认显式发送 `text.verbosity=low`（v2实测后按用户决定固定），通过 `--verbosity low|medium|high` 调整；端点还需支持此参数，不支持时不静默降级。该值写入运行配置和每次API调用记录。本地裁判固定medium，修改生成verbosity不同时改裁判。Codex入口不接受此覆盖，以免误报实际发送值。冻结版的请求探测显示Codex发送low，并注入额外工具与多智能体说明，尽管本地配置尝试关闭相关功能；没有内置工具调用不等于模型没有看到工具定义。[脱敏探测记录](../evidence/codex-hidden-context-20261017.json)。API版不复制这些注入，也不恢复旧的末轮强制作答。

要判断verbosity的作用，应在同一API模型、相同工具/提示/预算/裁判下只比较low与high；直接比较旧Codex与新API同时改变多个因素，不能把差值全部归因于verbosity或注入。本轮high未通过收益门槛，用户已将生成默认值固定为low；high仍可显式用于实验，但不能作为已验证的提分措施。更长的答案不保证事实更完整或更正确。

当前CLI预算（Codex/API一致）：

| 参数 | 默认 | 行为 |
| --- | --- | --- |
| `--max-turns` | 0 | 不设模型调用次数上限；正数到期直接标budget_exhausted，不强制作答 |
| `--timeout` | 300秒 | 单次请求超时 |
| `--case-timeout` | 3600秒 | 每题总超时；0关闭 |
| `--max-total-tokens` | 0 | 可选每题已记录输入＋输出总量（含缓存）；0关闭，在下一次调用前检查 |

token上限不是精确金额上限，一次请求可能越过阈值，失败请求还可能缺少usage；实际费用按供应商计费。若要同时取消每题时间上限，传 `--case-timeout 0`。上下文超限、拒答、未完成响应与技术失败保留为失败记录，不挑错题重抽。历史12/30轮配置仅用于原冻结成绩；当前即使显式设回12轮，也不再有旧的末轮强制作答行为，严格复现旧成绩须使用冻结源码。

离线验证覆盖14次调用自主收尾、真实本地search反馈、API推理上下文保留、原生call_id对应、多次同名调用及参数错误反馈、每次请求token计数、未完成响应、token停止，以及阻止导入Codex包后的benchmark生成／本地判分／resume。测试使用模拟HTTP响应，没有新增真实模型调用；API效果、价格与官方接受情况仍待实际验证。

API实现参考：[原生工具调用](https://developers.openai.com/api/docs/guides/function-calling)、[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)、[推理上下文](https://developers.openai.com/api/docs/guides/reasoning#keeping-reasoning-items-in-context)。

## 冻结与官方提交（2026-10-17）

**最新判分状态：** 用户追加授权后，冻结版BrowseComp现成830题已完成Qwen3-32B BF16判分，781/830＝94.10%（历史772/830保留），18胜9负803平；未重新生成答案，无有效判决重抽。官方评分逻辑保持不变，为单A800显存仅把引擎max_model_len设为8192；提交时须披露该调整及TP=1。见[结果](../experiments/browsecomp-frozen-judge-20261017/result.json)、[运行记录](../experiments/browsecomp-frozen-judge-20261017/notes.md)。以下“尚未判分”和“待补830题裁判”是此前路线，已由此次完成状态取代。Enterprise官方评测仍待完成；尚未发送提交邮件或获得维护方认可。

用户决定结束调优，冻结当前无 Grep 主路径。源码包：[source.tar.gz](../evidence/frozen-bm25-20261017/source.tar.gz)；版本、参数、SHA256 和文件清单：[manifest.json](../evidence/frozen-bm25-20261017/manifest.json)。这是工作区源码快照，包含尚未提交的改动，不能仅用清单中的 Git HEAD 复现；源码包不含语料、索引、权重或运行结果。实验源码一并存档不代表启用这些候选。此后只补复现与提交材料，停止优化实验。

冻结模型为 gpt-6.1-sol high，工具为 search/read。Enterprise 摘录4000字符、12轮、单调用180秒、每题900秒；BrowseComp 摘录2000字符、30轮、单调用180秒、每题1800秒。当前 indexed_rag.py 指纹与309/400那轮记录一致；这个核查不等于对全部环境重新验证。当前 Enterprise 77.25% 是本地 test400 结果，不是官方分数；历史81%和 BrowseComp 93.01%不移植到此版本。

### 完成标准与顺序

完成标准：维护方明确复核认可或官方榜单出现可公开核验的系统条目，条目能关联真实模型、版本与复现材料。自行运行官方脚本、发出邮件、维护方收到邮件均不等同于完成。

**2026-10-17成本核对后的路线修正：优先复用当前冻结配置的现成答案，不必先切回历史93.01%配置。** `runs/semantic-tool-20261016/browsecomp-A/` 已有完整830题轨迹和官方格式答案，尚未判分；检索器、适配器、循环及评分辅助源码与当前一致。归档中的 benchmark.py 指纹与该轮配置一致，和当前入口的差异是删除未启用的semantic分支；830题实际工具仅search/read。该轮8道技术失败曾按原授权重试，提交时保留全部尝试并披露，不称严格零重试。Enterprise现有test400与code-tools A/dev100的settings、回答格式相同，题号并集500、交集0；dev A只包装工具计时，不启用文件工具。正式提交仍需完成来源与导出核验及维护方接受条件确认。

按已有运行用量，全部重新生成（Enterprise500、BrowseComp830、Finance100、Hotpot100，共1530题）的中心估计为9042次生成调用、339,038,278输入token（其中未缓存67,439,558）、2,016,765记录输出token，不含裁判与失败重试。前两项取当前配置实测，后两项取旧搜索说明的纯BM25实测作估算。历史每题耗时总和折算并发5约4.86小时，是理想吞吐估计，不含排队、限流、评测及部署。订阅额度不能由这些token精确换算为额度百分比或现金账单。

当前推荐：两项主基准新增生成0，BrowseComp补830题官方裁判，Enterprise补官方流程；Finance/Hotpot本阶段不重跑。如需全部成绩对应新说明，辅助两项合计约572次生成调用、100次原本地Finance裁判，Hotpot使用本地公式评分。下面历史BrowseComp提交方案保留为备选，不再是优先路线。

1. **优先整理历史 BrowseComp 93.01% 的独立提交包。** 已有830题官方格式结果及Qwen3-32B BF16本地判分；本次离线核对 submission.json 有830个唯一题号、772题正确。官方README要求将评分摘要JSON发送给 s42chen@uwaterloo.ca，并填写检索方案和项目链接。现有提交JSON的 Link 为空，当前仓库没有配置 Git remote；先准备可访问的项目/复现页面、准确的历史运行说明和模型访问说明。历史 scaffold 中的选型叙述应改清：93.01%对应原A/v0，后续确有用户采用v3的决定，不能暗示A一直是主版本。披露开发集接触及多轮实验，保留4道原生成失败。历史源码复现完整性仍需核验，不能用本次冻结源码替代。维护方是否接受该模型访问方式与历史运行由其确认。
2. **Enterprise 保持为当前冻结版本的主目标。** 官方主集500题，提交格式为 question_id、answer、document_ids；现有309/400及此前历史500题导出不能直接冒充本冻结版本的官方成绩。先离线核对现有当前版本dev100与test400的配置及题号覆盖，仅在证明配置一致后合并复用；缺失部分再明确必要生成量。对齐引用UUID与物理文档版本，补全官方评测流程或按维护方要求验证，不继续使用本地代理裁判来宣称官方分数。
3. **优先确认验证方式，避免无效花费。** Enterprise官方要求联系 joachim@onyx.app，开源提交提供复现指南，闭源提交提供可验证的sandbox/endpoint。与维护方确认生成模型可访问性、指定裁判/环境、既有答案复用和已接触题集的披露要求，再执行必要的正式评测。此阶段不要求先冲到某个分数。

本次只冻结、核对公开规则和整理路线；没有新增模型生成/判分、租用GPU、公开上传或发送邮件。对外邮件内容及发布材料准备完整后再交用户确认发送。

2026-10-17核对的官方来源：[Enterprise提交要求](https://github.com/onyx-dot-app/EnterpriseRAG-Bench#leaderboard)、[Enterprise答案格式与评分](https://github.com/onyx-dot-app/EnterpriseRAG-Bench/blob/main/answer_evaluation/README.md)、[BrowseComp提交要求](https://github.com/texttron/BrowseComp-Plus#submitting-to-the-leaderboard)。入口可能更新，实际提交前再核对。

## 安装与离线演示

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m minimal_rag demo
```

`demo` 从 `fixtures/retail.jsonl` 建 SQLite 索引，回放预先写好的模型决策，生成调用为 0，无需账号。

## 自己的资料与真实问答

JSONL 每行包含 `id`、`title`、`text`、`source`。SQLite FTS5 索引保留原文和字符位置；构建器拒绝覆盖已有文件。

```bash
.venv/bin/python -m minimal_rag index examples/documents.jsonl --output data/example.sqlite
.venv/bin/python -m minimal_rag search --index data/example.sqlite --query "connection timeout"
.venv/bin/python -m pip install -e '.[codex]'
.venv/bin/python -m minimal_rag ask --live --index data/example.sqlite \
  --question "How should I investigate a connection timeout?" --max-turns 4
```

`ask --live` 默认使用Codex现有ChatGPT登录；`--backend api --model MODEL` 使用API密钥。会发送检索到的原文并消耗对应用量；调用数默认不设上限，可用 `--max-turns` 设置硬停止。模型自主决定搜索词、文档内搜索、翻页和继续读取；主实现不加载向量模型、重排或标题预览。5 篇原文的小例子见 [examples/sol-demo](../examples/sol-demo/README.md)。

## 当前 benchmark

```bash
.venv/bin/python -m pip install -e '.[codex,benchmarks]' -e ./experiments
.venv/bin/python -m minimal_rag_experiments.benchmark --dataset enterprise \
  --split test --live --output runs/bm25-enterprise
```

`--dataset` 支持 `enterprise`、`financebench`、`hotpotqa`、`browsecomp`。前三者使用现有 `data/benchmarks/<dataset>/{dev,test}.json`，BrowseComp 使用冻结的 dev100／剩余730；完整830可合并两个不重叠输出目录。Enterprise、FinanceBench、BrowseComp 读取完整 `corpus.sqlite`；HotpotQA distractor 按官方任务提供的每题10篇文档建同一种 BM25 索引。数据准备仍用 `scripts/prepare_rag_benchmarks.py`、`scripts/index_rag_benchmarks.py`；BrowseComp 下载建库入口保留在对应日期实验目录。

当前生成模型按所选后端明确记录，effort默认high。前三者保留4000字符摘录，BrowseComp保留2000字符；新入口默认不设调用次数上限，预算参数见上表。每题只生成一次，支持同配置 `--resume`，中断记录不会自动重新抽答案。冒烟用 `--split dev --limit 1`，test 始终运行完整题集。默认生成并发5、裁判并发1，可以分别指定。

Enterprise、FinanceBench 默认使用所选后端和模型做medium本地判分（不是官方分数），用 `--skip-judge` 可只生成、留待正式判分；HotpotQA使用官方EM/F1。BrowseComp写出 `official/` 下的官方格式答案，`retrieved_docids`只统计search返回；再交给固定的官方脚本＋Qwen3-32B BF16判分，生成完成本身不代表判分完成。所有输出包含逐题轨迹、配置、请求计数和汇总；历史成绩不会自动成为当前实现成绩。

历史混合检索已移入 `minimal_rag_experiments.hybrid_rag`，需要时安装 `pip install -e './experiments[historical]'`。`rag_optimization_eval`、日期目录的冻结 runner 及 `experiments/scripts/` 下的模型准备、dense 建库脚本只用于历史实验，当前 benchmark 不加载它们。历史成绩与新评测结果分栏见[实验索引](../experiments/README.md)。

2026-10-17 独立向量工具实验已按用户要求终止并[归档](../experiments/semantic-tool-20261016/archive.json)。`semantic_search`和`--semantic-url`不再是当前入口；归档代码、向量分片和云端快照仅保留作历史记录，不自动恢复运行。

## 可选文件工具开发对照

`minimal_rag.file_tools.CorpusFiles`现在只提供Grep，使用ripgrep和完整原文文件视图；`files.tools(kb)`得到search/read/Grep。独立Glob和按行Read已删除，原有按字符read保留。Grep只访问导出的`/corpus`视图，沿用成熟接口的参数名；其中`glob`仍是Grep原生的文件过滤参数，不是另一个工具。默认CLI与benchmark工具集仍保持原样。

Enterprise dev100原三工具对照74→78/100只属于历史版本，其runner使用冻结副本，见[旧协议](../experiments/code-tools-20261017/brief.md)。当前仅Grep的跨基准小样本入口见[BrowseComp dev30协议](../experiments/grep-browsecomp-20261017/brief.md)。复跑应使用新的输出目录；已有目录只允许同配置`--resume`，不会重新生成已有答案。

仅Grep的BrowseComp dev30对照已完成：A/B均28/30，输入token下降但没有正确率提升，因此仍为可选候选，详见[报告](../experiments/reports/grep-browsecomp-20261017.md)。

```bash
.venv/bin/python experiments/grep-browsecomp-20261017/run.py --live \
  --files data/benchmarks/browsecomp-plus/grep-files \
  --output runs/grep-browsecomp-new
```

## 测试

```bash
.venv/bin/python -m pip install -e '.[test,api,codex]' -e './experiments[test]'
.venv/bin/python -m pytest -q tests experiments/tests
```

实验包 `experiments/` 保存评测器、检索变体和各轮实验入口，核心包不导入它。各轮实验的运行命令写在对应日期目录的 brief 或 notes 中；复现历史成绩要从对应提交出发，当前代码不等于当时执行的版本。`data/`、`models/`、`runs/` 被 Git 忽略，保存语料、模型和原始运行材料。
