# v3：只迁移当前适配器，保留v0检索和完整摘录

**决定：从current_best=v0出发，仅将Enterprise历史Codex适配器换为当前src适配器。** 历史检索快照、hybrid/cascade/15条预览、4000字符search摘录、read、提示和裁判全部不变；BrowseComp已使用当前适配器，故仍复用A dev100。不是从失败v1/v2继续叠改，也不是两项统一BM25。

依据：v1组合迁移使Enterprise dev83→79，参考覆盖下降且调用增加；v2只缩到2000字符，panel26→23（1胜4负），虽未缓存输入779690→591682，仍净输3题而拒绝。两轮都有已检索文档中的条件遗漏，因此恢复v0完整上下文。过去“完整覆盖”单句提示24题两组同为19/24，无采用证据，本夜不重复该候选。v0 dev有34题曾从标题预览直接read，不能无依据删除预览。

当前适配器的SYSTEM_PROMPT、PROTOCOL、SDK配置和rag_codex_config与历史版本AST一致。CodexModel的实际变化是调用上限校验，以及最后一轮用final_response_required和tool_calls.maxItems=0要求收束；不改变12次上限，不增加模型调用。迁移使两项主基准共用当前适配器，去掉Enterprise的历史适配器恢复依赖，是明确的实现简化；检索仍为历史工具快照，不夸大为全部统一当前代码。

模型gpt-6.1-sol/high，Enterprise12轮、请求180秒/题900秒、v6提示/concise；本地sol medium裁判提示不变。BrowseComp A的30轮、180/1800秒与OpenRouter Qwen配置保持原样，100题复用。数据、分母、门槛、seed20261016及bootstrap10000次不变。

沿用night_id=20261015T102540，current_best=night_anchor=v0；本夜第3/最多3候选。panel60→dev200→Enterprise已知test400；同时对两个基线过关才可采用，复杂度简化作为收益，不能用它豁免质量/成本。主阶段600/1400/2800与整夜9000探索＋1000补判原账本不变；最多8小时，最后90/30分钟限制不变。预计必要链路2750请求/80分钟，按实测加20%再估计。

在线前验证：仅适配器模块来自src，其余检索模块来自冻结历史装配；变量/提示逐项比对，保存唯一差异的AST摘要。首3题在线装配验证后才继续panel。结束后无论采用或拒绝，候选数已满，冻结current_best；有资源才做最终Finance辅助，然后提交晨报并进入官方GPU复核。Hotpot0，不换模型、不改src/评分公式，不推送。
