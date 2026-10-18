# 官方入口：提示词兼容完成，同口径成绩尚未产生

**结论：当前只能发布本地兼容检查，不能发布 EnterpriseRAG 官方成绩或 SOTA。** 固定官方源码版本 `d36685e273713975ee20299bbf1ab64165575b3c`，答案 JSONL 导出与正确性／逐事实提示词已经跑通。固定前 4 个开发答案，sol 29 次评分，4/4 正确且事实全部支持；它们本来就是当前成功答案，检查的是接口与口径，不估计总体准确率。[记录](../../evidence/official-prompt-compatibility-20261013.json)

2026-10-13 核对的官方榜单仍由 CDL 领先：overall 90.91、correctness 92.4%、completeness 94.37%、document recall 93.13%。榜单仓库版本为 `85b78cb52cbcddbfe462dde6b0e68a2a29739613`，最后更新 2026-10-07。这些数不能与本地 24 题或旧 400 题直接相减。[固定榜单 CSV](https://huggingface.co/spaces/onyx-dot-app/EnterpriseRAG-Bench-Leaderboard/resolve/85b78cb52cbcddbfe462dde6b0e68a2a29739613/data/final_display_data/leaderboard.csv)

## 尚未对齐的部分

| 官方实现 | 本地兼容入口 |
| --- | --- |
| 默认 OpenAI 提供方与 gpt-5.4，可经环境覆盖 | 遵守本夜限制，仅 gpt-6.1-sol Codex 传输；默认源码不是榜单实际环境的证明 |
| 正确性与逐事实独立评分 | 同固定官方提示词独立调用；保留 Codex 外层结构化协议 |
| LLM 去引用、三次文档分类共识，必要时更新 gold 答案／事实 | 冻结参考，未执行这些步骤，明确不是完整流程 |
| 全部 500 主问题及匹配 corpus | 历史500题已完整格式导出；新sol24题导出／4题兼容评分，均未跑完整官方评测或提交榜单 |

官方流程、格式和独立评分来源见 [评测入口](https://github.com/onyx-dot-app/EnterpriseRAG-Bench/tree/d36685e273713975ee20299bbf1ab64165575b3c/src/scripts/answer_evaluation)、[方法](https://github.com/onyx-dot-app/EnterpriseRAG-Bench/blob/d36685e273713975ee20299bbf1ab64165575b3c/methodology.md)。本夜不调用默认其他模型，也没有为了凑“同口径”偷偷改 gold。

## UUID 与文本版本要先明确

本地原发布资料有 4 组非相同内容的重复 ID；本地保留为 `~duplicate2`。导出转换回官方 ID，并保留本地→官方映射；本批 24 个 agent 答案没有这种后缀引用。

官方 UUID 索引对重复 UUID 采用后遍历的文件覆盖；同一 UUID 的不同正文不会被分别标识。qst_0413 已证明新旧内容会改变正确性解释。提交前必须记录实际物理文件映射与对应版本，不能把“引用 ID 相同”当成“资料相同”。[官方索引实现](https://github.com/onyx-dot-app/EnterpriseRAG-Bench/blob/d36685e273713975ee20299bbf1ab64165575b3c/src/utils/document_index.py)

## 完整历史答案输入

**500题答案已可直接检查，仍没有新增评分。** [官方格式JSONL](../../runs/evidence/historical-enterprise-500-answers.jsonl)包含dev100＋原test400，500/500原生成完成；原文字、内文引用均未修改，没有新增astra或其他模型请求。另存[导出凭据与正文版本](../../evidence/official-history-export-20261013.json)：899个本地实际引用正文（合为898个官方UUID）、905个含四组版本替代的映射，未知引用ID为0、未观察引用的题为0。

qst_0413同时引用canonical与duplicate2，是这500题中唯一引用别名的题；转成官方UUID后两者合为一个ID。凭据保留两个物理来源与正文指纹，不能宣称导出本身消除了歧义。官方实际UUID索引仍须与这里的来源映射对齐。

905个正文版本均已与发行Parquet中的精确正文匹配、绑定零起始行号，没有未绑定项；整个1.41GB文件的SHA与原发行manifest一致。qst_0413的新／旧正文分别在189417和191029行。这证明本地资料来源，不证明官方Git JSON文件的遍历选择；不能把Parquet的先后顺序直接当成官方UUID覆盖顺序。

离线导出命令：`.venv/bin/python experiments/overnight-20261012/export_official_history.py`。需要保留的本地历史轨迹与corpus；仓库中的JSONL和凭据则无需下载语料即可检查。官方去引用、参考文档共识／纠正、答案重写及完整评分尚未执行，历史77.75%等本地成绩原样保留。

下一步先冻结 corpus 与 UUID 映射，再在允许的裁判模型、完整流程和统一题集下验证候选。当前已经接触的 500 题属于已知回归，不再称未接触的独立测试集。
