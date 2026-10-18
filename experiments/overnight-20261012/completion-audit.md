# 晨报交付前的实际完成核对

本文件是收尾核对，不能代替观察当前状态。目标是完成已授权整晚工作并交付晨报，推进三个项目目标；没有取得官方SOTA，不能把诊断结果改称官方最佳。

| 要求 | 当前证据 | 收尾仍须检查 |
| --- | --- | --- |
| 极简：模型判断、工具能力，核心循环不膨胀 | loop/主检索源码git diff为空；调查账本迁出主包、逐片段SHA删除；core wheel12模块与实验wheel21模块 | 最终diff、包边界；不继续为消耗时间添框架 |
| 同口径可信的SOTA方向 | Enterprise dev24三组、prefix12四组、跨基准、oracle high/max、官方提示词兼容、失败与版本审计，全部保留失败和口径 | Qwen实际gate失败、CPU候选拒绝；官方完整成绩尚缺，必须直说；旧600题原分不动 |
| 求职亮眼且可运行 | README、一页设计、约300字介绍、真实完整库单例、5篇公开原文小库及core CLI真实录制；独立wheel demo通过 | README时态、文件链接与晨报内容一致；不把人工选材demo当检索成绩 |
| 所有新增在线只用sol、最多500 | 498条连续预扣，498个completed记录，全部用量齐全；325生成173评分，见reconciliation | 账本仍498且没有新的在线进程；禁止利用余2次开新实验 |
| 不打断Qwen、不争GPU | 原PID12453自然完成，未发信号；本聊天仅在原任务结束后运行单线程CPU重排，实际参数CPU/float32；capture的六源码未变 | 最终阶段complete_without_full_build、自己的consumer已停；不能重启编码或修改gate |
| 不提交main | 当前codex/standalone-rag，完成版本已本地提交；没有向main提交、合并或发布 | 实际branch/status；不要从历史branch字段推断 |
| 排满整晚并在早上晨报 | 已安排本聊天30分钟heartbeat至08:00收尾；持续推进原文、输出、用量及gate核对 | 北京时间07:30后实际收尾、08:00交付；此前保持goal active |
| 晨报每项结论先行、做了什么、数据、下一步 | morning.py默认生成本地草稿，全部额外比较输入已接入 | `--final`后人工核对Qwen/完整官方缺口、案例限制及唯一下一步推荐；链接文件并在final聊天给自包含摘要 |
| 完成后结束自动续办 | heartbeat ID minimal-rag | 交付后通过automation_update关闭/暂停该heartbeat，再mark goal complete；不能先宣称完成 |

必要验证已通过：28项受影响行为／包边界、55项core、独立core wheel demo、调查离线demo、预算多线程边界、CPU consumer小库、记录核账、活动文档链接。除有新源码改动或失败，不重复相同检查。晨报生成不是部署或用户接受证据。

Qwen全量如果尚未完成，只报告进度和自动CPU评测入口。原任务通过抽样gate后自动建全量是既有授权；consumer可继续等待其完成。抽样gate未通过时，应核实原进程已结束后停止本聊天自己的等待consumer，保存原因，绝不能杀原编码任务或人为改gate。

07:31实际收尾核对已完成：498连续sol预扣、六原源码SHA匹配、原stage complete_without_full_build、full目录不存在、consumer已停。07:35 ps未见本聊天在线worker、原Qwen命令、CPU辅助或临时server。26份已有evidence JSON和21页活动Markdown本地链接通过；新晨报页另外检查。晨报已经生成并人工补齐配对区间／预算与下一轮唯一推荐；尚待08:00实际交付、关闭heartbeat和mark goal complete。

07:41后完成版本已提交到codex/standalone-rag，未推送、未合并；预算498不变。实际提交SHA以Git HEAD和08:00聊天交付为准，07:31核查中的head是提交前基线。
