# Enterprise dev100：成熟文件检索接口对照

后续状态：用户要求裁剪为仅Grep。原三工具实现保存在本目录frozen_file_tools.py，旧runner显式使用它；本报告与原结果不变。当前主包及后续实验只提供Grep增量。

用户授权：不换框架，按成熟代码助手原接口加入 grep、glob、read，由模型决定参数和调用次数；先完成 Enterprise 开发集同口径对照。

## 固定比较

- A：当前纯 BM25 `search/read`，包括已采用的新搜索说明。
- B：A 加 `Grep/Glob/Read`；search 命中多返回 file_path，附一句路径说明。既有 search/read 的查询、排名、片段、参数与实现不改。比较的是整个新增工具包，不能单独归因于某个工具。
- 固定现有 `enterprise/dev.json` 全部100题与完整511,962篇语料；两组都重新生成，不复用历史答案，不查看test题开发工具。
- 相同 gpt-6.1-sol high、原系统提示与问题格式、最多12次模型调用、每次180秒、每题900秒。共用生成并发5，按题交错A/B顺序；原sol medium盲裁判并发1。
- 模型自行选择工具、参数、调用顺序、次数和结束。无固定检索流程、额外查询扩展、向量或重排。
- 首次有效答案和判决保留，生成失败计错留在100题分母。只对裁判传输/格式失败使用原有最多3次重试；resume不重新生成已启动的问题。额度不足停止，不消费重置券或购买额度。
- 同轮数与超时上限，不宣称严格等token：新工具的默认返回量、行单位和接口说明是处理变量，实际输入（含缓存/不含缓存）、输出、时间与工具调用另报。

## 接口与适配边界

参数与输出字段依据已发布的 [Claude Agent SDK 0.3.301](https://unpkg.com/@anthropic-ai/claude-agent-sdk@0.3.301/sdk-tools.d.ts)，行为参照[官方工具说明](https://code.claude.com/docs/en/tools-reference)。这是公开接口的文本语料适配，不声称复刻厂商整个运行时或训练分布。

- `Grep`：pattern/path/glob/type/output_mode、-A/-B/-C/context、-i/-n/-o、head_limit/offset/multiline。默认files_with_matches、head_limit250、offset0；0取消条数限制，仍受输出预算约束。count的总数来自完整搜索，不只计算显示页面。正则错误原样返回诊断。
- `Glob`：pattern/path，标准递归与花括号模式，最多100路径，返回截断提示和总数。没有新增分页参数。
- `Read`：file_path、offset、limit、allow_large；offset从1开始按行，limit按行。仅原始文本，省略范围读取全文；不提供PDF pages参数或图像功能。
- 新工具匹配文本/路径的正文预算为80,000字符，结构元数据另计；Grep完整输出先扫描后分页，超出文本预算明确提示，不把截断说成无匹配。原工具原上限保留。
- `/corpus` 为只读语料路径视图。目录按真实source类型及ID末两位分片；文件名为ID加清理/截短标题，正文不添加标题或其他内容。所有mtime相同，不虚构来源时间；Glob并列按路径排序。ID由catalog映射，最终答案仍引用原文档ID。
- 全库导出逐篇字节比较原SQLite文本；原文含NUL的37篇仍作为文本参与grep。rg真实执行，原生正则、大小写、跨行、文件类型与上下文语义；无BM25候选前置过滤。模型无法访问题目标签、裁判或工作区文件。
- 路径发现和正文曝光分开。仅Grep content与非空Read计为新工具的正文曝光；Glob、files_with_matches和count不冒充已读证据。B的工具观察附带这些原文ID作为引用映射，下一轮历史可见；它们仅来自工具返回，不含参考答案或gold标签。

## 评估与决策

主要指标为100题正确数，报告逐题胜/负/平、按题型结果及配对bootstrap 95%区间（100,000次，seed20261017）。报告模型与工具调用、错误、时间、token实耗、正文参考文档曝光，以及新工具使用频率。开发集不用于宣称官方成绩或SOTA。B优于A只构成后续冻结test对照的候选依据，不自动替换默认工具或启动test；无收益则保留记录、默认仍A。

## 运行

```bash
.venv/bin/python experiments/code-tools-20261017/export_corpus.py \
  --index data/benchmarks/enterprise/corpus.sqlite \
  --output data/benchmarks/enterprise/code-tool-files
.venv/bin/python experiments/code-tools-20261017/run.py --live \
  --files data/benchmarks/enterprise/code-tool-files \
  --output runs/code-tools-20261017
```

导出只读全库原文，不读取dev/test标签。运行结果、逐题工具参数/返回和裁判尝试保存在runs目录；可提交的汇总不发布题目与标准答案。

## 开跑前验证

- 143项核心及实验测试通过，新增5项覆盖真实rg行为、上下文、大小写、跨行、分页、count总量、glob、按行读取、原接口参数经过原循环的执行。
- 全511,962篇导出逐篇对比原始UTF-8字节相同，见[导出验证](export-validation.json)。正文共2,464,987,973字符；最长41,317字符。
- 最终并行全库Grep探测约13.9–14.6秒，Glob约2.1秒；这是开跑前单请求实测，不代替实验中的并发耗时。见[耗时验证](latency-parallel.json)。
- UTC 2026-10-17 10:28启动正式对照。运行时不修改工具、提示或评测设置。
