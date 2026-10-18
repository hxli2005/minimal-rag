# 用核心 CLI 运行真实模型的小例子

**这 5 篇公开模拟企业原文可以直接建库，不需要下载 Enterprise 全库。** 资料人工选自一个已知开发案例，包含目标政策和四篇原首次查询的其他结果；它是演示，不是独立检索质量评测。模型自己决定查询、阅读和回答，源码没有写死模型决策。

本地用浏览器打开 [可视回放](report.html)，通过三个按钮查看实际检索、原文读取和完整答案。页面自带录制数据，无需服务器或账号，打开它不会调用模型。[预览图](preview.png)

```bash
# 在仓库根目录，完成 README 的安装后
.venv/bin/python -m minimal_rag index examples/sol-demo/documents.jsonl --output /tmp/sol-demo.sqlite
.venv/bin/python -m pip install -e '.[codex]'
.venv/bin/python -m minimal_rag ask --live --index /tmp/sol-demo.sqlite \
  --model gpt-6.1-sol --max-turns 4 \
  --question "What are the proposed internal time to mitigate targets by severity for incident response, regardless of customer deployment tier?"
```

建库拒绝覆盖已有文件；重复运行可直接复用已生成的索引。`ask --live` 需要现有 ChatGPT 登录，最多四次外层生成，消耗模型用量。题目询问历史文档中的 **proposed internal targets**，不代表真实公司当前承诺。

查阅 [已有真实 sol 报告](../../experiments/reports/real-sol-run-20261012.md)、[选材来源](provenance.json)与 [第三方许可](../../THIRD_PARTY_NOTICES.md)。本夜所有额外在线调用由 dated experiment 的共享账本控制；这些命令是读者以后主动运行的入口，不会因查看 README 自动调用模型。

本小库也已有 [真实录制的完整运行](recorded-summary.json)：2026-10-13 的 `gpt-6.1-sol`，3 次生成，25,079 输入 token（含13,440缓存）、288输出token。模型自己检索和阅读后回答 Sev0 30分钟、Sev1 2小时、Sev2 1工作日、Sev3 5工作日或下一发布列车，并保留draft限制。数值已人工对照原文，未新增模型评分；不作基准准确率或费用声明。
