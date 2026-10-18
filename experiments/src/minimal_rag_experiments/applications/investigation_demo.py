"""Authored decision replay over released synthetic-company source documents.

The answer and search plan are curated, not model-generated. Retrieval, source
reads, passage checking and call-budget enforcement execute for real, offline.
"""
import json
import tempfile
from importlib.resources import files
from pathlib import Path

from minimal_rag_experiments.evidence import CallBudget
from minimal_rag.indexed_rag import IndexedKnowledgeBase, indexed_tools
from minimal_rag_experiments.applications.investigation import investigate


TICKET = "dsid_abfb68dde21f4fd2aa7d70c472447f89"
FIX = "dsid_0f9c8d82259c4d5a95bc5865fc492919"
RUNBOOK = "dsid_8761a5900a7f422e8a2606576740f2bc"
QUESTION = ("请为 Northwind 的值班工程师整理交接：Runtime 1.21 开启 kernel auto-selection v2 后，"
            "Dedicated L40S 长上下文请求 p99 回退。历史上采取了什么缓解，怎样核查恢复，"
            "在什么条件下才能重新开启 v2？明确当前仍需确认的信息。")


def curated_brief(receipts):
    ticket, fix, runbook = (receipts[k] for k in (TICKET, FIX, RUNBOOK))
    return f"""## 事件与影响
- 历史工单记录的范围是 Northwind 的 nw-analytics-ded-01 池、us-east、L40S，以及 Runtime 1.21.0-canary.3 下的 Llama 长上下文请求；约 20k 输入时 p99 从 4.8–5.4 秒升至 8.9–10.6 秒，短请求和错误率未出现同类变化。这是工单中的历史观测。[{ticket}]
## 原因与证据
- 工单与修复 PR 指向长上下文 prefill 分桶和 L40S 内核选择：LONG 场景选用了表现较差的 fa3_fused_prefill_sm89，prefill 时间上升而 decode 基本稳定。PR 增加长上下文保护、边界滞回与 SKU 规则；文档中的修复描述不等于当前客户池已完成部署。[{ticket}] [{fix}]
## 已采取的缓解
- 工单记录已在受影响池关闭两个 Llama 模型的 v2，并添加 L40S + LONG 的临时内核 deny。客户随后反馈 p99 回到约 5.3–5.8 秒且无新错误；这不证明本次查看时服务仍然健康。[{ticket}]
## 核查与恢复条件
- 交接时核对现场 gate/config 状态、内核选择分布，以及相同 model/SKU/seq_bucket 下的 p95、p99、错误率和正确性信号；确认配置传播，保留回滚路径。具体执行须由有权限的值班工程师确认。[{runbook}]
- 客户特定条件是保留 deny 至 1.21 GA，待 1.21.1 补丁及 pr-28522 落地、L40S 长上下文验证通过后，再安排客户池 30–60 分钟的受控 canary；LONG bucket p99 回退超过 10% 是工单指定的回滚触发条件。[{ticket}]
- 修复 PR 的通用建议是先 Hosted canary、再经 24 小时观察推进 Dedicated；这属于通用发布路径，不能替代 Northwind 工单中的客户池验证和补丁前置条件。[{fix}] [{ticket}]
## 未知与交接
- 待核查：当前客户池是否已经安装 1.21.1 并包含对应修复，deny/gate 是否仍保持预期；本工具未连接现场配置和监控。
- 待核查：取得此次异常的 request IDs、时间窗口及 prompt-length 分布，核对是否与历史事件属于同一工作负载，并由值班工程师确认变更范围与执行人。
- 待核查：driver 550.54.14 / CUDA 12.4 的潜在交互是否已完成验证；工单曾要求验证完成前保留 deny。[{ticket}]
"""


async def demo(*, max_calls=4, timeout=180, checkpoint=lambda _: None):
    resources = files("minimal_rag_experiments").joinpath("fixtures")
    documents = [json.loads(line) for line in resources.joinpath("northwind.jsonl").read_text(encoding="utf-8").splitlines() if line]
    manifest = json.loads(resources.joinpath("northwind-manifest.json").read_text(encoding="utf-8"))
    step = 0

    async def replay(messages, specifications):
        nonlocal step
        step += 1
        calls = []
        if step == 1:
            calls = [("search", {"query": "Northwind L40S", "all_terms": True, "limit": 6})]
        elif step == 2:
            calls = [("search", {"query": '"long-context" "prefill" "thresholds"', "all_terms": True, "limit": 6})]
        elif step == 3:
            calls = [("read", {"document_id": ident, "limit": 16000}) for ident in (TICKET, FIX, RUNBOOK)]
        if calls:
            return json.dumps({"text": "", "sources": [], "tool_calls": [
                {"name": name, "arguments_json": json.dumps(args)} for name, args in calls]})
        receipts = {event["content"]["document_id"]: event["content"]["evidence_id"]
                    for event in messages if event.get("role") == "tool" and event.get("name") == "read"
                    and "evidence_id" in event["content"]}
        return json.dumps({"text": curated_brief(receipts), "tool_calls": [],
                           "sources": list(receipts.values())}, ensure_ascii=False)

    with tempfile.TemporaryDirectory(prefix="rag-investigation-demo-") as temp:
        path = Path(temp) / "corpus.sqlite"
        IndexedKnowledgeBase.build(path, documents)
        kb = IndexedKnowledgeBase(path)
        try:
            report = await investigate(replay, indexed_tools(kb), kb.read, QUESTION,
                                       budget=CallBudget(max_calls), mode="offline_authored_replay",
                                       timeout=timeout, checkpoint=checkpoint)
        finally:
            kb.close()
    report["corpus"] = manifest
    report["generation_calls"] = 0
    report["demonstration_notice"] = "人工编写决策与简报，使用真实本地检索和证据校验；模型生成与裁判调用均为 0。已知案例，不是质量评测。"
    return report
