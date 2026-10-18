"""测试最小循环的自由度和执行边界；脚本化回复不代表真实模型表现。"""
import asyncio
import copy
import json
import sqlite3
import unittest

from minimal_rag.indexed_rag import SearchArgs
from minimal_rag.loop import Tool, run_agent
from minimal_rag.indexed_rag import indexed_tools
from minimal_rag.demo import fixture_index


def reply(text="", calls=()):
    return json.dumps({"text": text, "tool_calls": [
        {"name": name, "arguments_json": json.dumps(args, ensure_ascii=False)}
        for name, args in calls]}, ensure_ascii=False)


class ScriptedModel:
    def __init__(self, replies):
        self.replies, self.inputs = iter(replies), []

    async def __call__(self, messages, tools):
        self.inputs.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        return next(self.replies)


class LoopTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fixture = fixture_index()
        self.kb = self.fixture.__enter__()

    def tearDown(self):
        self.fixture.__exit__(None, None, None)

    async def test_model_can_finish_without_a_prescribed_workflow(self):
        model = ScriptedModel([reply("你好，请告诉我你的问题。")])
        result = await run_agent(model, indexed_tools(self.kb), "你好")
        self.assertEqual(result, {"status": "finished", "answer": "你好，请告诉我你的问题。", "sources": []})
        self.assertNotIn("verified", result)

    async def test_search_observation_goes_back_to_model_and_can_support_answer(self):
        model = ScriptedModel([reply(calls=[("search", {"query": "MUG-C02"})]), reply("该型号属于定制商品。[C]")])
        events = []
        result = await run_agent(model, indexed_tools(self.kb), "型号是什么？", emit=events.append)
        messages, tools = model.inputs[1]
        self.assertEqual(messages[0]["content"], "型号是什么？")
        self.assertIn("定制", messages[-1]["content"]["hits"][0]["text"])
        self.assertEqual({t["name"] for t in tools}, {"search", "read"})
        self.assertEqual([e["role"] for e in events], ["assistant", "tool", "assistant"])
        self.assertEqual(result["answer"], "该型号属于定制商品。[C]")

    async def test_multiple_tool_requests_and_intermediate_text_are_preserved(self):
        model = ScriptedModel([reply("查两个来源。", [("read", {"document_id": "C"}), ("read", {"document_id": "B"})]), reply("结论。[C][B]")])
        await run_agent(model, indexed_tools(self.kb), "问题")
        messages = model.inputs[1][0]
        self.assertEqual(messages[1]["text"], "查两个来源。")
        self.assertEqual([m["content"]["document_id"] for m in messages if m["role"] == "tool"], ["C", "B"])

    async def test_same_tool_can_be_retried_without_a_no_progress_judge(self):
        model = ScriptedModel([reply(calls=[("search", {"query": "MUG-C02"})])] * 4 + [reply("已核对。[C]")])
        events = []
        result = await run_agent(model, indexed_tools(self.kb), "问题", emit=events.append)
        self.assertEqual(result["status"], "finished")
        self.assertEqual(sum(e["role"] == "tool" for e in events), 4)
        self.assertTrue(all("error" not in e["content"] for e in events if e["role"] == "tool"))

    async def test_read_after_source_update_returns_new_content(self):
        kb = self.kb
        model = ScriptedModel([reply(calls=[("read", {"document_id": "B"})])] * 2 + [reply("更新已读。")])
        events = []

        def observe(event):
            events.append(event)
            if event["role"] == "tool":
                with sqlite3.connect(kb.path) as writer:
                    writer.execute("UPDATE documents SET title=?,text=? WHERE id=?", ("规则", "新版本规则。", "B"))

        await run_agent(model, indexed_tools(kb), "问题", emit=observe)
        reads = [e["content"] for e in events if e["role"] == "tool"]
        self.assertNotEqual(reads[0]["text"], reads[1]["text"])
        self.assertEqual(reads[1]["text"], "新版本规则。")

    async def test_unknown_tool_and_invalid_arguments_return_errors_without_execution(self):
        model = ScriptedModel([
            reply(calls=[("shell", {"command": "anything"})]),
            reply(calls=[("search", {"query": "薪酬", "actor": "finance"})]),
            reply(calls=[("read", {"document_id": "PRIVATE"})]),
            reply("资料不可用。"),
        ])
        result = await run_agent(model, indexed_tools(self.kb), "问题")
        self.assertEqual(result["status"], "finished")
        for messages, _ in model.inputs[1:]:
            self.assertIn("error", messages[-1]["content"])

    async def test_model_can_repair_invalid_protocol(self):
        model = ScriptedModel(["not JSON", reply("修复后的回答")])
        result = await run_agent(model, {}, "问题")
        self.assertEqual(model.inputs[1][0][-1]["role"], "protocol_error")
        self.assertEqual(result["answer"], "修复后的回答")

    async def test_multiline_answer_and_citations_need_no_nested_json(self):
        model = ScriptedModel([json.dumps({"text": 'First line.\nA quoted "detail".', "tool_calls": [], "sources": ["B"]})])
        result = await run_agent(model, {}, "question")
        self.assertEqual(result["answer"], 'First line.\nA quoted "detail".')
        self.assertEqual(result["sources"], ["B"])

    async def test_only_resource_budget_ends_unfinished_loop(self):
        model = ScriptedModel([reply(calls=[("search", {"query": "MUG-C02"})])] * 2)
        result = await run_agent(model, indexed_tools(self.kb), "问题", max_turns=2)
        self.assertEqual(result["status"], "budget_exhausted")

    async def test_async_tool_errors_are_feedback_and_can_be_retried(self):
        count = 0

        async def flaky(query, limit, offset):
            nonlocal count
            count += 1
            if count == 1:
                raise RuntimeError("temporary failure")
            return {"text": "真实工具结果"}

        model = ScriptedModel([reply(calls=[("lookup", {"query": "x"})])] * 2 + [reply("完成")])
        result = await run_agent(model, {"lookup": Tool("查询", SearchArgs, flaky)}, "问题")
        self.assertEqual(model.inputs[1][0][-1]["content"]["error"], "RuntimeError")
        self.assertEqual(model.inputs[2][0][-1]["content"]["text"], "真实工具结果")
        self.assertEqual(result["status"], "finished")

    async def test_cancellation_propagates(self):
        async def cancelled(messages, tools):
            raise asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await run_agent(cancelled, {}, "问题")

    async def test_conversations_are_local_to_each_run(self):
        model = ScriptedModel([reply("a"), reply("b")])
        await run_agent(model, {}, "first")
        await run_agent(model, {}, "second")
        self.assertEqual(model.inputs[1][0], [{"role": "user", "content": "second"}])


if __name__ == "__main__":
    unittest.main()
