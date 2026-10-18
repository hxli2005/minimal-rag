"""Transport correctness with fake SDK results; live usage probes are saved separately."""
import asyncio
import json
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock

import pytest
pytest.importorskip("openai_codex")
from openai_codex import ExternalMessage
from minimal_rag.codex_adapter import CodexModel, token_usage
from minimal_rag.loop import run_agent
from minimal_rag.indexed_rag import indexed_tools
from minimal_rag.demo import fixture_index


def output(text="", calls=()):
    return json.dumps({"text": text, "sources": [], "tool_calls": [
        {"name": name, "arguments_json": json.dumps(args)} for name, args in calls]})


def handle(text, ident, total=100):
    usage = {"last": {"input_tokens": 100}, "total": {"input_tokens": total, "cached_input_tokens": 0}}
    result = SimpleNamespace(status=SimpleNamespace(value="completed"), items=[],
                             final_response=text, usage=SimpleNamespace(model_dump=lambda **_: usage))
    return SimpleNamespace(id=ident, run=AsyncMock(return_value=result), interrupt=AsyncMock())


class TransportTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.fixture = fixture_index()
        self.kb = self.fixture.__enter__()

    def tearDown(self):
        self.fixture.__exit__(None, None, None)

    async def test_last_permitted_call_has_no_tool_slots_and_keeps_observations(self):
        thread = SimpleNamespace(id="t", turn=AsyncMock(side_effect=[
            handle(output(calls=[("search", {"query": "MUG-C02"})]), "a"),
            handle(output("Evidence is insufficient."), "b")]))
        model = CodexModel(SimpleNamespace(thread_start=AsyncMock(return_value=thread)), "/tmp", max_calls=2)
        result = await run_agent(model, indexed_tools(self.kb), "MUG-C02?", max_turns=2)
        initial, final = thread.turn.call_args_list
        self.assertGreater(initial.kwargs["output_schema"]["properties"]["tool_calls"]["maxItems"], 0)
        self.assertEqual(final.kwargs["output_schema"]["properties"]["tool_calls"]["maxItems"], 0)
        payload = json.loads(final.args[0].content)
        self.assertEqual(payload["remaining_model_calls"], 1)
        self.assertIn("final_response_required", payload)
        self.assertEqual(payload["messages"][0]["role"], "tool")
        self.assertEqual(result["status"], "finished")
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            await model(model._messages, [])
        self.assertEqual(thread.turn.await_count, 2)

    async def test_noncompliant_final_tool_request_cannot_execute(self):
        thread = SimpleNamespace(id="t", turn=AsyncMock(return_value=handle(
            output(calls=[("search", {"query": "MUG-C02"})]), "a")))
        model = CodexModel(SimpleNamespace(thread_start=AsyncMock(return_value=thread)), "/tmp", max_calls=1)
        kb = self.kb
        kb.search = AsyncMock()
        with self.assertRaisesRegex(RuntimeError, "final response attempted"):
            await run_agent(model, indexed_tools(kb), "question", max_turns=1)
        kb.search.assert_not_awaited()
        self.assertEqual(model.runs[-1]["error_type"], "FinalizationProtocolError")

    async def test_starting_from_existing_history_preserves_assistant_messages(self):
        thread = SimpleNamespace(id="t", turn=AsyncMock(return_value=handle(output("done"), "a")))
        model = CodexModel(SimpleNamespace(thread_start=AsyncMock(return_value=thread)), "/tmp")
        messages = [{"role": "user", "content": "question"},
                    {"role": "assistant", "text": "earlier answer", "tool_calls": [], "sources": []}]
        await model(messages, [])
        self.assertEqual(json.loads(thread.turn.call_args.args[0])["messages"], messages)

    async def test_one_thread_and_only_new_observations_after_tool_execution(self):
        first = handle(output(calls=[("search", {"query": "MUG-C02"})]), "turn-1")
        second = handle(output("定制商品。[C]"), "turn-2", 230)
        second.run.return_value.items = [SimpleNamespace(model_dump=lambda **_: {"type": "functionCallOutput"})]
        thread = SimpleNamespace(id="one-thread", turn=AsyncMock(side_effect=[first, second]))
        client = SimpleNamespace(thread_start=AsyncMock(return_value=thread))
        model = CodexModel(client, "/tmp", max_calls=4)
        result = await run_agent(model, indexed_tools(self.kb), "MUG-C02是什么商品？")
        self.assertEqual(result["status"], "finished")
        client.thread_start.assert_awaited_once()
        options = client.thread_start.call_args.kwargs
        self.assertIn("使用提供的工具", options["base_instructions"])
        self.assertEqual(options["developer_instructions"], "")
        self.assertFalse(options["config"]["skills.include_instructions"])
        initial = json.loads(thread.turn.call_args_list[0].args[0])
        self.assertEqual({t["name"] for t in initial["tools"]}, {"search", "read"})
        following = thread.turn.call_args_list[1].args[0]
        self.assertIsInstance(following, ExternalMessage)
        delta = json.loads(following.content)
        self.assertNotIn("tools", delta)
        self.assertEqual([m["role"] for m in delta["messages"]], ["tool"])
        self.assertEqual(delta["messages"][0]["content"]["hits"][0]["document_id"], "C")
        self.assertEqual([initial["remaining_model_calls"], delta["remaining_model_calls"]], [4, 3])
        self.assertEqual(token_usage(model.runs)["input_tokens"], 230)
        self.assertIn("sources", thread.turn.call_args.kwargs["output_schema"]["required"])

    async def test_new_question_gets_a_separate_thread_and_fresh_budget(self):
        threads = [SimpleNamespace(id=f"thread-{i}", turn=AsyncMock(return_value=handle(output("done"), f"turn-{i}")))
                   for i in range(2)]
        client = SimpleNamespace(thread_start=AsyncMock(side_effect=threads))
        model = CodexModel(client, "/tmp", max_calls=2)
        await run_agent(model, {}, "first")
        await run_agent(model, {}, "second")
        second = json.loads(threads[1].turn.call_args.args[0])
        self.assertEqual(second["messages"], [{"role": "user", "content": "second"}])
        self.assertEqual(second["remaining_model_calls"], 2)
        self.assertEqual(token_usage(model.runs)["input_tokens"], 200)

    async def test_malformed_reply_feedback_is_incremental_and_can_recover(self):
        thread = SimpleNamespace(id="t", turn=AsyncMock(side_effect=[
            handle("invalid JSON", "a"), handle(output("repaired"), "b", 240)]))
        model = CodexModel(SimpleNamespace(thread_start=AsyncMock(return_value=thread)), "/tmp")
        result = await run_agent(model, {}, "question")
        delta = json.loads(thread.turn.call_args.args[0].content)
        self.assertEqual([m["role"] for m in delta["messages"]], ["protocol_error"])
        self.assertEqual(result["answer"], "repaired")

    async def test_cancel_interrupts_turn_and_records_failed_attempt(self):
        turn = handle("", "a")
        turn.run = AsyncMock(side_effect=asyncio.CancelledError())
        thread = SimpleNamespace(id="t", turn=AsyncMock(return_value=turn))
        model = CodexModel(SimpleNamespace(thread_start=AsyncMock(return_value=thread)), "/tmp")
        messages = [{"role": "user", "content": "question"}]
        with self.assertRaises(asyncio.CancelledError):
            await model(messages, [])
        turn.interrupt.assert_awaited_once()
        self.assertEqual(model.runs[0]["status"], "interrupted")
        with self.assertRaisesRegex(RuntimeError, "silently resumed"):
            await model(messages, [])

    async def test_thread_start_failure_is_counted_and_not_reused(self):
        client = SimpleNamespace(thread_start=AsyncMock(side_effect=RuntimeError("startup failed")))
        model = CodexModel(client, "/tmp")
        messages = [{"role": "user", "content": "question"}]
        with self.assertRaisesRegex(RuntimeError, "startup failed"):
            await model(messages, [])
        self.assertEqual(model.runs[0]["status"], "failed")
        with self.assertRaisesRegex(RuntimeError, "silently resumed"):
            await model(messages, [])


class UsageTests(TestCase):
    def test_cumulative_totals_are_not_added_repeatedly_and_old_runs_still_work(self):
        runs = [{"thread_id": tid, "usage": {"total": {"input_tokens": n, "cached_input_tokens": c}}}
                for tid, n, c in [("A", 100, 0), ("B", 90, 20), ("A", 250, 80)]]
        self.assertEqual(token_usage(runs), {"input_tokens": 340, "cached_input_tokens": 100})
        self.assertEqual(token_usage([{"usage": None}]), {})
