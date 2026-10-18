import builtins
import json

import pytest

httpx = pytest.importorskip("httpx")

from minimal_rag.api_adapter import APIModel
from minimal_rag.codex_adapter import token_usage
from minimal_rag.demo import fixture_index
from minimal_rag.indexed_rag import indexed_tools
from minimal_rag.loop import run_agent
from minimal_rag.model_backend import TokenBudget, TokenBudgetExceeded, model_session


def api_response(reply, number=1, status="completed"):
    output = [{"type": "reasoning", "id": f"rs_{number}", "summary": [], "encrypted_content": "opaque"}]
    if reply["tool_calls"]:
        output.extend({"type": "function_call", "id": f"fc_{number}_{i}", "call_id": f"call_{number}_{i}",
                       "name": call["name"], "arguments": call["arguments_json"], "status": "completed"}
                      for i, call in enumerate(reply["tool_calls"]))
    else:
        output.append({"type": "message", "id": f"msg_{number}", "role": "assistant", "status": "completed",
                       "content": [{"type": "output_text", "text": json.dumps({"answer": reply["text"],
                                    "sources": reply["sources"]}), "annotations": []}]})
    return {"id": f"resp_{number}", "model": "fixture-api", "status": status,
            "usage": {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 40},
                      "output_tokens": 20, "total_tokens": 120},
            "output": output}


@pytest.mark.asyncio
async def test_api_retains_context_beyond_twelve_turns_with_native_tools_and_no_codex(monkeypatch):
    original_import = builtins.__import__
    def without_codex(name, *args, **kwargs):
        if name.startswith("openai_codex"):
            raise ImportError("Codex not installed")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_codex)
    requests = []
    def serve(request):
        body = json.loads(request.content); requests.append(body)
        number = len(requests)
        if number == 1:
            assert {t["name"] for t in body["tools"]} == {"search", "read"}
            assert body["input"] == [{"role": "user", "content": "Question"}]
            assert "tool_calls" not in body["text"]["format"]["schema"]["properties"]
            assert "外层JSON" not in body["instructions"]
        elif number <= 14:
            assert body["input"][-3]["type"] == "reasoning"
            observation = body["input"][-1]
            assert observation["type"] == "function_call_output"
            assert observation["call_id"] == f"call_{number-1}_0"
            assert json.loads(observation["output"])["hits"]
        reply = {"text": "Done [C]" if number == 14 else "", "sources": ["C"] if number == 14 else [],
                 "tool_calls": [] if number == 14 else [{"name": "search", "arguments_json": '{"query":"MUG-C02"}'}]}
        assert "max_output_tokens" not in body and body["store"] is False
        assert body["text"]["verbosity"] == "low"
        return httpx.Response(200, json=api_response(reply, number))
    async with httpx.AsyncClient(transport=httpx.MockTransport(serve), base_url="https://fixture/v1/") as client:
        model = APIModel(client, "fixture-api")
        with fixture_index() as kb:
            result = await run_agent(model, indexed_tools(kb), "Question")
        assert result == {"status": "finished", "answer": "Done [C]", "sources": ["C"]}
        assert len(requests) == 14
        assert token_usage(model.runs) == {"input_tokens": 1400, "cached_input_tokens": 560,
                                          "output_tokens": 280, "total_tokens": 1680}
        await model([{"role": "user", "content": "New question"}], [])
        assert len(requests[-1]["input"]) == 1  # no cross-question history


@pytest.mark.asyncio
async def test_parallel_native_calls_keep_ids_and_return_validation_errors_to_model():
    bodies = []
    def serve(request):
        body = json.loads(request.content); bodies.append(body)
        if len(bodies) == 1:
            reply = {"text": "", "sources": [], "tool_calls": [
                {"name": "read", "arguments_json": '{"document_id":"C"}'},
                {"name": "read", "arguments_json": '{"document_id":"B","offset":"invalid"}'}]}
        else:
            results = body["input"][-2:]
            assert [r["call_id"] for r in results] == ["call_1_0", "call_1_1"]
            assert json.loads(results[0]["output"])["document_id"] == "C"
            assert json.loads(results[1]["output"])["error"] == "ValidationError"
            reply = {"text": "Done [C]", "sources": ["C"], "tool_calls": []}
        return httpx.Response(200, json=api_response(reply, len(bodies)))
    async with httpx.AsyncClient(transport=httpx.MockTransport(serve), base_url="https://fixture/") as client:
        with fixture_index() as kb:
            result = await run_agent(APIModel(client, "fixture-api"), indexed_tools(kb), "Question")
        assert result["sources"] == ["C"] and len(bodies) == 2


@pytest.mark.asyncio
async def test_incomplete_api_answer_is_not_counted_as_finished():
    def serve(request):
        return httpx.Response(200, json=api_response({"text": "partial", "tool_calls": [], "sources": []}, status="incomplete"))
    async with httpx.AsyncClient(transport=httpx.MockTransport(serve), base_url="https://fixture/") as client:
        model = APIModel(client, "fixture-api")
        with pytest.raises(RuntimeError, match="did not complete"):
            await run_agent(model, {}, "Question")
        assert token_usage(model.runs)["total_tokens"] == 120


@pytest.mark.asyncio
async def test_token_budget_stops_before_another_request_without_forcing_answer():
    def serve(request):
        return httpx.Response(200, json=api_response({"text": "", "sources": [],
            "tool_calls": [{"name": "search", "arguments_json": '{"query":"MUG-C02"}'}]}))
    async with httpx.AsyncClient(transport=httpx.MockTransport(serve), base_url="https://fixture/") as client:
        model = TokenBudget(APIModel(client, "fixture-api"), 120)
        with fixture_index() as kb, pytest.raises(TokenBudgetExceeded):
            await run_agent(model, indexed_tools(kb), "Question")
        assert len(model.runs) == 1


@pytest.mark.asyncio
async def test_api_requires_key_and_never_falls_back_to_subscription(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        async with model_session("api", "fixture-api", 300):
            pytest.fail("Missing API key was accepted")
