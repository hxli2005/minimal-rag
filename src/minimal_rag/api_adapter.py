"""Direct Responses API transport; no Codex runtime or built-in agent tools."""
import json

from pydantic import Field

from minimal_rag.prompts import SYSTEM_PROMPT
from minimal_rag.loop import Reply, StrictModel, ToolCall


class Answer(StrictModel):
    answer: str
    sources: list[str] = Field(description="Document IDs supporting the answer; empty when no source supports it.")


class APIModel:
    def __init__(self, client, model, timeout=300, effort="high", system_prompt=SYSTEM_PROMPT, verbosity="low"):
        self.client, self.model, self.timeout = client, model, timeout
        self.effort, self.system_prompt = effort, system_prompt
        self.verbosity = verbosity
        self.runs = []
        self._messages, self._cursor, self._history = None, 0, []
        self._pending = []

    async def __call__(self, messages, tools):
        fresh = messages is not self._messages
        if fresh:
            self._messages, self._cursor, self._history = messages, 0, []
            self._pending = []
        delta = messages if fresh else [m for m in messages[self._cursor:] if m["role"] != "assistant"]
        for message in delta:
            if message["role"] == "tool":
                call = self._pending.pop(0)
                self._history.append({"type": "function_call_output", "call_id": call["call_id"],
                                      "output": json.dumps(message["content"], ensure_ascii=False)})
            elif message["role"] == "protocol_error":
                # Resolve native calls even if the generic loop rejected their envelope.
                self._history.extend({"type": "function_call_output", "call_id": call["call_id"],
                                      "output": json.dumps({"error": message["error"]})} for call in self._pending)
                self._pending = []
                self._history.append({"role": "user", "content": message["error"]})
            else:
                self._history.append(message)
        body = {"model": self.model, "store": False, "input": self._history,
                "instructions": self.system_prompt, "text": {"verbosity": self.verbosity}}
        if tools:
            # Preserve optional arguments and their local defaults; do not silently require every field.
            body["tools"] = [{"type": "function", **tool, "strict": False} for tool in tools]
            body["text"]["format"] = {"type": "json_schema", "name": "answer",
                                      "schema": Answer.model_json_schema(), "strict": True}
        if self.effort != "none":
            body.update(reasoning={"effort": self.effort}, include=["reasoning.encrypted_content"])
        record = {"status": "started", "transport": "responses-api-native-tools", "usage": None, "verbosity": self.verbosity}
        self.runs.append(record)
        try:
            response = await self.client.post("responses", json=body, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            record.update(response_id=data.get("id"), model=data.get("model"), status=data.get("status"),
                          response_output=data.get("output", []), incomplete_details=data.get("incomplete_details"))
            usage = data.get("usage")
            if usage is not None:
                record["api_usage"] = usage
                # Each API response reports this request, unlike Codex's cumulative thread total.
                record["usage"] = {"total": {"input_tokens": usage["input_tokens"],
                    "cached_input_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens", 0),
                    "output_tokens": usage["output_tokens"], "total_tokens": usage["total_tokens"]}}
            if data.get("status") != "completed":
                raise RuntimeError(f"API response did not complete: {data.get('status')}")
            output = data.get("output", [])
            if any(item["type"] not in ("message", "reasoning", "function_call") for item in output):
                raise RuntimeError("Unexpected API tool activity")
            raw = "".join(part["text"] for item in output if item["type"] == "message"
                          for part in item.get("content", []) if part["type"] == "output_text")
            calls = [item for item in output if item["type"] == "function_call"]
            if not raw and not calls:
                raise RuntimeError("API returned no answer text (possibly a refusal)")
            # Preserve reasoning items and assistant outputs for the next tool observation.
            self._history.extend(output)
            self._pending = calls
            self._cursor = len(messages)
            if calls:
                reply = Reply(text=raw, tool_calls=[ToolCall(name=c["name"], arguments_json=c["arguments"]) for c in calls])
            elif tools:
                answer = Answer.model_validate_json(raw)
                reply = Reply(text=answer.answer, sources=answer.sources, tool_calls=[])
            else:
                # Tool-free consumers (e.g. the judge) receive their requested text directly.
                reply = Reply(text=raw, tool_calls=[])
            record["output"] = reply.model_dump_json()
            return record["output"]
        except BaseException as error:
            record.update(status="failed", error_type=type(error).__name__, error=str(error))
            raise
