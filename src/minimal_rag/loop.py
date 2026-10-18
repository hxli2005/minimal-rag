"""The complete model → tool → observation loop and its wire protocol."""
import inspect
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


@dataclass
class Tool:
    description: str
    arguments: type[BaseModel]
    execute: Callable


class ToolCall(StrictModel):
    name: str
    arguments_json: str


class Reply(StrictModel):
    text: str
    tool_calls: list[ToolCall] = Field(max_length=16)
    sources: list[str] = Field(default_factory=list, max_length=100)


async def run_agent(complete, tools, question, max_turns=12, emit=lambda _: None):
    """complete(messages, tools)返回Reply的JSON；循环只传消息、执行工具。"""
    if not question.strip() or max_turns < 1:
        raise ValueError("question must be nonempty and max_turns positive")
    messages = [{"role": "user", "content": question}]
    specs = [{"name": name, "description": t.description, "parameters": t.arguments.model_json_schema()}
             for name, t in tools.items()]
    for _ in range(max_turns):
        raw = await complete(messages, specs)
        try:
            reply = Reply.model_validate_json(raw)
        except ValidationError as error:
            event = {"role": "protocol_error", "raw": raw, "error": str(error)}
            messages.append(event)
            emit(event)
            continue
        event = {"role": "assistant", **reply.model_dump()}
        messages.append(event)
        emit(event)
        if not reply.tool_calls:
            return {"status": "finished", "answer": reply.text, "sources": reply.sources}
        for call in reply.tool_calls:
            try:
                tool = tools[call.name]
                args = tool.arguments.model_validate_json(call.arguments_json)
                result = tool.execute(**args.model_dump())
                if inspect.isawaitable(result):
                    result = await result
            except Exception as error:
                result = {"error": type(error).__name__, "message": str(error)}
            event = {"role": "tool", **call.model_dump(), "content": result}
            messages.append(event)
            emit(event)
    return {"status": "budget_exhausted", "answer": "运行预算已用尽；模型尚未结束回答。"}


