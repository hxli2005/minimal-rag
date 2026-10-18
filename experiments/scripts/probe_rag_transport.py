"""Small live subscription probe: identical request, different Codex context.

Not an accuracy benchmark. Saves each attempt immediately, including failures.
"""
import asyncio
import json
import tempfile
import time
from pathlib import Path

from openai_codex import AsyncCodex, ApprovalMode, CodexConfig, Sandbox
from openai_codex.generated.v2_all import ReasoningEffort
from minimal_rag.codex_adapter import SYSTEM_PROMPT
from minimal_rag.loop import Reply
from minimal_rag.indexed_rag import indexed_tools
from minimal_rag.demo import fixture_index

OUT = Path(__file__).resolve().parents[2] / "data/benchmarks/optimization-20261011/transport"
PROTOCOL = ("只输出一个JSON对象：text为回答，tool_calls为工具请求，sources为引用ID。"
            "本次只返回text为OK、tool_calls和sources为空数组。不要使用Codex内置工具。")
LEAN_CONFIG = {
    "include_apps_instructions": False,
    "include_collaboration_mode_instructions": False,
    "include_environment_context": False,
    "include_permissions_instructions": False,
    "skills.include_instructions": False,
    "project_doc_max_bytes": 0,
    "web_search": "disabled",
    "personality": "none",
    **{f"features.{key}": False for key in (
        "shell_tool", "apply_patch_freeform", "apps", "plugins", "multi_agent",
        "goals", "memories", "memory_tool", "view_image", "code_mode",
        "image_generation", "imagegenext", "computer_use", "browser_use",
        "hooks", "skill_search", "tool_suggest", "workspace_dependencies",
    )},
}


async def main():
    path = OUT / "prompt-overhead.json"
    if path.exists():
        raise FileExistsError(path)
    with fixture_index() as kb:
        specs = [{"name": name, "description": t.description,
              "parameters": t.arguments.model_json_schema()}
             for name, t in indexed_tools(kb).items()]
    payload = json.dumps({"messages": [{"role": "user", "content": "Return OK."}],
                          "tools": specs, "as_of": "2026-10-11"}, ensure_ascii=False)
    schema = Reply.model_json_schema()
    schema["required"] = list(schema["properties"])
    report = {"model": "gpt-6.1-sol", "effort": "medium", "input": payload,
              "instructions": SYSTEM_PROMPT + PROTOCOL, "lean_config": LEAN_CONFIG,
              "note": "One identical short request per arm; overhead diagnostic, not quality evidence.", "arms": []}
    with tempfile.TemporaryDirectory(prefix="rag-transport-probe-") as cwd:
        async with AsyncCodex(CodexConfig(cwd=cwd)) as client:
            wrapper = await client.account(refresh_token=False)
            if getattr(getattr(wrapper.account, "root", wrapper.account), "type", None) != "chatgpt":
                raise RuntimeError("existing ChatGPT subscription required")
            for mode in ("default", "base_only", "lean"):
                row = {"mode": mode}
                report["arms"].append(row)
                begin = time.monotonic()
                try:
                    opts = {"developer_instructions": SYSTEM_PROMPT + PROTOCOL} if mode == "default" else {
                        "base_instructions": SYSTEM_PROMPT + PROTOCOL, "developer_instructions": ""}
                    if mode == "lean":
                        opts["config"] = LEAN_CONFIG
                    thread = await client.thread_start(model=report["model"], cwd=cwd, ephemeral=True,
                        sandbox=Sandbox.read_only, approval_mode=ApprovalMode.deny_all, **opts)
                    turn = await thread.turn(payload, effort=ReasoningEffort.medium, output_schema=schema)
                    try:
                        result = await asyncio.wait_for(turn.run(), 90)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        await asyncio.shield(turn.interrupt())
                        raise
                    row.update(thread_id=thread.id, turn_id=turn.id, output=result.final_response,
                               usage=result.usage.model_dump(mode="json") if result.usage else None,
                               item_types=[x.model_dump(mode="json", by_alias=True).get("type") for x in result.items])
                except Exception as error:
                    row["error"] = {"type": type(error).__name__, "detail": str(error)}
                finally:
                    row["seconds"] = time.monotonic() - begin
                    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                    print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
