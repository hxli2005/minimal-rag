"""Choose a transport explicitly; never fall back between subscription and API billing."""
from contextlib import asynccontextmanager
import os

from minimal_rag import codex_adapter as rag


class TokenBudgetExceeded(RuntimeError):
    pass


class TokenBudget:
    """Optional per-question stop, checked between requests; one request may exceed it."""
    def __init__(self, model, limit):
        self.model, self.limit, self.runs = model, limit, model.runs

    async def __call__(self, messages, tools):
        usage = rag.token_usage(self.runs)
        if usage.get("input_tokens", 0) + usage.get("output_tokens", 0) >= self.limit:
            raise TokenBudgetExceeded("Per-question token budget exhausted")
        return await self.model(messages, tools)


@asynccontextmanager
async def model_session(backend, model, timeout, base_url=None, max_total_tokens=None, verbosity=None):
    def wrap(instance):
        return TokenBudget(instance, max_total_tokens) if max_total_tokens else instance

    if backend == "api":
        import httpx
        from minimal_rag.api_adapter import APIModel
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("Set OPENAI_API_KEY for --backend api")
        endpoint = base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        async with httpx.AsyncClient(base_url=endpoint.rstrip("/") + "/",
                                    headers={"Authorization": f"Bearer {key}"}) as client:
            def factory(cwd, effort="high", system_prompt=rag.SYSTEM_PROMPT, verbosity=verbosity or "low"):
                return wrap(APIModel(client, model, timeout, effort, system_prompt, verbosity))
            yield factory
    elif backend == "codex":
        if verbosity is not None:
            raise ValueError("verbosity overrides are supported only by the API backend")
        from openai_codex import AsyncCodex
        async with AsyncCodex(rag.rag_codex_config("/private/tmp")) as client:
            account = (await client.account(refresh_token=False)).account
            if getattr(getattr(account, "root", account), "type", None) != "chatgpt":
                raise RuntimeError("Requires existing ChatGPT login; no API fallback")
            def factory(cwd, effort="high", system_prompt=rag.SYSTEM_PROMPT, verbosity=None):
                if verbosity is not None:
                    raise ValueError("verbosity overrides are supported only by the API backend")
                # The outer loop owns stopping; no forced final answer at a call boundary.
                return wrap(rag.CodexModel(client, cwd, model, timeout, effort=effort,
                                           system_prompt=system_prompt))
            yield factory
    else:
        raise ValueError(f"Unknown backend: {backend}")
