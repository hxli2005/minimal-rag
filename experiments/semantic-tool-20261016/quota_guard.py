"""Use the user's authorized free reset credits only after included usage is blocked.

Uses the documented account/rateLimitResetCredit/consume API. No model calls,
credit purchases, account changes, or changes to benchmark generation settings.
"""
import argparse
import asyncio
import fcntl
import json
from pathlib import Path
import time
import uuid

from openai_codex.async_client import AsyncCodexClient
from openai_codex.generated.v2_all import (
    ConsumeAccountRateLimitResetCreditResponse, GetAccountRateLimitsResponse,
)
from minimal_rag.codex_adapter import rag_codex_config

OUT = Path(__file__).resolve().parents[2] / "runs/semantic-tool-20261016"


def save(state):
    temporary = OUT / "quota-state.json.tmp"
    temporary.write_text(json.dumps(state, indent=2)+"\n")
    temporary.replace(OUT / "quota-state.json")


async def main(watch):
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "quota-guard.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = OUT / "quota-state.json"
    state = json.loads(path.read_text()) if path.exists() else {"reset_events": []}
    until = time.time()+48*3600
    while time.time() < until and not (OUT / "workflow-complete.json").exists() and not (OUT / "STOP").exists():
        try:
            async with AsyncCodexClient(rag_codex_config("/private/tmp")) as client:
                await client.initialize()
                account = (await client.account_read({"refreshToken": False})).account
                if getattr(getattr(account, "root", account), "type", None) != "chatgpt":
                    raise RuntimeError("requires the existing ChatGPT account")
                while True:
                    raw = await client.request("account/rateLimits/read", {"excludeResetCreditDetails": True},
                                               response_model=GetAccountRateLimitsResponse)
                    data = raw.model_dump(by_alias=True, mode="json")
                    limits = data.get("rateLimitsByLimitId") or {"codex": data["rateLimits"]}
                    windows = [b[k] for b in limits.values() for k in ("primary", "secondary") if b.get(k)]
                    used = max((w["usedPercent"] for w in windows), default=None)
                    available = (data.get("rateLimitResetCredits") or {}).get("availableCount", 0)
                    allowed = data.get("ordinaryUsageAllowed")
                    state.update(checked_at=time.time(), used_percent=used, ordinary_usage_allowed=allowed,
                                 available_resets=available, status="watching" if watch else "read_only_check")
                    if allowed is True: state["awaiting_recovery"] = False
                    save(state)
                    if not watch:
                        print(json.dumps(state), flush=True); return
                    if time.time() >= until or (OUT / "workflow-complete.json").exists() or (OUT / "STOP").exists():
                        return
                    should_reset = allowed is False and available > 0 and not state.get("awaiting_recovery")
                    if should_reset or state.get("pending_idempotency_key"):
                        key = state.setdefault("pending_idempotency_key", str(uuid.uuid4())); save(state)
                        result = await client.request("account/rateLimitResetCredit/consume", {"idempotencyKey": key},
                                                      response_model=ConsumeAccountRateLimitResetCreditResponse)
                        outcome = result.model_dump(mode="json")["outcome"]
                        event = {"at": time.time(), "idempotency_key": key, "outcome": outcome}
                        state["reset_events"].append(event)
                        state.pop("pending_idempotency_key", None)
                        state["awaiting_recovery"] = True
                        save(state); print(json.dumps(event), flush=True)
                    await asyncio.sleep(5 if used is not None and used >= 95 else 60)
        except Exception as error:
            state.update(status="error", error=type(error).__name__, checked_at=time.time()); save(state)
            print(json.dumps({"quota_guard_error": type(error).__name__}), flush=True)
            if not watch: raise
            await asyncio.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    asyncio.run(main(parser.parse_args().watch))
