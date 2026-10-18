"""One-night experiment budget; reserve every attempted request before transport."""
import fcntl
import json
import os
import time
from pathlib import Path

from minimal_rag.codex_adapter import CodexModel

DIRECTORY = Path(__file__).resolve().parent
LEDGER = DIRECTORY / "calls.jsonl"
LIMIT = 500
MODEL = "gpt-6.1-sol"


def reserve(label, model=MODEL):
    if model != MODEL:
        raise ValueError("Tonight permits only gpt-6.1-sol")
    with LEDGER.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        rows = [json.loads(line) for line in stream if line.strip()]
        if len(rows) >= LIMIT:
            raise RuntimeError("Shared 500-call overnight budget exhausted")
        record = {"call": len(rows) + 1, "model": model, "label": label,
                  "reserved_unix": time.time(), "pid": os.getpid()}
        stream.seek(0, os.SEEK_END)
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        return record


class BudgetModel(CodexModel):
    def __init__(self, *args, label, **kwargs):
        kwargs["model"] = MODEL
        super().__init__(*args, **kwargs)
        self.label = label

    async def __call__(self, messages, tools):
        # Local per-conversation exhaustion is not a request and is not charged.
        if self.max_calls is not None and messages is self._messages and self._calls >= self.max_calls:
            raise RuntimeError("Conversation call budget exhausted")
        reservation = reserve(self.label, self.model)
        try:
            return await super().__call__(messages, tools)
        finally:
            if self.runs:
                self.runs[-1]["overnight_call"] = reservation["call"]
