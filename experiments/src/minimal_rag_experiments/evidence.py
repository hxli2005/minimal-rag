"""Investigation-only evidence receipts for actually observed original text.

Receipts prove provenance of a passage, not whether it entails a model's claim.
The reader must enforce the same authorization as the wrapped search/read tools.
"""
from __future__ import annotations

import inspect
import time
from dataclasses import asdict, dataclass, field
from threading import Lock

from minimal_rag.loop import Tool


class EvidenceError(ValueError):
    pass


@dataclass(frozen=True)
class Passage:
    id: str
    document_id: str
    title: str
    source: str
    offset: int
    text: str


class EvidenceLedger:
    def __init__(self, reader):
        self.reader = reader
        self.passages: dict[str, Passage] = {}
        self._identifiers: dict[tuple, str] = {}
        self.timings: list[dict] = []

    def capture(self, row):
        if not row.get("text") or not row.get("document_id"):
            return dict(row)
        offset, text = row.get("offset", 0), row["text"]
        original = self.reader(row["document_id"], offset=offset, limit=len(text))
        if (original.get("error") or original.get("text") != text
                or original.get("title") != row.get("title")
                or original.get("source") != row.get("source")):
            raise EvidenceError("tool passage differs from the authorized source at its stated offset")
        identity = {"document_id": row["document_id"], "title": row["title"],
                    "source": row["source"], "offset": offset, "text": text}
        key = tuple(identity.values())
        if key not in self._identifiers:
            ident = f"E{len(self.passages) + 1}"
            self.passages[ident] = Passage(id=ident, **identity)
            self._identifiers[key] = ident
        return {**row, "evidence_id": self._identifiers[key]}

    def wrap(self, tools):
        def wrapped(name, tool):
            async def execute(**arguments):
                started = time.monotonic()
                try:
                    result = tool.execute(**arguments)
                    if inspect.isawaitable(result):
                        result = await result
                finally:
                    self.timings.append({"tool": name, "seconds": time.monotonic() - started})
                # Publish receipts only if the entire response reaches the model.
                # A later bad hit must not leave earlier, unobserved receipts valid.
                staged = EvidenceLedger(self.reader)
                staged.passages = dict(self.passages)
                staged._identifiers = dict(self._identifiers)
                if "hits" in result:
                    output = {**result, "hits": [staged.capture(row) for row in result["hits"]]}
                else:
                    output = staged.capture(result)
                self.passages, self._identifiers = staged.passages, staged._identifiers
                return output
            return Tool(tool.description + " Each visible text passage has an evidence_id; cite it as [E1].",
                        tool.arguments, execute)
        return {name: wrapped(name, tool) for name, tool in tools.items()}

    def check(self, identifiers):
        errors = []
        for ident in sorted(set(identifiers)):
            passage = self.passages.get(ident)
            if passage is None:
                errors.append(f"{ident}: evidence was not observed in this investigation")
                continue
            try:
                current = self.reader(passage.document_id, offset=passage.offset, limit=len(passage.text))
            except Exception:
                errors.append(f"{ident}: source could not be checked")
                continue
            if (current.get("error") or current.get("text") != passage.text
                    or current.get("title") != passage.title or current.get("source") != passage.source):
                errors.append(f"{ident}: source passage changed or is no longer readable")
        return errors

    def export(self):
        return [asdict(passage) for passage in self.passages.values()]


class CallLimitReached(RuntimeError):
    pass


@dataclass
class CallBudget:
    """Share across workers/adapters to cap attempted outer provider calls.

    Reservation happens before dispatch; failures and retries consume a slot.
    This is not a token, subscription-quota or provider-internal-retry limit.
    """
    limit: int
    used: int = field(default=0, init=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self):
        if type(self.limit) is not int or self.limit < 1:
            raise ValueError("call limit must be a positive integer")

    def reserve(self):
        with self._lock:
            if self.used >= self.limit:
                raise CallLimitReached("shared outer model call budget exhausted")
            self.used += 1

    def snapshot(self):
        with self._lock:
            return {"limit": self.limit, "used": self.used, "remaining": self.limit - self.used,
                    "unit": "outer_provider_calls"}
