"""User-authorized retries of key-query failures that sent zero model requests."""
import asyncio
import json
from pathlib import Path
import time

import httpx

import run as pilot


async def main():
    lock = (pilot.OUT / "runner.lock").open("a")
    pilot.fcntl.flock(lock.fileno(), pilot.fcntl.LOCK_EX | pilot.fcntl.LOCK_NB)
    archive = pilot.OUT / "key-query-retries"
    authorization = pilot.read(archive / "authorization.json")
    if not authorization:
        raise pilot.StopRun("User authorization for this recovery class is missing")
    for name, digest in pilot.read(pilot.OUT / "execution-config.json")["source_sha256"].items():
        if pilot.hashlib.sha256((pilot.ROOT / name).read_bytes()).hexdigest() != digest:
            raise pilot.StopRun("Started-run source changed; do not rerun under another configuration")
    headers = {"Authorization": "Bearer " + pilot.KEY.read_text().strip(), "X-OpenRouter-Metadata": "enabled",
               "X-OpenRouter-Title": "Minimal RAG fixed API validation"}
    from openai_codex import AsyncCodex
    adapter = pilot.frozen_codex()
    decisions = []
    async with httpx.AsyncClient(base_url="https://openrouter.ai/api/v1/", headers=headers) as client, AsyncCodex(adapter.rag_codex_config("/private/tmp")) as judge_client:
        guard = pilot.GuardedClient(client)
        await guard.key_state()
        candidates = []
        for path in sorted(pilot.OUT.glob("stage*/*/*.json")):
            row = pilot.read(path)
            result = row.get("result", {})
            if result.get("status") != "pending_retry" and result.get("error") not in ("ConnectTimeout", "ConnectError", "ReadTimeout", "RemoteProtocolError"):
                continue
            if any(r["stage"] == row["stage"] and r["id"] == row["id"] and r["arm"] == row["arm"] for r in guard.rows):
                continue
            if row.get("judge_attempts") or row.get("events"):
                raise pilot.StopRun("Do not retry a case with model activity or a valid judgement")
            candidates.append((path, row))
        for path, original in candidates:
            folder = archive / f"stage{original['stage']}-{original['id']}-{original['arm']}"
            folder.mkdir(exist_ok=True)
            if not (folder / "original-case.json").exists():
                pilot.save(folder / "original-case.json", original)
            # Read-only readiness retry is permitted; it cannot generate or resample an answer.
            for attempt in range(3):
                try:
                    pilot.check_before_request(guard.rows, await guard.key_state(), original["stage"])
                    break
                except httpx.TransportError:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2)
            attempt_number = len(list(folder.glob("pending-before-rerun-*.json"))) + 1
            path.rename(folder / f"pending-before-rerun-{attempt_number}.json")
            decision = {"time": time.time(), "stage": original["stage"], "id": original["id"], "arm": original["arm"],
                        "billable_requests_before_retry": 0, "valid_answer_resampled": False,
                        "source_sha256": pilot.hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "authorization": "key-query-retries/authorization.json", "status": "started"}
            pilot.save(folder / "decision.json", decision)
            pilot.save(folder / f"decision-{attempt_number}.json", decision)
            record = await pilot.generate(guard, original["stage"], original["id"], original["arm"])
            if record["result"]["status"] != "finished" and not any(r["stage"] == original["stage"] and r["id"] == original["id"] and r["arm"] == original["arm"] for r in guard.rows):
                record["result"]["status"] = "pending_retry"
                pilot.save(path, record)
                decision["status"] = "still_pending_without_model_call"
                pilot.save(folder / "decision.json", decision)
                pilot.save(folder / f"decision-{attempt_number}.json", decision)
                raise pilot.StopRun("Repeated metadata connection failure; kept pending, not incorrect")
            await pilot.judge(record, judge_client, adapter)
            decision.update(status="completed", result_status=record["result"]["status"], correct=record["judgement"]["correct"])
            pilot.save(folder / "decision.json", decision)
            pilot.save(folder / f"decision-{attempt_number}.json", decision)
            decisions.append(decision)
            pilot.overview()
    rows = [pilot.read(pilot.report_path(1, q, "low")) for q in pilot.SAMPLE["migration_ids"]]
    if all("judgement" in r for r in rows):
        correct = sum(r["judgement"]["correct"] for r in rows)
        finished = sum(r["result"]["status"] == "finished" for r in rows)
        pilot.save(pilot.OUT / "migration-gate.json", {"passed": correct >= 19 and finished >= 29,
            "correct": correct, "finished": finished, "n": 30, "baseline_correct": 22,
            "screen_not_equivalence": True, "amendment": "User-authorized replacement of key-query failures with first actual model runs; old records retained."})
    pilot.save(archive / "summary.json", {"decisions": decisions, "total_known_usd": sum(r["cost_usd"] or 0 for r in guard.rows)})
    print(json.dumps({"reruns_completed": len(decisions), "decisions": decisions}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
