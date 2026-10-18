"""对冻结的极简RAG运行HotpotQA dev-distractor样本；评测逻辑不进入Agent内核。

运行示例：.venv/bin/python -m minimal_rag_experiments.hotpot_benchmark --sample PATH --output DIR
只向模型提供问题、格式要求，以及通过工具取得的候选段落；gold答案和支持标签仅用于评分。
官方Answer EM/F1；另外统计支持句原文的可见范围，不冒充官方Supporting Facts或Joint分数。
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import statistics
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import ModuleType
from zoneinfo import ZoneInfo

from minimal_rag import codex_adapter as rag
from minimal_rag.loop import run_agent
from minimal_rag.indexed_rag import IndexedKnowledgeBase, indexed_tools

from minimal_rag.paths import ROOT
SCORER = Path(__file__).with_name("_vendor") / "hotpot_evaluate_v1.py"
FORMAT_REQUEST = (
    'Answer this benchmark question using the supplied knowledge tools and passages. '
    'Your final text must be one JSON object: {"answer": "short answer only", "sources": ["D0"]}. '
    'Replace D0 with the actual source IDs you used. Use the original English names, '
    'or yes/no for a yes/no question. Do not put explanations or citations inside answer. '
    'If the available evidence is insufficient, use an empty answer string.\nQuestion: '
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_scorer(path=SCORER):
    """已人工检查官方120行脚本；只把可选ujson导入替换成stdlib json。"""
    module = ModuleType("hotpot_official")
    code = Path(path).read_text().replace("import ujson as json", "import json")
    exec(compile(code, str(path), "exec"), module.__dict__)
    return module


class HotpotKnowledgeBase(IndexedKnowledgeBase):
    """Per-question SQLite corpus; original passages only, no gold labels."""
    def __init__(self, question_id, context):
        self.question_id = question_id
        self.titles = {f"D{i}": title for i, (title, _) in enumerate(context)}
        self.temporary = tempfile.TemporaryDirectory(prefix="hotpot-corpus-")
        path = Path(self.temporary.name) / "corpus.sqlite"
        self.build(path, [{"id": f"D{i}", "title": title, "text": "".join(sentences),
                          "source": f"hotpotqa://dev-distractor/{question_id}/D{i}"}
                         for i, (title, sentences) in enumerate(context)])
        super().__init__(path, excerpt_chars=240)

    def close(self):
        super().close()
        self.temporary.cleanup()


def parse_prediction(text):
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        value = value.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    data = json.loads(value)
    if not isinstance(data, dict) or not isinstance(data.get("answer"), str):
        raise ValueError("final JSON must contain a string answer")
    sources = data.get("sources", [])
    if not isinstance(sources, list) or any(not isinstance(x, str) for x in sources):
        raise ValueError("sources must be a list of source IDs")
    return data["answer"].strip(), list(dict.fromkeys(sources))


def covered(start, end, intervals):
    cursor = start
    for lo, hi in sorted(intervals):
        if lo > cursor:
            break
        cursor = max(cursor, hi)
    return cursor >= end


def exposure(case, events):
    """严格依据工具实际返回的字符区间；出现文档标题不代表看到完整支持句。"""
    titles = {f"D{i}": title for i, (title, _) in enumerate(case["context"])}
    ranges = {}
    for event in events:
        if event.get("role") != "tool":
            continue
        content = event.get("content", {})
        rows = content.get("hits", []) if event.get("name") == "search" else [content]
        for row in rows:
            doc_id, text = row.get("document_id"), row.get("text", "")
            if doc_id in titles and text:
                start = row.get("offset", 0)
                ranges.setdefault(titles[doc_id], []).append((start, start + len(text)))
    gold = {tuple(pair) for pair in case["supporting_facts"]}
    gold_titles = {title for title, _ in gold}
    paragraphs = dict(case["context"])
    seen, invalid = [], []
    for title, index in sorted(gold):
        sentences = paragraphs.get(title, [])
        if not 0 <= index < len(sentences):
            invalid.append([title, index])
            continue
        sentence = sentences[index]
        start = sum(map(len, sentences[:index]))
        left = start + len(sentence) - len(sentence.lstrip())
        right = start + len(sentence.rstrip())
        if sentence.strip() and covered(left, right, ranges.get(title, [])):
            seen.append([title, index])
    return {
        "gold_document_exposure_recall": len(gold_titles & ranges.keys()) / len(gold_titles) if gold_titles else 0,
        "all_gold_documents_exposed": bool(gold_titles) and gold_titles <= ranges.keys(),
        "gold_sentence_exposure_recall": len(seen) / len(gold) if gold else 0,
        "all_gold_sentences_exposed": bool(gold) and len(seen) == len(gold),
        "exposed_titles": sorted(ranges), "visible_ranges": ranges,
        "exposed_gold_sentences": seen, "invalid_gold_annotations": invalid,
    }


def score_case(case, report, scorer):
    prediction, sources, format_error = "", [], None
    result = report.get("result", {})
    if result.get("status") == "finished":
        try:
            prediction, sources = parse_prediction(result.get("answer", ""))
        except (ValueError, IndexError) as error:
            format_error = str(error)
    else:
        format_error = result.get("status", "missing_result")
    f1, precision, recall = scorer.f1_score(prediction, case["answer"])
    usage = rag.token_usage(report.get("model_runs", []))
    return {"id": case["_id"], "question": case["question"], "type": case["type"], "level": case["level"],
            "prediction": prediction, "gold_answer": case["answer"], "sources": sources,
            "em": float(scorer.exact_match_score(prediction, case["answer"])),
            "f1": f1, "precision": precision, "recall": recall, "format_error": format_error,
            "status": result.get("status"), "seconds": report.get("seconds", 0),
            "model_calls_attempted": report.get("model_calls_attempted", 0),
            "model_runs_completed": sum(r.get("status") == "completed" for r in report.get("model_runs", [])),
            "sdk_reported_tokens": dict(usage),
            "tool_calls": sum(e.get("role") == "tool" for e in report.get("events", [])),
            **exposure(case, report.get("events", []))}


def percentile(values, quantile):
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def aggregate(rows):
    def mean(key):
        return statistics.mean(row[key] for row in rows)
    n = len(rows)
    p, z = mean("em"), 1.96
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return {"n": n, "answer_em": p, "answer_f1": mean("f1"),
            "em_wilson_95_interval": [center - half, center + half],
            "all_gold_documents_exposed_rate": mean("all_gold_documents_exposed"),
            "gold_sentence_exposure_recall": mean("gold_sentence_exposure_recall"),
            "all_gold_sentences_exposed_rate": mean("all_gold_sentences_exposed"),
            "mean_model_calls": mean("model_calls_attempted"), "mean_tool_calls": mean("tool_calls"),
            "total_sdk_input_tokens": sum(r["sdk_reported_tokens"].get("input_tokens", 0) for r in rows),
            "total_sdk_cached_input_tokens": sum(r["sdk_reported_tokens"].get("cached_input_tokens", 0) for r in rows),
            "total_sdk_output_tokens": sum(r["sdk_reported_tokens"].get("output_tokens", 0) for r in rows),
            "mean_seconds": mean("seconds"), "median_seconds": percentile([r["seconds"] for r in rows], 0.5),
            "p95_seconds": percentile([r["seconds"] for r in rows], 0.95),
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "format_or_runtime_failures": sum(r["format_error"] is not None for r in rows)}


def write_summary(cases, output, config, scorer, elapsed=None):
    rows = []
    predictions = {"answer": {}}
    for case in cases:
        path = output / "traces" / f"{case['_id']}.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        if "result" not in report:
            continue
        row = score_case(case, report, scorer)
        rows.append(row)
        predictions["answer"][case["_id"]] = row["prediction"]
    if rows:
        summary = {"config": config, "completed": len(rows), "requested": len(cases),
                   "wall_seconds": elapsed, "metrics": aggregate(rows),
                   "by_type": {kind: aggregate([r for r in rows if r["type"] == kind]) for kind in sorted({r["type"] for r in rows})},
                   "cases": rows, "limits": [
                       "Fixed random subset of public dev-distractor; not full development set or test leaderboard.",
                       "Questions use the supplied distractor paragraphs (usually ten; original shorter cases are retained); this is not full-Wikipedia retrieval.",
                       "Only official answer EM/F1 are reported; sentence exposure is a custom trace diagnostic, not official supporting-fact or joint scoring.",
                       "Public benchmark contamination or model prior knowledge cannot be ruled out; no no-retrieval baseline was run.",
                       "Times and call counts include the Codex SDK harness; they are not bare-model API measurements.",
                       "All completed attempts, including model/runtime/format failures, stay in the denominator; no best-of-N selection.",
                   ]}
        save(output / "summary.json", summary)
        save(output / "predictions.json", predictions)
    return rows


async def main(args):
    from openai_codex import AsyncCodex
    import importlib.metadata
    sample, output = Path(args.sample), Path(args.output)
    cases = json.loads(sample.read_text())
    if not cases or len({c["_id"] for c in cases}) != len(cases):
        raise ValueError("sample must contain distinct cases")
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists():
        raise ValueError("choose a new output directory; benchmark attempts are not overwritten")
    implementation = Path(rag.__file__)
    config = {"dataset": "HotpotQA", "split": "dev-distractor", "sample": str(sample.resolve()),
              "sample_sha256": sha256(sample), "n": len(cases), "model": args.model,
              "candidate_count_distribution": dict(Counter(len(c["context"]) for c in cases)),
              "reasoning_effort": "medium", "max_turns": args.max_turns, "turn_timeout": args.timeout,
              "case_timeout": args.case_timeout, "concurrency": args.concurrency,
              "implementation": str(implementation), "implementation_sha256": sha256(implementation),
              "runner_sha256": sha256(__file__), "scorer_sha256": sha256(SCORER),
              "scorer_source": "https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py",
              "scorer_import_substitution": "ujson -> stdlib json only; scoring functions unchanged",
              "format_request": FORMAT_REQUEST, "sdk_version": importlib.metadata.version("openai-codex"),
              "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}
    save(config_path, config)
    scorer, gate, started = load_scorer(), asyncio.Semaphore(args.concurrency), time.monotonic()
    completed = 0
    async with AsyncCodex(rag.rag_codex_config("/private/tmp")) as client:
        account_wrapper = await client.account(refresh_token=False)
        account = getattr(account_wrapper.account, "root", account_wrapper.account)
        if getattr(account, "type", None) != "chatgpt":
            raise RuntimeError("benchmark requires existing ChatGPT login; no API-key fallback")

        async def run_one(case):
            nonlocal completed
            async with gate:
                if sha256(implementation) != config["implementation_sha256"]:
                    raise RuntimeError("RAG implementation changed during benchmark")
                path = output / "traces" / f"{case['_id']}.json"
                report = {"id": case["_id"], "question": case["question"], "events": [],
                          "model_calls_attempted": 0, "source": "live_codex_hotpotqa"}
                begin = time.monotonic()
                kb = HotpotKnowledgeBase(case["_id"], case["context"])
                report["document_titles"] = kb.titles

                def emit(event):
                    report["events"].append(event)
                    save(path, report)

                try:
                    with tempfile.TemporaryDirectory(prefix="hotpot-rag-") as cwd:
                        model = rag.CodexModel(client, cwd, args.model, args.timeout)
                        report["model_runs"] = model.runs

                        async def complete(messages, tools):
                            report["model_calls_attempted"] += 1
                            return await model(messages, tools)

                        # 只传question；answer/supporting_facts均不进入模型或工具定义。
                        report["result"] = await asyncio.wait_for(
                            run_agent(complete, indexed_tools(kb), FORMAT_REQUEST + case["question"], args.max_turns, emit),
                            timeout=args.case_timeout)
                except Exception as error:
                    report["result"] = {"status": "runtime_error", "answer": "", "error": type(error).__name__, "detail": str(error)}
                finally:
                    report["seconds"] = time.monotonic() - begin
                    save(path, report)
                completed += 1
                row = score_case(case, report, scorer)
                print(json.dumps({"completed": completed, "total": len(cases), "id": case["_id"],
                                  "status": row["status"], "em": row["em"], "f1": round(row["f1"], 3),
                                  "model_calls": row["model_calls_attempted"], "seconds": round(row["seconds"], 1)}, ensure_ascii=False), flush=True)
                write_summary(cases, output, config, scorer, time.monotonic() - started)

        await asyncio.gather(*(run_one(case) for case in cases))
    assert sha256(implementation) == config["implementation_sha256"]
    rows = write_summary(cases, output, config, scorer, time.monotonic() - started)
    assert len(rows) == len(cases)
    print(json.dumps({"summary": str(output / 'summary.json'), "metrics": aggregate(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--case-timeout", type=float, default=300)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()
    if min(args.max_turns, args.timeout, args.case_timeout, args.concurrency) <= 0:
        parser.error("budgets and concurrency must be positive")
    asyncio.run(main(args))
