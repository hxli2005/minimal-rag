"""Offline official-format export of the frozen historical Enterprise 500 answers."""
import hashlib
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/benchmarks"
HISTORY = BASE / "optimization-20261011"


def read(path):
    return json.loads(path.read_text())


def fingerprint(path):
    with path.open("rb") as stream:
        return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


HISTORICAL = {"folders": (("dev", HISTORY / "runs/enterprise-dev100-hybrid-minilm-wide"), ("test", HISTORY / "holdout/enterprise")),
              "output": ROOT / "runs/evidence/historical-enterprise-500-answers.jsonl",
              "evidence": ROOT / "evidence/official-history-export-20261013.json",
              "meta": {"kind": "offline_historical_official_format_export", "date": "2026-10-13",
                       "generation_model": "gpt-6-astra (historical, no new generation)", "variant": "v6"}}


def main(folders=HISTORICAL["folders"], output=HISTORICAL["output"], evidence_path=HISTORICAL["evidence"],
         meta=HISTORICAL["meta"]):
    aliases = read(BASE / "enterprise/duplicate-id-aliases.json")
    answers, cases, sources = [], [], []
    for split, folder in folders:
        questions = BASE / f"enterprise/{split}.json"
        config = folder / "config.json" if (folder / "config.json").exists() else folder.parent / "config.json"
        sources += [fingerprint(questions), fingerprint(config), fingerprint(folder / "summary.json")]
        for case in read(questions):
            trace = folder / "traces" / f"{case['question_id']}.json"
            report = read(trace) if trace.exists() else {}
            result = report.get("result", {})
            finished = result.get("status") == "finished"
            if finished and not report.get("native_answer"):
                raise ValueError(f"Non-native answer needs explicit conversion: {trace}")
            local_ids = result.get("sources", []) if finished else []
            answers.append({"question_id": case["question_id"], "answer": result.get("answer", "") if finished else "",
                            "document_ids": list(dict.fromkeys(aliases.get(doc, doc) for doc in local_ids))})
            seen = {hit.get("document_id") for event in report.get("events", []) if event.get("role") == "tool"
                    for hit in event["content"].get("hits", [event["content"]])}
            cases.append({"question_id": case["question_id"], "split": split, "generation_finished": finished,
                          "local_document_ids": local_ids, "unobserved_citations": sorted(set(local_ids) - seen),
                          "trace": fingerprint(trace) if trace.exists() else {"path": str(trace.relative_to(ROOT)), "missing": True}})
    ids = [row["question_id"] for row in answers]
    if len(ids) != 500 or len(set(ids)) != 500:
        raise ValueError("Expected exactly500 unique questions, including empty failed answers")
    cited = {doc for case in cases for doc in case["local_document_ids"]}
    versions, unknown = [], []
    # Publish the four version alternatives as well as actual citations; UUID alone is insufficient.
    with sqlite3.connect((BASE / "enterprise/corpus.sqlite").resolve().as_uri() + "?mode=ro", uri=True) as db:
        for doc in sorted(cited | set(aliases) | set(aliases.values())):
            row = db.execute("SELECT title,text,source FROM documents WHERE id=?", (doc,)).fetchone()
            if row is None:
                unknown.append(doc)
                continue
            title, text, source = row
            versions.append({"local_id": doc, "official_id": aliases.get(doc, doc), "title": title,
                             "source": source, "text_chars": len(text),
                             "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "cited": doc in cited})
    import pyarrow.parquet as pq
    parquet = BASE / "enterprise/source/data/documents/test.parquet"
    needed = {(v["official_id"], v["text_sha256"]): v for v in versions}
    requested_ids = {v["official_id"] for v in versions}
    position = 0
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=4096, columns=["doc_id", "content"], use_threads=False):
        for index, doc in enumerate(batch.column(0).to_pylist()):
            if doc not in requested_ids:
                continue
            text = batch.column(1)[index].as_py()
            match = needed.get((doc, hashlib.sha256(text.encode()).hexdigest()))
            if match is not None:
                match["parquet_row_zero_based"] = position + index
        position += batch.num_rows
    unbound = [v["local_id"] for v in versions if "parquet_row_zero_based" not in v]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in answers))
    evidence = {**meta, "model_calls": 0, "questions": 500, "cited_local_documents": len(cited),
                "cited_official_ids": len({aliases.get(doc,doc) for doc in cited}), "bound_document_versions": len(versions)-len(unbound),
                "finished": sum(c["generation_finished"] for c in cases),
                "missing_as_empty_answer": True, "unobserved_citation_cases": sum(bool(c["unobserved_citations"]) for c in cases),
                "unknown_citation_ids": unknown,
                "unbound_parquet_versions": unbound, "parquet": fingerprint(parquet),
                "alias_citation_cases": [c["question_id"] for c in cases if set(c["local_document_ids"]) & set(aliases)],
                "answers": fingerprint(output), "sources": sources + [fingerprint(BASE / "enterprise/duplicate-id-aliases.json")],
                "document_versions": versions, "cases": cases,
                "limits": ["No new score, official evaluation or submission; historical local scores unchanged.",
                           "Retains the exact native answer, including in-text citations. Upstream citation removal has not run.",
                           "Canonical export collapses four aliases; each mapped text is bound to its release-Parquet row, not an upstream Git JSON filename.",
                           "The upstream UUID traversal mapping must still be aligned before evaluation; this is the local corpus mapping.",
                           "All500 questions are now known regression questions, not an untouched holdout."]}
    from minimal_rag_experiments.evidence_summary import publish
    publish(evidence_path, evidence)
    print(json.dumps({k: v for k, v in evidence.items() if k not in ("document_versions", "cases", "sources")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
