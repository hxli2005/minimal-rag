"""Dataset adapters and scoring shared by current and historical runs."""
import json
import statistics
from collections import Counter
from minimal_rag import codex_adapter as rag
from minimal_rag.paths import ROOT
from minimal_rag_experiments.hotpot_benchmark import FORMAT_REQUEST, exposure, parse_prediction

BASE = ROOT / "data/benchmarks"
NATIVE_FORMAT = ('Use the supplied corpus. Put the final answer directly in the outer text field, and supporting '
                 'document IDs in the outer sources array. Do not put another JSON object inside text.\nQuestion: ')
NATIVE_SHORT = ('Use the supplied passages. Put only the minimal answer phrase that directly answers the question in the outer text field, '
                'and supporting document IDs in the outer sources array. Use original English names. '
                'Omit explanations, citations, sentence framing and unnecessary descriptive modifiers from text. '
                'Use yes/no when appropriate, or empty text if evidence is insufficient. Do not nest JSON inside text.\nQuestion: ')
EXTRACTIVE_SHORT = ('Answer the question from the supplied passages. Put only the shortest sufficient answer in the outer text field, '
                    'and supporting document IDs in sources. When an answer is stated in the text, copy its wording verbatim: '
                    'preserve name spellings, punctuation and units. Prefer the full name of an entity over a nickname. '
                    'For a comparison or calculation, give only the requested result. Omit explanations and citation markers from text. '
                    'Use yes/no when appropriate, or empty text if essential evidence is unavailable.\nQuestion: ')
FORMAT = ('Use the supplied corpus to answer the question. Your final text must be JSON: '
          '{"answer":"your answer, with necessary explanation and calculations","sources":["actual document ID"]}. '
          'Include only source IDs that support the answer. If the corpus does not contain the answer, explicitly say so. '
          'Do not use outside sources.\nQuestion: ')


def normalized_case(dataset, row):
    if dataset == "hotpotqa":
        return {"id": row["_id"], "question": row["question"], "category": row["type"],
                "gold": row["answer"], "facts": [row["answer"]],
                "gold_ids": list({f"D{i}" for i, (title, _) in enumerate(row["context"]) if title in {t for t, _ in row["supporting_facts"]}}),
                "raw": row}
    if dataset == "financebench":
        return {"id": str(row["financebench_id"]), "question": row["question"], "category": row["question_type"],
                "gold": row["answer"], "facts": [row["answer"]],
                "gold_ids": list({f"{e.get('evidence_doc_name', e.get('doc_name', row['doc_name']))}:p{e['evidence_page_num']}" for e in row["evidence"]})}
    return {"id": row["question_id"], "question": row["question"], "category": row["question_type"],
            "gold": row["gold_answer"], "facts": row["answer_facts"], "gold_ids": row["expected_doc_ids"]}


def retrieved_ids(report):
    ids = set()
    for event in report.get("events", []):
        if event.get("role") == "tool":
            content = event["content"]
            for row in content.get("hits", [content]):
                if row.get("document_id") and row.get("text"):
                    ids.add(row["document_id"])
    return ids


def score(case, report, dataset, scorer):
    answer, sources, error = "", [], None
    if report.get("result", {}).get("status") == "finished":
        try:
            if report.get("native_answer"):
                answer, sources = report["result"]["answer"].strip(), report["result"]["sources"]
            else:
                answer, sources = parse_prediction(report["result"]["answer"])
        except Exception as exc:
            error = type(exc).__name__
    else:
        error = report.get("result", {}).get("status", "unfinished")
    gold, seen = set(case["gold_ids"]), retrieved_ids(report)
    alias_path = BASE / dataset / "duplicate-id-aliases.json"
    aliases = json.loads(alias_path.read_text()) if alias_path.exists() else {}
    canonical_seen = {aliases.get(doc_id, doc_id) for doc_id in seen}
    canonical_sources = {aliases.get(doc_id, doc_id) for doc_id in sources}
    usage = rag.token_usage(report.get("model_runs", []))
    tool_errors = [{"tool": e.get("name"), "error": e["content"]["error"]}
                   for e in report.get("events", []) if e.get("role") == "tool"
                   and isinstance(e.get("content"), dict) and e["content"].get("error")]
    row = {"id": case["id"], "category": case["category"], "prediction": answer, "sources": sources,
           "status": report.get("result", {}).get("status"), "format_or_runtime_error": error,
           "tool_errors": tool_errors,
           "protocol_errors": sum(e.get("role") == "protocol_error" for e in report.get("events", [])),
           "seconds": report.get("seconds", 0), "model_calls": len(report.get("model_runs", [])),
           "tool_calls": sum(e.get("role") == "tool" for e in report.get("events", [])),
           "tokens": dict(usage), "gold_count": len(gold),
           "document_recall": len(gold & canonical_sources) / len(gold) if gold else None,
           "retrieved_document_recall": len(gold & canonical_seen) / len(gold) if gold else None,
           "all_gold_retrieved": gold <= canonical_seen if gold else None,
           "unseen_citations": sorted(set(sources) - seen)}
    if dataset == "hotpotqa":
        row.update({"em": float(scorer.exact_match_score(answer, case["gold"])),
                    "f1": scorer.f1_score(answer, case["gold"])[0], **exposure(case["raw"], report.get("events", []))})
    else:
        judgement = report.get("judgement", {})
        row["graded"] = "correct" in judgement
        row["correct"] = float(judgement["correct"]) if "correct" in judgement else None
        values = judgement.get("facts_supported", [])
        row["completeness"] = statistics.mean(values) if values else (1.0 if row["graded"] else None)
        row["overall"] = row["correct"] * row["completeness"] if row["graded"] else None
    return row


def aggregate(rows):
    result = {"n": len(rows), "status": dict(Counter(r["status"] for r in rows)),
              "errors": sum(bool(r["format_or_runtime_error"]) for r in rows),
              "tool_errors": sum(len(r.get("tool_errors", [])) for r in rows),
              "cases_with_tool_errors": sum(bool(r.get("tool_errors")) for r in rows),
              "protocol_errors": sum(r.get("protocol_errors", 0) for r in rows),
              "unseen_citations": sum(bool(r["unseen_citations"]) for r in rows)}
    for key in ("em", "f1", "correct", "completeness", "overall", "document_recall", "retrieved_document_recall",
                "all_gold_retrieved", "seconds", "model_calls", "tool_calls", "all_gold_sentences_exposed"):
        values = [r[key] for r in rows if r.get(key) is not None]
        if values:
            result[key] = statistics.mean(values)
    result["graded"] = sum(r.get("graded", "em" in r) for r in rows)
    for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
        result[key] = sum(r["tokens"].get(key, 0) for r in rows)
    return result


def validate_evaluation_scope(split, final_holdout, known_regression, limit, ids):
    if final_holdout and known_regression:
        raise ValueError("a cohort cannot be both untouched holdout and known regression")
    if split == "test" and not (final_holdout or known_regression):
        raise ValueError("test requires --final-holdout or --known-regression after freezing a candidate")
    if final_holdout or known_regression:
        if split != "test" or limit or ids:
            raise ValueError("frozen evaluation must use the complete test cohort")

