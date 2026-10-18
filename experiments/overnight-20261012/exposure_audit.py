"""Audit original-source character intervals for 70 historical failed answers."""
import json
import sqlite3
import statistics
from collections import Counter

from worker import BASE, OUTPUT, ROOT, read, save


def union_length(intervals):
    end, covered = 0, 0
    for lo, hi in sorted(intervals):
        covered += max(0, hi - max(lo, end))
        end = max(end, hi)
    return covered


def main():
    failures = read(OUTPUT / "failure-analysis.json")
    aliases = read(BASE / "enterprise/duplicate-id-aliases.json")
    db = sqlite3.connect((BASE / "enterprise/corpus.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    audit = {"model_calls": 0, "kind": "retrospective_original_character_exposure", "groups": {},
        "limitations": ["Fraction of document characters is not fraction of answer facts covered.",
            "Full document exposure does not prove attention, reasoning or correct interpretation.",
            "Reference IDs are fixed historical labels; alternative legitimate evidence is not excluded."]}
    try:
        for group in ("dev_wrong", "cdl_losses"):
            rows = []
            for case in failures[group]["cases"]:
                trace = read(ROOT / case["trace"])
                gold = {aliases.get(doc, doc) for doc in case["gold_ids"]}
                texts = {doc: db.execute("SELECT text FROM documents WHERE id=?", (doc,)).fetchone()[0] for doc in gold}
                intervals, mismatches, variants = {doc: [] for doc in gold}, [], []
                for event in trace["events"]:
                    if event.get("role") != "tool":
                        continue
                    for hit in event["content"].get("hits", [event["content"]]):
                        raw_doc = hit.get("document_id")
                        doc = aliases.get(raw_doc, raw_doc)
                        if doc not in gold or not hit.get("text"):
                            continue
                        lo, body = hit.get("offset", 0), hit["text"]
                        hi = lo + len(body)
                        original = texts[doc] if raw_doc == doc else db.execute("SELECT text FROM documents WHERE id=?", (raw_doc,)).fetchone()[0]
                        if original[lo:hi] != body:
                            mismatches.append({"document_id": doc, "offset": lo, "chars": len(body)})
                        elif raw_doc != doc and texts[doc][lo:hi] != body:
                            # Recall aliases can merge release IDs, not coordinates of different text versions.
                            variants.append({"raw_document_id": raw_doc, "canonical_id": doc, "offset": lo, "chars": len(body)})
                        else:
                            intervals[doc].append([lo, hi])
                documents = [{"document_id": doc, "document_chars": len(texts[doc]),
                    "exposed_chars": union_length(intervals[doc]), "intervals": sorted(intervals[doc]),
                    "character_fraction": union_length(intervals[doc]) / len(texts[doc]) if texts[doc] else 1.}
                    for doc in sorted(gold)]
                rows.append({"id": case["id"], "category": case["category"], "bucket": case["bucket"],
                    "documents": documents, "source_slice_mismatches": mismatches, "alias_text_variants": variants,
                    "all_reference_documents_fully_exposed": all(d["exposed_chars"] == d["document_chars"] for d in documents) if documents else None,
                    "minimum_document_character_fraction": min(d["character_fraction"] for d in documents) if documents else None})
            seen_rows = [r for r in rows if r["bucket"] == "all_reference_documents_seen_answer_failed"]
            audit["groups"][group] = {"n": len(rows), "all_reference_seen_failed": len(seen_rows),
                "all_reference_fully_exposed_failed": sum(r["all_reference_documents_fully_exposed"] for r in seen_rows),
                "median_minimum_character_fraction_when_all_ids_seen": statistics.median(r["minimum_document_character_fraction"] for r in seen_rows),
                "mismatched_source_slices": sum(len(r["source_slice_mismatches"]) for r in rows),
                "alias_text_variant_slices": sum(len(r["alias_text_variants"]) for r in rows), "cases": rows}
    finally:
        db.close()
    save(OUTPUT / "exposure-audit.json", audit)
    print(json.dumps({k: {key: value for key, value in v.items() if key != "cases"} for k, v in audit["groups"].items()}, indent=2))
    return audit


if __name__ == "__main__":
    main()
