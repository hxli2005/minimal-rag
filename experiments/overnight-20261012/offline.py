"""One-night CPU query replay and evidence-window audit. No model calls."""
import argparse
import json
import re
import sqlite3
import statistics
import time
from collections import defaultdict

from worker import BASE, DIRECTORY, HISTORY, OUTPUT, failure_analysis, prepare, read, save

STOP = set("a an the is are was were be been being to of for on in at by with from that this which what who how when where why do does did and or as it its they their i we you our can could would should please tell me about most main specific explain give find according under during after before into".split())
VARIANTS = {"current": (2, False), "title0": (0, False), "title8": (8, False), "content_words": (2, True)}


def expression(query, remove_stop=False, all_terms=False):
    terms = list(dict.fromkeys(a or b for a, b in re.findall(r'"([^"\n]+)"|([\w]+)', query)))
    if remove_stop:
        terms = [t for t in terms if t.lower() not in STOP]
    return (" AND " if all_terms else " OR ").join('"' + term.replace('"', '""') + '"' for term in terms)


def ranking(db, query, weight, remove_stop, aliases, scope="", all_terms=False, offset=0):
    match = expression(query, remove_stop, all_terms)
    if not match:
        return []
    sql = "SELECT c.document_id,c.start,c.text FROM chunks c JOIN documents d ON d.id=c.document_id WHERE chunks MATCH ? AND rank MATCH ?"
    params = [match, f"bm25(0,0,{weight},1)"]
    if scope:
        if db.execute("SELECT 1 FROM documents WHERE id=?", (scope,)).fetchone():
            sql += " AND c.document_id=?"
            params.append(scope)
            ranges = db.execute("SELECT first_chunk,last_chunk FROM document_ranges WHERE document_id=?", (scope,)).fetchone()
            if ranges:
                sql += " AND c.rowid>=? AND c.rowid<=?"
                params.extend(ranges)
        else:
            # Only literal scopes are used in frozen traces; escape SQL wildcards.
            pattern = "%" + scope.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            sql += " AND (d.id LIKE ? ESCAPE '\\' OR d.title LIKE ? ESCAPE '\\' OR d.source LIKE ? ESCAPE '\\')"
            params.extend([pattern] * 3)
    sql += " ORDER BY rank"
    rows, seen = [], set()
    cursor = db.execute(sql, params)
    try:
        for row in cursor:
            canonical = aliases.get(row[0], row[0])
            if canonical in seen:
                continue
            seen.add(canonical)
            if len(seen) > offset:
                rows.append({"document_id": canonical, "offset": int(row[1]), "text": row[2]})
            if len(rows) == 20:
                break
    finally:
        cursor.close()
    return rows


def bm25():
    aliases = read(BASE / "enterprise/duplicate-id-aliases.json")
    cases = prepare()["datasets"]["enterprise"]
    # Also replay first historical query per dev question; later queries are adaptive and label this limitation.
    all_cases = read(BASE / "enterprise/dev.json")
    jobs = []
    for index, case in enumerate(all_cases):
        trace = read(HISTORY / "runs/enterprise-dev100-hybrid-minilm-wide/traces" / f"{case['question_id']}.json")
        query = next((json.loads(e["arguments_json"]) for e in trace["events"]
                      if e["role"] == "tool" and e["name"] == "search"), {"query": case["question"]})
        jobs.append({"id": case["question_id"], "category": case["question_type"],
            "gold": [aliases.get(x, x) for x in case["expected_doc_ids"]],
            "query": query, "partition": "selection" if index % 2 == 0 else "validation", "kind": "historical_first_query"})
    jobs += [{"id": c["id"], "category": c["category"], "gold": c["gold_ids"], "query": {"query": c["question"]},
              "partition": "diagnostic", "kind": "natural_question"} for c in cases]
    directory = OUTPUT / "bm25"
    directory.mkdir(exist_ok=True, parents=True)
    db = sqlite3.connect((BASE / "enterprise/corpus.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    db.execute("PRAGMA cache_size=-16000")
    try:
        for job in jobs:
            for variant, (weight, stop) in VARIANTS.items():
                path = directory / f"{job['kind']}-{job['id']}-{variant}.json"
                if path.exists():
                    continue
                begin = time.monotonic()
                q = job["query"]
                hits = ranking(db, q["query"], weight, stop, aliases, q.get("scope", ""), q.get("all_terms", False), q.get("offset", 0))
                gold = set(job["gold"])
                ids = [h["document_id"] for h in hits]
                save(path, {**job, "variant": variant, "ranked": ids, "seconds": time.monotonic() - begin,
                    "recall5": len(gold & set(ids[:5])) / len(gold) if gold else None,
                    "recall20": len(gold & set(ids)) / len(gold) if gold else None,
                    "hit_chars5": sum(len(h["text"]) for h in hits[:5]),
                    "settings": {"title_weight": weight, "remove_stopwords": stop, "tokenizer": "existing porter unicode61"}})
                print(json.dumps({"bm25": job["id"], "kind": job["kind"], "variant": variant}), flush=True)
                time.sleep(.2)
    finally:
        db.close()
    rows = [read(p) for p in directory.glob("*.json")]
    grouped = defaultdict(list)
    for row in rows:
        if row["recall20"] is not None:
            grouped[(row["kind"], row["partition"], row["variant"])].append(row)
    summary = {"model_calls": 0, "records": len(rows), "complete": len(rows) == len(jobs) * len(VARIANTS),
        "note": "Equal held-out dev halves selected before replay; not untouched benchmark holdout. No index rebuild or tokenizer change. Gold used only after ranking. Scope retains document filtering but same-document hits deduplicate by document, so this is document recall only.",
        "groups": {"/".join(k): {"n_with_gold": len(v), "recall5": statistics.mean(r["recall5"] for r in v),
            "recall20": statistics.mean(r["recall20"] for r in v), "mean_seconds": statistics.mean(r["seconds"] for r in v)} for k, v in grouped.items()}}
    save(OUTPUT / "bm25-summary.json", summary)
    return summary


def windows():
    aliases = read(BASE / "enterprise/duplicate-id-aliases.json")
    previous = HISTORY / "runs/enterprise-dev100-hybrid-minilm-wide"
    db = sqlite3.connect((BASE / "enterprise/corpus.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    cases = {c["question_id"]: c for c in read(BASE / "enterprise/dev.json")}
    rows = []
    try:
        for path in sorted((previous / "traces").glob("*.json")):
            trace = read(path)
            gold = set(cases[trace["id"]]["expected_doc_ids"])
            original = shorter = gold_original = gold_shorter = 0
            examined = []
            for event in trace["events"]:
                if event["role"] != "tool" or event["name"] != "search":
                    continue
                query = json.loads(event["arguments_json"])["query"]
                terms = [t.lower() for t in re.findall(r'\w+', query) if t.lower() not in STOP]
                for hit in event["content"].get("hits", []):
                    text = hit.get("text", "")
                    if not text:
                        continue
                    # Heuristic 1600-char contiguous window in the saved 4000-char observation.
                    # It is not exact FTS chunk reconstruction or a tokenizer measurement.
                    if len(text) <= 1600:
                        start = 0
                    else:
                        starts = range(0, len(text) - 1600 + 1, 200)
                        start = max(starts, key=lambda i: sum(text[i:i+1600].lower().count(t) for t in terms))
                    clipped = text[start:start+1600]
                    is_gold = aliases.get(hit["document_id"], hit["document_id"]) in gold
                    original += len(text); shorter += len(clipped)
                    if is_gold:
                        gold_original += len(text); gold_shorter += len(clipped)
                    examined.append({"document_id": hit["document_id"], "old_offset": hit.get("offset", 0),
                                     "new_offset": hit.get("offset", 0) + start, "original_chars": len(text), "window_chars": len(clipped), "gold_document": is_gold})
            rows.append({"id": trace["id"], "original_search_chars": original, "window_search_chars": shorter,
                         "gold_document_original_chars": gold_original, "gold_document_window_chars": gold_shorter, "windows": examined})
    finally:
        db.close()
    original, clipped = (sum(r[key] for r in rows) for key in ("original_search_chars", "window_search_chars"))
    summary = {"model_calls": 0, "n": len(rows), "original_search_chars": original, "window_search_chars": clipped,
        "search_text_char_reduction": 1 - clipped / original, "token_reduction": None,
        "note": "Offline character estimate, same returned documents, lexical-density windows. No guarantee of preserving all answer facts; dropped surrounding context requires paired online validation. Not a measured token/cost saving or exact hit-only implementation.", "cases": rows}
    save(OUTPUT / "window-audit.json", summary)
    return {k: v for k, v in summary.items() if k != "cases"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["bm25", "windows"])
    a = p.parse_args()
    OUTPUT.mkdir(exist_ok=True, parents=True)
    result = bm25() if a.stage == "bm25" else windows()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
