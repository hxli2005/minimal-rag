"""CPU experiment: prioritize conjunctive hits, then fill from original OR rank."""
import sqlite3
import statistics
import time
from collections import defaultdict

from offline import ranking
from worker import BASE, OUTPUT, read, save

db = sqlite3.connect((BASE / "enterprise/corpus.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
db.execute("PRAGMA cache_size=-16000")
aliases = read(BASE / "enterprise/duplicate-id-aliases.json")
directory = OUTPUT / "conjunction"
directory.mkdir(parents=True, exist_ok=True)
try:
    for source in sorted((OUTPUT / "bm25").glob("historical_first_query-*-current.json")):
        baseline = read(source)
        target = directory / source.name
        if target.exists():
            continue
        query = baseline["query"]
        begin = time.monotonic()
        strict = ranking(db, query["query"], 2, False, aliases, query.get("scope", ""), True, query.get("offset", 0))
        ids = list(dict.fromkeys([r["document_id"] for r in strict] + baseline["ranked"]))[:20]
        gold = set(baseline["gold"])
        row = {"id": baseline["id"], "category": baseline["category"], "partition": baseline["partition"],
            "strict_hit_count": len(strict), "query": query, "ranked": ids, "seconds": time.monotonic()-begin,
            "recall5": len(gold & set(ids[:5]))/len(gold) if gold else None,
            "recall20": len(gold & set(ids))/len(gold) if gold else None,
            "baseline_recall5": baseline["recall5"], "baseline_recall20": baseline["recall20"]}
        save(target, row)
        print(row["id"], "AND hits", len(strict), flush=True)
        time.sleep(.2)
finally:
    db.close()
rows = [read(p) for p in directory.glob("*.json")]
groups = {}
for partition in ("selection", "validation"):
    selected = [r for r in rows if r["partition"] == partition and r["recall20"] is not None]
    delta = [r["recall20"]-r["baseline_recall20"] for r in selected]
    groups[partition] = {"n": len(selected), "recall5": statistics.mean(r["recall5"] for r in selected),
        "recall20": statistics.mean(r["recall20"] for r in selected),
        "baseline_recall5": statistics.mean(r["baseline_recall5"] for r in selected),
        "baseline_recall20": statistics.mean(r["baseline_recall20"] for r in selected),
        "gain20": statistics.mean(delta), "wins": sum(d>0 for d in delta), "losses": sum(d<0 for d in delta)}
save(OUTPUT / "conjunction-summary.json", {"completed": len(rows), "requested": 100, "complete": len(rows)==100,
    "model_calls": 0, "groups": groups, "rule": "Title2. Require all original literal terms/phrases first, then fill from original OR ranking to20. No query or gold changes.",
    "limitations": ["Known dev halves, post-hoc additional hypothesis after initial title experiments; not preregistered original family.",
        "Document recall only; duplicate-ID aliases retained, no answer-quality or official score claim."]})
