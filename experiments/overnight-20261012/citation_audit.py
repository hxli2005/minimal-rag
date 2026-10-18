"""Check current live answers against actually supplied source IDs; no models."""
from collections import defaultdict

from worker import OUTPUT, RESULTS, HISTORY, prepare, read, save
from minimal_rag_experiments.hotpot_benchmark import exposure


def seen(events):
    sources = set()
    for event in events:
        if event.get("role") != "tool":
            continue
        for row in event["content"].get("hits", [event["content"]]):
            if row.get("document_id") and row.get("text"):
                sources.add(row["document_id"])
    return sources


cohort = prepare()
cases = {c["id"]: c for rows in cohort["datasets"].values() for c in rows}
cases.update({c["id"]: c for c in cohort["oracle"]})
groups = defaultdict(list)
max_root = OUTPUT / "oracle-max-results"
for path in sorted([*RESULTS.glob("*/*/*.json"), *max_root.glob("*/*/*.json")]):
    dataset, mode = path.parts[-3:-1]
    row = read(path)
    if row.get("result", {}).get("status") != "finished":
        continue
    allowed = seen(row.get("events", []))
    if dataset == "oracle":
        allowed = set(cases[row["id"]]["gold_ids"]) if mode == "gold" else seen(read(
            HISTORY / "runs/enterprise-dev100-hybrid-minilm-wide/traces" / f"{row['id']}.json")["events"])
    cited = set(row["result"].get("sources", []))
    item = {"id": row["id"], "provided_source_count": len(allowed), "cited_source_count": len(cited),
            "unseen_citations": sorted(cited - allowed)}
    if dataset == "hotpotqa":
        item["support_exposure"] = exposure(cases[row["id"]]["raw"], row["events"])
    name = f"oracle_max/{mode}" if path.is_relative_to(max_root) else f"{dataset}/{mode}"
    groups[name].append(item)
save(OUTPUT / "citation-audit.json", {"model_calls": 0, "kind": "actual_supplied_source_id_audit",
    "groups": {name: {"finished": len(rows), "answers_with_unseen_citations": sum(bool(r["unseen_citations"]) for r in rows),
                "cases": rows} for name, rows in groups.items()},
    "limitations": ["A cited ID was supplied does not prove its passage entails the answer.",
        "This checks raw local IDs, preserving different text versions; no alias coordinate merging.",
        "One-shot observations and oracle evidence were supplied directly in the user payload, not via model-initiated tools."]})
print({name: (len(rows),sum(bool(r["unseen_citations"]) for r in rows)) for name, rows in groups.items()})
