"""Render the existing three-call recording; never generate or judge an answer."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "examples/sol-demo"


def main():
    run = json.loads((ROOT / "runs/evidence/sol-demo-recorded-run.json").read_text())
    search, reading = [event for event in run["events"] if event["role"] == "tool"]
    source = reading["content"]
    begin = source["text"].index("Proposed draft SLA targets")
    end = source["text"].index("TTF (tier-adjusted)")
    usage = run["model_runs"][-1]["usage"]["total"]
    data = {"question": run["question"], "answer": run["result"]["answer"],
            "model": run["model"], "calls": len(run["model_runs"]),
            "input": usage["input_tokens"], "cached": usage["cached_input_tokens"],
            "search": json.loads(search["arguments_json"]), "read": json.loads(reading["arguments_json"]),
            "hits": [{"title": hit["title"], "offset": hit["offset"], "chars": len(hit["text"]),
                      "selected": hit["document_id"] == source["document_id"]} for hit in search["content"]["hits"]],
            "source": {"id": source["document_id"], "title": source["title"],
                       "offset": source["offset"] + begin, "text": source["text"][begin:end].strip()},
            "read_chars": len(source["text"])}
    (DIRECTORY / "recorded-summary.json").write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
    template = Path(__file__).with_name("showcase-template.html").read_text()
    (DIRECTORY / "report.html").write_text(template.replace("{{DATA}}", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))
    print(DIRECTORY / "report.html")


if __name__ == "__main__":
    main()
