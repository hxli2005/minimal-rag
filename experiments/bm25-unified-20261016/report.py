"""Publish the completed BM25 regression results without touching historical scores."""
import json
from pathlib import Path
import random
import statistics
from collections import Counter

from minimal_rag.paths import ROOT
from minimal_rag_experiments.hotpot_benchmark import save, sha256

HERE = Path(__file__).resolve().parent
RUN = ROOT / "runs/bm25-unified-20261016"
DATASETS = ("enterprise", "financebench", "hotpotqa")


def compare(previous, current, metric):
    old = {row["id"]: row for row in previous["cases"]}
    new = {row["id"]: row for row in current["cases"]}
    assert old.keys() == new.keys(), "comparison must use the same complete cohort"
    deltas = [new[qid][metric] - old[qid][metric] for qid in sorted(new)]
    rng = random.Random(20261016)
    draws = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) * 100 for _ in range(10000))
    return {"metric": metric, "previous": statistics.mean(row[metric] for row in old.values()),
            "current": statistics.mean(row[metric] for row in new.values()),
            "delta_pp": statistics.mean(deltas) * 100, "paired_bootstrap_95_pp": [draws[249], draws[9749]],
            "wins": sum(d > 0 for d in deltas), "losses": sum(d < 0 for d in deltas),
            "ties": sum(d == 0 for d in deltas)}


def main():
    summaries = {d: json.loads((RUN / f"{d}-test/summary.json").read_text()) for d in DATASETS}
    if any(not s["complete"] for s in summaries.values()):
        print(json.dumps({d: {k: s[k] for k in ("completed", "requested", "complete")} for d, s in summaries.items()}))
        return 1
    result = {"implementation": "bm25-agent", "datasets": {},
              "browsecomp": json.loads((HERE / "browsecomp-replay.json").read_text()),
              "comparison": "Same questions, generation model, prompts, output formats, budgets and local judging rules. Historical retrieval/transport versus current BM25/transport; no attribution to retrieval alone."}
    for dataset, summary in summaries.items():
        current_path = RUN / f"{dataset}-test/summary.json"
        old_path = ROOT / f"data/benchmarks/sota-20261014/{dataset}-test/summary.json"
        previous = json.loads(old_path.read_text())
        metrics = ("em", "f1") if dataset == "hotpotqa" else ("correct",)
        assert summary["metrics"]["graded"] == summary["requested"]
        assert summary["config"]["model"] == previous["config"]["model"]
        assert summary["config"]["format_request"] == previous["config"]["format_request"]
        assert summary["config"]["settings"]["prompt"] == previous["config"]["settings"]["prompt"]
        if dataset != "hotpotqa":
            assert summary["config"]["judge_prompt"] == previous["config"]["judge_prompt"]
        result["datasets"][dataset] = {k: summary[k] for k in ("config", "requested", "completed", "complete", "metrics", "requests", "by_category")}
        tool_counts = Counter()
        multi_search_cases = read_cases = 0
        for trace in (RUN / f"{dataset}-test/traces").glob("*.json"):
            record = json.loads(trace.read_text())
            counts = Counter(e["name"] for e in record["events"] if e.get("role") == "tool")
            tool_counts.update(counts)
            multi_search_cases += counts["search"] > 1
            read_cases += counts["read"] > 0
        result["datasets"][dataset]["agent_behavior"] = {"tool_calls": dict(tool_counts),
            "cases_with_multiple_searches": multi_search_cases, "cases_with_read": read_cases}
        result["datasets"][dataset].update(
            previous_metrics=previous["metrics"],
            previous_generation_requests=sum(r["model_calls"] for r in previous["cases"]),
            comparison=[compare(previous, summary, metric) for metric in metrics],
            summary_path=str(current_path.relative_to(ROOT)), summary_sha256=sha256(current_path),
            previous_path=str(old_path.relative_to(ROOT)), previous_sha256=sha256(old_path))
    save(HERE / "summary.json", result)
    save(ROOT / "evidence/bm25-unified-20261016.json", result)
    lines = ["# 纯 BM25 统一实现评测", "",
             "当前问答与四个 benchmark 共用 `IndexedKnowledgeBase`、`indexed_tools`、`CodexModel` 和 `run_agent`。模型自主执行多轮 search/read；保留文档内搜索、分页和继续阅读。历史混合检索、cascade、重排及其成绩保存在实验目录。", "",
             "代码测试138项通过，三个开发冒烟全部完成且正确。主入口导入不加载 NumPy、Torch、embedding 或混合检索模块；历史冻结入口的加载与 `--help` 验证通过。", "",
             "| 基准 | 当前纯 BM25 | 历史实现 | 差值 | 配对胜／负 |", "| --- | ---: | ---: | ---: | ---: |"]
    for dataset in DATASETS:
        for row in result["datasets"][dataset]["comparison"]:
            label = dataset + (f" {row['metric'].upper()}" if dataset == "hotpotqa" else "")
            lines.append(f"| {label} | {row['current']*100:.2f}% | {row['previous']*100:.2f}% | {row['delta_pp']:+.2f}pp | {row['wins']}／{row['losses']} |")
    ent = result["datasets"]["enterprise"]
    lines += ["", f"**Enterprise 存在明确回退。** 配对正确率差为{ent['comparison'][0]['delta_pp']:+.2f}pp，95%区间{ent['comparison'][0]['paired_bootstrap_95_pp']}pp；生成调用从{ent['previous_generation_requests']}增至{ent['requests']['generation']}，输入token（含缓存）从{ent['previous_metrics']['input_tokens']/1e6:.2f}M增至{ent['metrics']['input_tokens']/1e6:.2f}M。观察到的参考文档召回从{ent['previous_metrics']['retrieved_document_recall']*100:.2f}%降至{ent['metrics']['retrieved_document_recall']*100:.2f}%。本轮完成了结构统一，但没有提升Enterprise成绩或降低模型调用量。", "",
              "Enterprise 为400题，FinanceBench与HotpotQA各100题。生成均为sol high；Enterprise／Finance使用相同sol medium裁判，HotpotQA为官方EM/F1。前三者仍为4000字符摘录、12轮。历史Enterprise对照为v0 test400，v3只有dev记录。对比保留了题目、模型、提示、格式和预算；新实现同时统一当前适配器，不将差值全部归因于检索。", "",
              "BrowseComp保持原来的BM25、2000字符、30轮；抽取12题共24次历史搜索，排名、文本、偏移逐字段一致，没有新增生成调用。原有772/830（93.01%）继续保留在历史成绩中。", "",
              "## 执行记录", ""]
    for dataset in DATASETS:
        s = result["datasets"][dataset]
        lines.append(f"- {dataset}：完成并判分{s['completed']}/{s['requested']}；生成请求{s['requests']['generation']}，裁判请求{s['requests'].get('judge',0)}；运行错误{s['metrics']['errors']}，工具错误{s['metrics']['tool_errors']}。")
    lines += ["", "逐题答案、工具轨迹、裁判输入和判决在 `runs/bm25-unified-20261016/<dataset>-test/`。可提交汇总及配对区间见 [summary.json](../bm25-unified-20261016/summary.json)，运行命令见 [brief](../bm25-unified-20261016/brief.md)。区间使用按题配对bootstrap 10,000次；Enterprise／Finance分数属于本地裁判口径。"]
    (ROOT / "experiments/reports/bm25-unified-20261016.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({d: result["datasets"][d]["comparison"] for d in DATASETS}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
