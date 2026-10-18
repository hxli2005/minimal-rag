"""Summarize the prespecified paired development comparison, including costs."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from minimal_rag_experiments.benchmark import BASE, load_cases

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/code-tools-20261017"


def behavior(group, golds, aliases):
    calls, cases, timings = Counter(), Counter(), defaultdict(list)
    new_only, exclusive_gold, first_gold = [], [], []
    parameter_counts = defaultdict(Counter)
    for path in sorted((OUT / group / "traces").glob("*.json")):
        trace = json.loads(path.read_text())
        names, seen, extra, first = set(), set(), set(), {}
        for event in trace.get("events", []):
            if event.get("role") != "tool":
                continue
            name, content = event["name"], event["content"]
            calls[name] += 1; names.add(name)
            exposed = {aliases.get(doc, doc) for doc in event.get("observed_document_ids", [])}
            extra.update(exposed)
            for hit in content.get("hits", [content]):
                if hit.get("document_id") and hit.get("text"):
                    doc = aliases.get(hit["document_id"], hit["document_id"])
                    seen.add(doc); exposed.add(doc)
            for doc in exposed:
                first.setdefault(doc, name)
            if name == "Grep":
                arguments = json.loads(event["arguments_json"])
                parameter_counts["Grep.output_mode"][arguments.get("output_mode", "files_with_matches")] += 1
                parameter_counts["Grep.head_limit"][str(arguments.get("head_limit", 250))] += 1
                parameter_counts["Grep.scope"]["global" if arguments.get("path", "/corpus") == "/corpus" else "scoped"] += 1
        cases.update(names)
        if extra - seen:
            new_only.append(trace["id"])
        gold = golds[trace["id"]]
        if (extra - seen) & gold:
            exclusive_gold.append(trace["id"])
        if any(first.get(doc) in ("Grep", "Read") for doc in gold):
            first_gold.append(trace["id"])
        for row in trace.get("tool_timings", []):
            timings[row["name"]].append(row["seconds"])
    return {"calls": dict(calls), "questions_using_tool": dict(cases),
            "questions_with_text_from_documents_only_seen_via_new_tools": new_only,
            "new_tools_exclusive_reference_questions": exclusive_gold,
            "new_tools_first_reference_questions": first_gold,
            "parameter_counts": {name: dict(values) for name, values in parameter_counts.items()},
            "tool_seconds": {name: {"total": sum(values), "median": statistics.median(values),
                                     "p95": float(np.quantile(values, .95))} for name, values in timings.items()}}


def main():
    summaries = {group: json.loads((OUT / group / "summary.json").read_text()) for group in ("A", "B")}
    if not all(row["complete"] and row["requested"] == 100 for row in summaries.values()):
        raise ValueError("both development groups must be complete and graded before comparison")
    rows = {group: {row["id"]: row for row in data["cases"]} for group, data in summaries.items()}
    if rows["A"].keys() != rows["B"].keys():
        raise ValueError("paired cohorts differ")
    ids = list(rows["A"])
    _, cases = load_cases("enterprise", "dev")
    golds = {case["id"]: set(case["gold_ids"]) for case in cases}
    aliases = json.loads((BASE / "enterprise/duplicate-id-aliases.json").read_text())
    differences = np.array([rows["B"][qid]["correct"] - rows["A"][qid]["correct"] for qid in ids])
    counts = np.array([(differences == d).sum() for d in (-1, 0, 1)])
    draws = np.random.default_rng(20261017).multinomial(100, counts/100, size=100000)
    interval = np.quantile((draws[:, 2]-draws[:, 0]), [.025, .975]).tolist()
    groups = {group: {"correct": data["correct"], "metrics": data["metrics"], "requests": data["requests"],
                      "behavior": behavior(group, golds, aliases)} for group, data in summaries.items()}
    categories = {}
    for category in sorted({row["category"] for row in rows["A"].values()}):
        subset = [qid for qid in ids if rows["A"][qid]["category"] == category]
        categories[category] = {"n": len(subset), **{group: sum(rows[group][qid]["correct"] for qid in subset) for group in groups}}
    result = {"complete": True, "dataset": "Enterprise dev100", "groups": groups,
              "wins": [qid for qid, d in zip(ids, differences) if d > 0],
              "losses": [qid for qid, d in zip(ids, differences) if d < 0],
              "ties": int((differences == 0).sum()), "delta_pp": float(differences.sum()),
              "paired_bootstrap_95_pp": interval, "bootstrap_samples": 100000, "seed": 20261017,
              "by_category": categories,
              "decision": "candidate_for_frozen_test_not_promoted" if differences.sum() > 0 else "do_not_promote",
              "limitations": ["Known development cohort, local sol judge; not an official or SOTA score.",
                              "Single generation per question; bootstrap does not cover generation/judge resampling.",
                              "Same model-turn/time caps, not equal actual tokens; tool defaults and path bridge are treatment.",
                              "New-tool exclusive document exposure is descriptive, not causal necessity.",
                              "File listing/count do not count as original-text evidence exposure."],
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
                  "experiments/code-tools-20261017/frozen_file_tools.py", "src/minimal_rag/indexed_rag.py", "src/minimal_rag/loop.py",
                  "src/minimal_rag/codex_adapter.py", "experiments/code-tools-20261017/run.py",
                  "experiments/src/minimal_rag_experiments/benchmark.py",
                  "experiments/src/minimal_rag_experiments/benchmark_metrics.py",
                  "experiments/src/minimal_rag_experiments/rag_judge.py",
                  "data/benchmarks/enterprise/dev.json")}}
    for path in (ROOT / "experiments/code-tools-20261017/summary.json", ROOT / "evidence/code-tools-20261017.json"):
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    a, b = groups["A"], groups["B"]
    verdict = ("B开发集正确数更高，可作为后续冻结test的候选；本轮不自动晋级默认工具。"
               if result["delta_pp"] > 0 else "新增工具未提高开发集正确数，默认工具继续保持A。")
    lines = ["# Enterprise dev100：成熟文件检索工具对照", "", f"**{verdict}**", "",
             f"A {a['correct']:.0f}/100，B {b['correct']:.0f}/100，差{result['delta_pp']:+.0f}个百分点；"
             f"逐题胜/负/平={len(result['wins'])}/{len(result['losses'])}/{result['ties']}，"
             f"配对bootstrap 95%区间[{interval[0]:+.0f}, {interval[1]:+.0f}]pp。", "",
             "A为当前纯BM25 search/read；B在A上增加Grep/Glob/Read和路径/来源ID映射。"
             "同dev100、全库原文、sol high、12次模型调用、相同超时、原sol medium裁判；"
             "按题交错执行，首次有效结果保留。开发集与本地裁判成绩，不是官方或SOTA成绩。", "",
             "| 指标 | A | B |", "| --- | ---: | ---: |"]
    def metric(label, x, y):
        lines.append(f"| {label} | {x} | {y} |")
    metric("正确数", f"{a['correct']:.0f}/100", f"{b['correct']:.0f}/100")
    for label, key in (("生成请求", "generation"), ("裁判请求（含无效尝试）", "judge")):
        metric(label, a["requests"][key], b["requests"][key])
    for label, key in (("生成输入token（含缓存）", "input_tokens"), ("其中缓存token", "cached_input_tokens"),
                       ("生成输出token", "output_tokens"), ("运行失败", "errors"), ("工具错误", "tool_errors")):
        metric(label, a["metrics"].get(key, 0), b["metrics"].get(key, 0))
    metric("生成输入token（不含缓存）", *[g["metrics"].get("input_tokens", 0)-g["metrics"].get("cached_input_tokens", 0) for g in (a, b)])
    metric("平均单题回答耗时（秒）", f"{a['metrics']['seconds']:.1f}", f"{b['metrics']['seconds']:.1f}")
    lines += ["", "轮数和超时上限相同，实际token不等额。新工具的默认返回长度、行单位、路径字段与来源映射属于处理变量。"
              "耗时受共享并发队列影响；token只统计生成，裁判请求单列，不能等同账单金额。", "",
              "## 工具实际使用", "", "| 工具 | A调用 | B调用 | B使用题数 |", "| --- | ---: | ---: | ---: |"]
    for name in ("search", "read", "Grep", "Glob", "Read"):
        lines.append(f"| {name} | {a['behavior']['calls'].get(name, 0)} | {b['behavior']['calls'].get(name, 0)} | {b['behavior']['questions_using_tool'].get(name, 0)} |")
    params = b["behavior"]["parameter_counts"]
    lines += ["", "模型自主选择的Grep参数分布：", "",
              "- 输出模式：" + json.dumps(params.get("Grep.output_mode", {}), ensure_ascii=False),
              "- 搜索范围：" + json.dumps(params.get("Grep.scope", {}), ensure_ascii=False),
              "- head_limit：" + json.dumps(params.get("Grep.head_limit", {}), ensure_ascii=False),
              "", "本轮Grep的1次错误为不支持的look-around正则；模型收到诊断后继续调用成功，最终该题正常完成。"
              "未使用count模式或新Read，不能把这些能力视作已验证有效。"]
    lines += ["", f"B中有{len(b['behavior']['new_tools_exclusive_reference_questions'])}题的参考文档正文仅经新工具观察到；"
              f"有{len(b['behavior']['new_tools_first_reference_questions'])}题的参考文档正文首次经新工具出现。"
              "这是轨迹描述，不代表这些工具对最终正确性具有因果必要性。仅列路径或计数不计正文曝光。", "",
              "## 按题型", "", "| 题型 | 题数 | A正确 | B正确 |", "| --- | ---: | ---: | ---: |"]
    for name, values in categories.items():
        lines.append(f"| {name} | {values['n']} | {values['A']:.0f} | {values['B']:.0f} |")
    lines += ["", "## 验证与边界", "",
              "143项测试通过；全511,962篇导出逐篇核对原文UTF-8字节。"
              "Grep使用真实ripgrep扫描完整语料，不先经过BM25候选筛选。"
              "同接口保留默认参数与分页含义，文本语料适配和预算差异见冻结协议。", "",
              "单次生成与判分，bootstrap仅反映题目差异，不覆盖模型/裁判重复采样。"
              "本轮只比较新增工具包整体，不单独归因于Grep、Glob或Read；没有调用的能力不构成效果验证。"
              "未启动test评测，未替换框架、BM25、核心循环或默认工具。", "",
              "[冻结协议](../code-tools-20261017/brief.md) · [完整汇总](../code-tools-20261017/summary.json) · "
              "[导出核验](../code-tools-20261017/export-validation.json) · "
              "[生成完整性](../code-tools-20261017/generation-validation.json) · "
              "[判分核验](../code-tools-20261017/judgement-validation.json)", "",
              "逐题回答、工具参数/返回、模型用量与裁判尝试保存在 `runs/code-tools-20261017/{A,B}/`。"]
    (ROOT / "experiments/reports/code-tools-20261017.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({key: result[key] for key in ("complete", "delta_pp", "paired_bootstrap_95_pp", "decision")}, indent=2))


if __name__ == "__main__":
    main()
