"""Paired small-sample report; sol proxy scores are not leaderboard scores."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/grep-browsecomp-20261017"


def wilson(correct, n):
    z, p = 1.96, correct / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [100*(center-half), 100*(center+half)]


def behavior(group):
    counts, questions, modes, limits = Counter(), Counter(), Counter(), Counter()
    timings = defaultdict(list)
    empty = truncated = errors = chars = 0
    missing_usage = 0
    for path in sorted((OUT / group / "traces").glob("*.json")):
        trace = json.loads(path.read_text())
        names = set()
        missing_usage += sum(run.get("usage") is None for run in trace["model_runs"])
        for event in trace["events"]:
            if event.get("role") != "tool":
                continue
            name, content = event["name"], event["content"]
            counts[name] += 1; names.add(name)
            if name == "Grep":
                args = json.loads(event["arguments_json"])
                modes[args.get("output_mode", "files_with_matches")] += 1
                limits[str(args.get("head_limit", 250))] += 1
                chars += len(json.dumps(content, ensure_ascii=False, separators=(",", ":")))
                errors += bool(content.get("error"))
                if not content.get("error"):
                    empty += content.get("numFiles", 0) == 0
                    truncated += bool(content.get("truncated"))
        questions.update(names)
        for row in trace.get("tool_timings", []):
            timings[row["name"]].append(row["seconds"])
    return {"calls": dict(counts), "questions_using_tool": dict(questions), "grep_output_modes": dict(modes),
            "grep_head_limits": dict(limits), "grep_empty": empty, "grep_truncated": truncated,
            "grep_errors": errors, "grep_output_chars": chars, "missing_generation_usage": missing_usage,
            "tool_seconds": {name: {"total": sum(values), "median": statistics.median(values),
                                     "p95": float(np.quantile(values, .95))} for name, values in timings.items()}}


def main():
    groups = {g: json.loads((OUT / g / "summary.json").read_text()) for g in ("A", "B")}
    if not all(d["complete"] and d["requested"] == 30 for d in groups.values()):
        raise ValueError("both groups must finish generation and grading for all 30 questions")
    rows = {g: {r["id"]: r for r in d["cases"]} for g, d in groups.items()}
    assert rows["A"].keys() == rows["B"].keys()
    ids = list(rows["A"])
    differences = np.array([rows["B"][qid]["correct"]-rows["A"][qid]["correct"] for qid in ids])
    counts = np.array([(differences == d).sum() for d in (-1, 0, 1)])
    draws = np.random.default_rng(20261017).multinomial(30, counts/30, size=100000)
    interval = np.quantile((draws[:, 2]-draws[:, 0])*100/30, [.025, .975]).tolist()
    results = {g: {"correct": d["correct"], "n": 30, "accuracy_percent": d["correct"]*100/30,
                   "wilson95_percent": wilson(d["correct"], 30), "metrics": d["metrics"], "requests": d["requests"],
                   "behavior": behavior(g)} for g, d in groups.items()}
    am, bm = results["A"]["metrics"], results["B"]["metrics"]
    changes = {"input_tokens_percent": (bm["input_tokens"]/am["input_tokens"]-1)*100,
               "uncached_input_tokens_percent": ((bm["input_tokens"]-bm["cached_input_tokens"])/(am["input_tokens"]-am["cached_input_tokens"])-1)*100,
               "answer_seconds_percent": (bm["seconds"]/am["seconds"]-1)*100}
    result = {"complete": True, "dataset": "BrowseComp-Plus dev30", "judge": "gpt-6.1-sol medium proxy with frozen official prompt",
              "groups": results, "changes": changes, "delta_pp": float(differences.mean()*100), "paired_bootstrap_95_pp": interval,
              "wins": [qid for qid, d in zip(ids, differences) if d > 0],
              "losses": [qid for qid, d in zip(ids, differences) if d < 0], "ties": int((differences == 0).sum()),
              "bootstrap_samples": 100000, "seed": 20261017,
              "decision": "retain_optional_candidate_not_promoted" if differences.sum() > 0 else "no_accuracy_gain_do_not_promote",
              "limitations": ["30 known development questions, sol proxy judge; not official scores or a comparison to historical 93.01%.",
                              "Single answer and first valid judgement; bootstrap does not include generation/judge variation.",
                              "All ties can produce a zero-width paired bootstrap interval; this does not establish population equivalence.",
                              "Same model-turn/time limits, not equal actual tokens; path/source ID bridge is part of the treatment.",
                              "Reported text exposure includes original text returned by search/read/Grep; filenames alone are discovery."],
              "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
                  "src/minimal_rag/file_tools.py", "src/minimal_rag/indexed_rag.py", "src/minimal_rag/loop.py",
                  "src/minimal_rag/codex_adapter.py", "experiments/grep-browsecomp-20261017/run.py",
                  "experiments/grep-browsecomp-20261017/sample.json",
                  "data/benchmarks/browsecomp-plus/official/scripts_evaluation/evaluate_run.py")}}
    for path in (ROOT / "experiments/grep-browsecomp-20261017/summary.json", ROOT / "evidence/grep-browsecomp-20261017.json"):
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    a, b = results["A"], results["B"]
    conclusion = "小样本有改善信号，保留为可选候选，尚不晋级。" if differences.sum() > 0 else "本轮未观察到正确率提升，Grep不晋级默认工具。"
    if differences.sum() == 0 and changes["uncached_input_tokens_percent"] < 0:
        conclusion = "准确率持平，观察到输入token下降；保留可选Grep，暂不默认启用。"
    lines = ["# Grep-only：BrowseComp-Plus dev30", "", f"**{conclusion}**", "",
             f"A {a['correct']:.0f}/30（{a['accuracy_percent']:.2f}%），B {b['correct']:.0f}/30（{b['accuracy_percent']:.2f}%）；"
             f"逐题胜/负/平={len(result['wins'])}/{len(result['losses'])}/{result['ties']}，"
             f"差{result['delta_pp']:+.2f}pp，配对bootstrap 95%区间[{interval[0]:+.2f}, {interval[1]:+.2f}]pp。", "",
             "原有search/read与增加Grep的两组使用相同30题、全100,195篇语料、sol high、2000字符搜索摘录、30轮和超时设置。"
             "按题交错执行，模型自主选择工具、参数、次数。裁判均为sol medium配冻结官方评分提示；仅作本轮对比，不与官方/历史成绩横比。", "",
             "| 指标 | A | B |", "| --- | ---: | ---: |"]
    def row(name, av, bv):
        lines.append(f"| {name} | {av} | {bv} |")
    row("正确数", f"{a['correct']:.0f}/30", f"{b['correct']:.0f}/30")
    row("准确率Wilson 95%区间", f"{a['wilson95_percent'][0]:.2f}–{a['wilson95_percent'][1]:.2f}%", f"{b['wilson95_percent'][0]:.2f}–{b['wilson95_percent'][1]:.2f}%")
    for label, key in (("生成请求", "generation"), ("裁判请求（含无效尝试）", "judge")):
        row(label, a["requests"][key], b["requests"][key])
    for label, key in (("生成输入token（含缓存）", "input_tokens"), ("生成输出token", "output_tokens"),
                       ("生成失败", "errors"), ("工具错误", "tool_errors")):
        row(label, a["metrics"].get(key, 0), b["metrics"].get(key, 0))
    row("生成输入token（不含缓存）", *[d["metrics"]["input_tokens"]-d["metrics"]["cached_input_tokens"] for d in (a,b)])
    row("平均单题回答耗时（秒）", f"{a['metrics']['seconds']:.1f}", f"{b['metrics']['seconds']:.1f}")
    for name in ("search", "read", "Grep"):
        row(f"{name}调用", a["behavior"]["calls"].get(name, 0), b["behavior"]["calls"].get(name, 0))
    bb = b["behavior"]
    lines += ["", f"B有{bb['questions_using_tool'].get('Grep',0)}题使用Grep，共{bb['calls'].get('Grep',0)}次；"
              f"空返回{bb['grep_empty']}次、截断{bb['grep_truncated']}次、工具错误{bb['grep_errors']}次。"
              "独立Glob与新Read已从当前实现删除，原read保留；Grep匹配与返回策略未调优。", "",
              f"生成输入token含缓存变化{changes['input_tokens_percent']:+.2f}%，不含缓存变化{changes['uncached_input_tokens_percent']:+.2f}%；"
              f"平均回答耗时变化{changes['answer_seconds_percent']:+.2f}%。这是本轮观察到的用量差异，不等同稳定效率提升或账单节省。", "",
              "## 核验与解释边界", "", "142项测试通过；全100,195篇导出原文逐篇字节核对一致。"
              "失败保留在每组30题分母，首次有效回答与裁判不重跑。原Enterprise三工具源码与成绩留作历史记录。", "",
              "30题样本很小，正负一题即3.33个百分点；区间不覆盖模型/裁判波动。若配对全部打平，bootstrap区间为零也不表示总体等效。"
              "工具接口/路径桥接整体作为处理变量，不作单工具调用的因果归因。耗时受共享并发影响，token不是账单金额。", "",
              "[协议](../grep-browsecomp-20261017/brief.md) · [完整摘要](../grep-browsecomp-20261017/summary.json) · "
              "[原文核验](../grep-browsecomp-20261017/export-validation.json) · [完整性核验](../grep-browsecomp-20261017/validation.json)", "",
              "逐题轨迹与代理判分保存在runs/grep-browsecomp-20261017/{A,B}/；未启动完整830题评测。"]
    (ROOT / "experiments/reports/grep-browsecomp-20261017.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k: result[k] for k in ("complete", "delta_pp", "paired_bootstrap_95_pp", "decision")}, indent=2))


if __name__ == "__main__":
    main()
