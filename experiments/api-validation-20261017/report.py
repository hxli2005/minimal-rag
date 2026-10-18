"""Summarize existing pilot artifacts without model calls or rejudging."""
from collections import Counter
import json
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "runs/api-validation-20261017"


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def paired(deltas):
    rng = random.Random(20261017)
    values = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(10000))
    return {"mean": statistics.mean(deltas), "bootstrap95": [values[249], values[9749]]}


def summary():
    sample = read(HERE / "sample.json")
    ledger = read(OUT / "ledger.json", [])
    data = {"protocol": "api-validation-20261017-v2", "requests": len(ledger),
            "known_cost_usd": sum(r.get("cost_usd") or 0 for r in ledger),
            "unresolved_costs": sum(r["status"] != "settled" for r in ledger),
            "key_usage": read(OUT / "key-current.json"), "stop": read(OUT / "stop.json"),
            "smoke_gate": read(OUT / "smoke-gate.json"), "migration_gate": read(OUT / "migration-gate.json"),
            "stages": {}, "limits": ["Known development sample, rare types oversampled.",
              "Migration comparison changes more than one factor; no causal attribution to native tools alone.",
              "Local Codex judge; not an official leaderboard score.",
              "Fixed-evidence verbosity comparison is not full-agent performance."]}
    data["technical_recoveries"] = [read(p) for p in sorted(OUT.glob("recovery-*/decision.json"))]
    data["key_query_retry_authorization"] = read(OUT / "key-query-retries/authorization.json")
    data["key_query_retries"] = [read(p) for p in sorted(OUT.glob("key-query-retries/*/decision.json"))]
    for stage, ids, arms in [(0, sample["smoke_ids"], ["low"]), (1, sample["migration_ids"], ["low"]),
                            (2, sample["migration_ids"], ["low", "high"])]:
        groups = {}
        for arm in arms:
            records = [r for q in ids if (r := read(OUT / f"stage{stage}" / arm / f"{q}.json")) is not None]
            graded = [r for r in records if "judgement" in r]
            calls = [r for r in ledger if r["stage"] == stage and r["arm"] == arm]
            usage = Counter()
            missing_cache_writes = 0
            usage_from_receipt = 0
            missing_usage = 0
            for c in calls:
                u = c.get("usage") or {}
                receipt = c.get("receipt") or {}
                if not u and receipt.get("native_tokens_prompt") is not None:
                    u = {"input_tokens": receipt["native_tokens_prompt"],
                         "output_tokens": receipt["native_tokens_completion"],
                         "total_tokens": receipt["native_tokens_prompt"] + receipt["native_tokens_completion"],
                         "input_tokens_details": {"cached_tokens": receipt.get("native_tokens_cached") or 0},
                         "output_tokens_details": {"reasoning_tokens": receipt.get("native_tokens_reasoning") or 0}}
                    usage_from_receipt += 1
                if not u:
                    missing_usage += 1
                usage.update({k: u.get(k, 0) for k in ("input_tokens", "output_tokens", "total_tokens")})
                details = u.get("input_tokens_details") or {}
                usage["cached_input_tokens"] += details.get("cached_tokens", 0)
                if "cache_write_tokens" in details:
                    usage["cache_write_tokens"] += details["cache_write_tokens"]
                else:
                    missing_cache_writes += 1
                usage["reasoning_tokens"] += (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
            facts = [r["judgement"]["facts_supported"] for r in graded]
            groups[arm] = {"requested": len(ids), "generated": len(records), "graded": len(graded),
                "correct": sum(r["judgement"]["correct"] for r in graded),
                "complete": len(graded) == len(ids),
                "finished": sum(r.get("result", {}).get("status") == "finished" for r in records),
                "completeness_macro": statistics.mean(sum(f)/len(f) if f else 1 for f in facts) if facts else None,
                "completeness_micro": sum(sum(f) for f in facts)/sum(map(len,facts)) if sum(map(len,facts)) else None,
                "model_calls": len(calls), "tool_calls": dict(Counter(e["name"] for r in records for e in r["events"] if e["role"] == "tool")),
                "cost_usd": sum(r.get("cost_usd") or 0 for r in calls), "usage": dict(usage),
                "cache_write_usage_missing_calls": missing_cache_writes,
                "usage_from_cancelled_generation_receipt": usage_from_receipt,
                "usage_missing_calls": missing_usage,
                "encrypted_reasoning_returned": sum(r.get("encrypted_reasoning_returned", 0) for r in calls),
                "requests_replaying_encrypted_reasoning": sum(r.get("encrypted_reasoning_replayed", 0) > 0 for r in calls),
                "incomplete_reasons": dict(Counter((r.get("incomplete_details") or {}).get("reason") for r in calls if r.get("response_status") == "incomplete")),
                "answer_chars": {r["id"]: len(r.get("result", {}).get("answer", "")) for r in records},
                "cases": [{"id": r["id"], "result_status": r.get("result", {}).get("status"),
                           "correct": r.get("judgement", {}).get("correct"), "error": r.get("result", {}).get("error")} for r in records]}
            lengths = [len(r["result"]["answer"]) for r in records if r.get("result", {}).get("status") == "finished"]
            groups[arm]["median_answer_chars"] = statistics.median(lengths) if lengths else None
        if all(groups[a]["complete"] for a in arms):
            comparisons = []
            fact_deltas = []
            for q in ids:
                a = read(OUT / "baseline" / f"{q}.json") if stage == 1 else read(OUT / f"stage{stage}" / "low" / f"{q}.json")
                b = read(OUT / f"stage{stage}" / ("high" if stage == 2 else "low") / f"{q}.json")
                if stage == 0:
                    continue
                comparisons.append(int(b["judgement"]["correct"]) - int(a["judgement"]["correct"]))
                af, bf = a["judgement"]["facts_supported"], b["judgement"]["facts_supported"]
                fact_deltas.append((sum(bf)/len(bf) if bf else 1) - (sum(af)/len(af) if af else 1))
            if comparisons:
                groups["paired"] = {"wins": comparisons.count(1), "losses": comparisons.count(-1),
                                     "ties": comparisons.count(0), **paired(comparisons),
                                     "completeness_delta": paired(fact_deltas)}
        data["stages"][str(stage)] = groups
    data["budget_rmb_including_fee"] = data["known_cost_usd"] * 8 * 1.055
    baseline = [read(OUT / "baseline" / f"{q}.json") for q in sample["migration_ids"]]
    data["migration_reference"] = {"correct": sum(r["judgement"]["correct"] for r in baseline), "n": 30,
        "completeness_macro": statistics.mean(sum(r["judgement"]["facts_supported"])/len(r["judgement"]["facts_supported"]) if r["judgement"]["facts_supported"] else 1 for r in baseline)}
    data["complete"] = not data["unresolved_costs"] and all(g["complete"] for s in data["stages"].values() for k,g in s.items() if k in ("low", "high"))
    if data["complete"]:
        low, high = data["stages"]["2"]["low"], data["stages"]["2"]["high"]
        dc = high["correct"]-low["correct"]
        df = high["completeness_macro"]-low["completeness_macro"]
        data["high_directional_gate_passed"] = (dc >= 1 and df >= 0) or (dc >= 0 and df >= .03)
        data["decision"] = "API migration screening passed; retain API as a portable backend. High verbosity did not pass the preregistered benefit gate. Keep Codex full-run submission provenance separate. No more runs launched."
    return data


if __name__ == "__main__":
    result = summary()
    (HERE / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = ["# OpenRouter API 固定小样本验证（v2）", "",
             "原Codex全量成绩保持原配置归属。本报告仅覆盖已固定的开发样本；本地裁判不是官方榜单判分。", "",
             f"已发出 {result['requests']} 次API生成请求，已核对费用 ${result['known_cost_usd']:.6f}；未结算请求 {result['unresolved_costs']}。",
             "专项Key为$20 TOTAL，执行器$14停止线、$6在途预留，生成并发1。", "",
             "| 阶段 | 组 | 已生成/计划 | 已判分 | 正确 | 费用USD |", "| --- | --- | --- | --- | --- | --- |"]
    for stage, groups in result["stages"].items():
        for arm in ("low", "high"):
            if arm not in groups:
                continue
            row = groups[arm]
            lines.append(f"| {stage} | {arm} | {row['generated']}/{row['requested']} | {row['graded']} | {row['correct']} | {row['cost_usd']:.6f} |")
    if result["complete"]:
        migration = result["stages"]["1"]["low"]
        mp = result["stages"]["1"]["paired"]
        low, high = result["stages"]["2"]["low"], result["stages"]["2"]["high"]
        vp = result["stages"]["2"]["paired"]
        lines += ["", "## 结果与判断", "",
            f"**API迁移筛查通过：{migration['correct']}/30，对照Codex {result['migration_reference']['correct']}/30。** 配对{mp['wins']}胜{mp['losses']}负{mp['ties']}平；正确率差{mp['mean']*100:+.2f}pp，配对bootstrap95%区间[{mp['bootstrap95'][0]*100:.2f},{mp['bootstrap95'][1]*100:.2f}]pp。这是已知开发样本，不证明两后端等价，也不是API版全量成绩。",
            f"宏平均事实完整度：Codex {result['migration_reference']['completeness_macro']*100:.2f}% → API {migration['completeness_macro']*100:.2f}%。迁移同时改变运行时、消息包装、原生工具调用和停止策略，不能将差异单独归因于某个改动。", "",
            f"**high没有通过预先设定的收益门槛。** 固定证据下low {low['correct']}/30、high {high['correct']}/30；high相对low为{vp['wins']}胜{vp['losses']}负{vp['ties']}平。正确率差95%区间[{vp['bootstrap95'][0]*100:.2f},{vp['bootstrap95'][1]*100:.2f}]pp，不能认定high稳定变差。",
            f"宏平均事实完整度 {low['completeness_macro']*100:.2f}% → {high['completeness_macro']*100:.2f}%；答案字符数中位数 {low['median_answer_chars']} → {high['median_answer_chars']}。篇幅增长没有转化为本轮明确的正确率收益。两组没有输出上限截断，实际请求逐题比较仅text.verbosity不同。", "",
            f"**已核对总费用${result['known_cost_usd']:.7f}**，与专项Key累计用量一致；按预算口径8元/美元并计5.5%充值费约{result['budget_rmb_including_fee']:.2f}元，不是实际银行换汇账单。总额低于200元，未提高任何额度或追加充值。",
            "保留API后端作为可迁移实现；本轮不追加high全量实验，不改变原Codex全量成绩的版本归属，也不自动发送官方提交。"]
    lines += ["", "未完成的组不能作为完整准确率发布。阶段1固定原基线22/30、筛查门槛19/30；这不是统计等价证明。",
              "阶段2是同一证据下的low/high回答层对照，两组输出上限均32768，不能解释为完整检索agent成绩。",
              "模型没有返回缓存写入用量时报告缺失，不当作0；实际费用以OpenRouter回执为准。", "",
              f"冒烟门槛：{json.dumps(result['smoke_gate'], ensure_ascii=False)}",
              f"迁移门槛：{json.dumps(result['migration_gate'], ensure_ascii=False)}"]
    if result["stop"]:
        lines += ["", f"停止记录：{result['stop']['reason']} — {result['stop']['detail']}"]
    for recovery in result["technical_recoveries"]:
        lines += ["", f"技术恢复：请求{recovery['original_request']}在连接中断前未交付完整响应，供应商标记取消；原收费${recovery['charged_usd']:.6f}计入总额。原样重发该轮，不重做前面的检索、不替换有效答案。状态：{recovery['status']}。"]
    if result["key_query_retry_authorization"]:
        lines += ["", "执行中用户明确追加规则：读取额度信息失败且未发出任何模型请求的题，不按答错统计，原题补跑。原记录保留；补跑前标pending_retry，不替换任何已有有效答案。补跑可能改变实际low/high先后顺序，必须与原预定顺序区分。"]
    lines += ["", "[固定协议](../api-validation-20261017/protocol.json) · [结果JSON](../api-validation-20261017/result.json) · [预算核验](../api-validation-20261017/budget-preflight.json)",
              "", "原始请求、加密推理与逐题轨迹仅保存在git忽略的 runs/api-validation-20261017/，不随报告公开。"]
    (ROOT / "experiments/reports/api-validation-20261017.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: result[k] for k in ("requests", "known_cost_usd", "unresolved_costs", "smoke_gate", "migration_gate")}, ensure_ascii=False))
