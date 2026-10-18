"""Build historical snapshots in runs/; the reviewed morning report is maintained separately."""
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo

from budget import LEDGER, MODEL
from minimal_rag.paths import ROOT
from worker import DIRECTORY, OUTPUT, RESULTS, read, save, summarize


def main(final):
    summary = summarize()
    extra_files = {name: OUTPUT / filename for name, filename in (
        ("prefix12", "prefix12-comparison.json"), ("guided_once", "guided-once-summary.json"),
        ("oracle_max", "oracle-max-summary.json"), ("official_compatibility", "official-summary.json"),
        ("exposure", "exposure-audit.json"), ("conjunction", "conjunction-summary.json"),
        ("rrf_grid", "rrf-grid.json"), ("loss_queries", "loss-query-replay.json"), ("citations", "citation-audit.json"), ("requests", "request-reconciliation.json"),
        ("cpu_rerank", "qwen-posthoc-rerank/summary.json"), ("qwen_consumer", "qwen-full-recall/status.json"))}
    extra_files["full_bm25"] = OUTPUT / "full-bm25-calibration/summary.json"
    extra_files["final_checks"] = ROOT / "evidence/overnight-final-checks-20261013.json"
    extra_files["showcase"] = ROOT / "examples/sol-demo/provenance.json"
    extra_files["official_history"] = ROOT / "evidence/official-history-export-20261013.json"
    extra_files["qwen_sample"] = ROOT / "runs/qwen3-chunk-scale-20261012/summary.json"
    extras = {name: read(path) for name, path in extra_files.items() if path.exists()}
    if "official_history" in extras:
        extras["official_history"] = {k: v for k, v in extras["official_history"].items()
                                      if k not in ("document_versions", "cases", "sources")}
    ledger = [json.loads(line) for line in LEDGER.read_text().splitlines()] if LEDGER.exists() else []
    if len(ledger) > 500 or any(row["model"] != MODEL for row in ledger):
        raise RuntimeError("Budget/model invariant failed; do not publish a clean-compliance claim")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    failure = read(ROOT / "evidence/failure-analysis-20261012.json")
    windows = read(OUTPUT / "window-audit.json")
    bm25 = read(OUTPUT / "bm25-summary.json") if (OUTPUT / "bm25-summary.json").exists() else None
    qwen_status = read(ROOT / "runs/qwen3-chunk-scale-20261012/status.json")
    full_paths = [ROOT / "data/benchmarks/enterprise/corpus.qwen3-0.6b.dense.building/progress.json",
                  ROOT / "data/benchmarks/enterprise/corpus.qwen3-0.6b.dense/progress.json"]
    sample_paths = [ROOT / "runs/qwen3-chunk-scale-20261012/qwen.dense.building/progress.json",
                    ROOT / "runs/qwen3-chunk-scale-20261012/qwen.dense/progress.json"]
    progress_paths = full_paths + sample_paths if qwen_status["stage"] in ("building_full", "complete") else sample_paths + full_paths
    qwen_progress = next((read(p) for p in progress_paths if p.exists()), {})
    completed = f"{qwen_progress['completed']:,}" if "completed" in qwen_progress else "未知"
    total = f"{qwen_progress['total']:,}" if "total" in qwen_progress else "未知"
    categories = Counter("judge" if "/judge" in row["label"] or row["label"].startswith("official/") else "generation" for row in ledger)
    qwen_failed = "qwen_sample" in extras and not extras["qwen_sample"]["gate"]["passed"]
    recommendation = "**推荐下一步：保留原提示词，比较工具返回日期、版本、取代状态及目录/章节读取能力；两组各24题、同预算。** Qwen扩库已停止；历史小样本是开发诊断，没有官方SOTA。"
    lines = ["# Minimal RAG 晨报" if final else "# Minimal RAG 进展快照（未收尾）",
        f"\n快照时间：{now:%Y-%m-%d %H:%M} 北京时间。",
        "\n" + recommendation,
        "\n## 1. 失败分析",
        "\n**结论：检索缺失和已读资料后的回答失真都必须解决。** 开发最佳的 22 错题与 CDL 的 48 负题已按保存的裁判解释逐题分类、核对原文暴露区间；关键案例回到原文精读。后者 26 题参考文档不全、21 题参考文档全部曾返回、1 题无参考 ID。semantic 占 20/48。回答主标签是漏细节 17、数值／条件错误 13、错实体 5、额外要求 1、拒答 5、冲突或过度不确定 7。文档曾返回不证明关键事实暴露，标签来自保存的本地裁判解释，不能直接当作因果根因比例。",
        "\n**原文审计确认条件错配和版本歧义，历史分数未改。** qst_0016把合同工30天选成内部员工90天；qst_0248有Finance >15%审批而gold未列；qst_0442的Prefer与有序流程需审定。qst_0413新版确实记录Nina审批，gold对应同ID旧版，不能把这个条款说成模型编造。48道CDL负题有12道完整参考原文已经暴露，文档ID与文本版本要分开。gold-only剩余7错均补齐针对性核对：0240漏host组件但裁判否定的double-counting/eviction确有来源；0369的429 SHOULD与gold强制范围需审定；0351、0376则存在真实遗漏。修正判因没有换成提分。见[逐题分析](../experiments/reports/failure-analysis-20261012.md)。",
        "\n## 2. 同条件 sol 与跨基准",
        "\n**agent的小样本质量收益明确，但调用更多，且没有超过历史最佳同题20/24。** 此次是CPU BM25新诊断，历史最佳混合配置没有重跑。 同模型、同题、同正文库、同盲裁判，闭卷／一次检索各 1 次生成，企业 agent 最多 6 次；金融／HotpotQA agent 最多 5 次，质量收益要和调用／token 成本一起看。分层小样本不代表全基准。",
        "\n| 组 | 计划／完成／计分 | 分数 | 生成调用 | 输入 token（含缓存） |",
        "| --- | --- | --- | ---: | ---: |"]
    for name, group in summary["groups"].items():
        score = f"{group['planned_denominator_score']:.1%}" if group["score_is_final"] else "未完成，不报最终成绩"
        if name.startswith("hotpotqa") and group["score_is_final"]:
            score += f" EM；{group['f1']:.1%} F1"
        lines.append(f"| {name} | {group['planned']}／{group['terminal']}／{group['scored']} | {score} | {group['generation_calls']} | {group['generation_tokens']['input_tokens']:,} |")
    lines += ["\nOracle 组使用已知错题的完整参考文档，与历史观察对照，属于含标签的定位实验，不能当基准成绩。跨基准 HotpotQA 只检索题内给定 distractor 文档，EM/F1 使用官方公式；金融和企业是本地 sol 裁判。",
        "\n## 3. BM25 与片段",
        f"\n**缩窗实测省26%输入但一胜一负，暂不晋级；离线搜索正文字符估计减少 {windows['search_text_char_reduction']:.1%}。** 已审计历史 100 题，{windows['original_search_chars']:,} → {windows['window_search_chars']:,} 字符；词密度窗口不保证关键事实保留，不是实测 token 或费用下降。线上 excerpt=1600 候选和这个估算不是完全同一实现，scope 内窗口去重也可能变化。"]
    if "prefix12" in extras:
        lines.append("\n**默认保留4000字符，固定查询规划器不增加。** 同前12题：自然一次检索6/12、模型选一次5/12、agent10/12、1600窗口10/12。窗口输入961,762→711,762，实测下降26.0%；生成49→52，一胜一负，不能称无损。额外AND优先回放在验证半集没有增益。另48道CDL负题的原题／历史首查文本重放已完成：47题@20为46.50%／43.84%，4胜2负41平；semantic20首查仅15%／10%。这不是整个agent的最终召回，不把原题回补升成固定流程。[同题资源对照](../experiments/reports/sol-prefix12-20261013.md)。")
    if "oracle_max" in extras:
        maximum = extras["oracle_max"]
        lines.append(f"\n**max 强度没有改变这批 gold-only 正确数。** high5/12，max{maximum['correct']:g}/{maximum['planned']}（完成{maximum['terminal']}），逐题正确性完全一致。此次结果不支持仅提高推理强度；还需处理条件绑定和参考版本口径，不逐题挑最好答案。")
    if "cpu_rerank" in extras:
        rerank = extras["cpu_rerank"]
        if rerank.get("complete"):
            lines.append("\n**事后CPU重排候选拒绝。** 100题全部完成，semantic@20=76%、整体94题=25.80%，低于BM25的80%／26.86%，也低于RRF。现有前100+100候选上限96%不是实现成绩；没有改原gate或全量构建。")
    if bm25:
        lines += [f"\n**BM25 重放已完成 {bm25['records']}/496 条记录；晋级必须看预先冻结的验证半集。** 100 条历史首次查询＋24 条自然问题，对照原标题权重 2、0、8、去停用词，共四个固定变体。只统计有参考标签的题。",
                  "\n| 验证半集的历史首次查询 | n | recall@5 | recall@20 |", "| --- | ---: | ---: | ---: |"]
        for key, row in bm25["groups"].items():
            if "/validation/" in key:
                lines.append(f"| {key.rsplit('/',1)[-1]} | {row['n_with_gold']} | {row['recall5']:.1%} | {row['recall20']:.1%} |")
    else:
        lines.append("\n**BM25 重放尚未完成，暂不宣布新默认。** 每条记录可续跑，完整结果在本地 bm25/ 目录。")
    lines += ["\n## 4. Qwen 与官方入口",
        (f"\n**Qwen冻结gate未通过，停止全量构建。** semantic@20 BM25 80%→混合84%，仅+4pp<+10pp门槛，1胜0负24平；纯Qwen52%、BGE24%。整体94题26.86%→27.93%未退步，但该抽样库有100次非semantic参考出现缺失，不能作全库整体保护证据。原PID12453自然结束，没有发信号或重启；自己的等待consumer已停止。全量CPU入口保留备用，**没有全量索引或全库召回结果**。另12组缓存RRF参数按既有选择半集选出k100：选择semantic7/9→8/9，验证16题仍14/16、整体验证也不变，不晋级。" if qwen_failed else
         f"\n**Qwen当前实际阶段是 `{qwen_status['stage']}`。** 最近检查点{completed}／{total}块；全库召回以完成后summary为准。已准备单线程CPU入口，复用原查询向量，不加载编码器、不写向量。"),
        "\n**全库规模校准明显更难，抽样结果不能外推。** 同semantic25自然问题、同BM25配置与返回预算，全库recall@20=44%，抽样80%，相差36pp；没有全库Qwen混合成绩。",
        "\n**官方提示词入口已准备，但完整官方评测尚未宣称复现。** 固定官方版本 d36685e273713975ee20299bbf1ab64165575b3c，支持全部计划题导出 JSONL，缺失题保留空答案；固定前4题的正确性与逐事实兼容评分全部通过，共29次请求，不是完整评测。另将历史dev100＋test400的500/500原答案完整导出，899个本地引用正文／898个官方UUID／905个含替代版本的映射，未知或未观察引用为0；905/905正文版本已绑定发行Parquet行号，文件SHA与发行manifest一致。保留原文字，不新增生成或分数。0413两个本地版本合为一个官方UUID，仍需对齐物理来源。本地兼容评分使用 sol、冻结参考，不做三裁判文档校正、参考答案重写或模型去引用。[运行入口与边界](runbook.md#2026-10-12-夜间实验)。",
        "\n## 5. 极简与求职展示",
        "\n**核心循环和默认检索器保持克制，展示材料已有真实 sol 证据。** 调查账本从core迁入实验包，逐片段SHA和显示移除，保留真实bug的原文区间检查；loop未改。本晚实验隔离在日期目录；README、一页设计、面试讲解、真实运行与70题失败分类更新。完整库单例3次生成＋1次评分、5/5参考事实；可克隆5篇原文小库另有3次sol生成录制、数值人工核对，无新增裁判。两者都只作展示，不作基准成绩。新增自带录制数据的可视报告，3步切换、完整答案展开、桌面／320与360窄屏／深色主题实际核对，页面无横向溢出，fresh控制台0错误。[可视回放](../examples/sol-demo/report.html)。28项相关检查和55个核心测试、独立core wheel demo、实验离线demo通过；一次性实验没有新建测试套件。[真实运行](real-sol-run-20261012.md) · [可克隆5篇原文](../examples/sol-demo/README.md) · [面试讲解](interview.md)。",
        "\n## 预算与证据",
        f"\n**累计预扣 {len(ledger)}/500 次实验外层请求，全部为 gpt-6.1-sol。** 生成 {categories['generation']}、评分／官方兼容评分 {categories['judge']}；失败和重试也计入，预扣但未完成传输不退款。本聊天没有启动GPU编码或重排，原Qwen自然完成；没有向main提交。此计数是实验 SDK 外层请求，不是本聊天内部推理轮数或额度百分比。"]
    state = read(DIRECTORY / "state.json")
    if "requests" in extras and "usage" in extras["requests"]:
        usage = extras["requests"]["usage"]["all"]
        lines.append(f"\n已核账的累计输入{usage['input_tokens']:,} token（其中缓存{usage['cached_input_tokens']:,}），输出{usage['output_tokens']:,}；每个会话取最后一次累计值，缓存不重复相加。不是费用或订阅额度估计。所有498个请求均保留completed记录、用量无缺失。")
    if state.get("last_error"):
        lines.append(f"\n已记录的运行错误：`{state['last_error']}`；不能把中止或缺少评分的任务标成成功。")
    report = OUTPUT / "report-final-snapshot.md" if final else OUTPUT / "report-draft.md"
    report.write_text("\n".join(lines) + "\n")
    if final:
        source_paths = [OUTPUT / "summary.json", LEDGER, OUTPUT / "failure-analysis.json", OUTPUT / "window-audit.json", ROOT / "evidence/reference-audit-20261012.json", ROOT / "evidence/current-errors-20261013.json", ROOT / "evidence/validation.json"]
        source_paths += [OUTPUT / "bm25-summary.json"] if bm25 else []
        source_paths += [path for path in extra_files.values() if path.exists()]
        save(ROOT / "evidence/overnight-summary-20261013.json", {**summary, "reported_beijing": now.isoformat(),
            "additional_experiments": extras,
            "budget_unit": "pre-reserved experiment outer SDK requests including retries",
            "calls_by_role": dict(categories), "qwen_status": qwen_status, "qwen_progress": qwen_progress,
            "bm25": bm25, "window_estimate": {k: v for k, v in windows.items() if k != "cases"},
            "sources": [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in source_paths if p.exists()]})
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    main(args.final)
