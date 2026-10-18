"""Self-contained, escaped HTML for reviewing a brief and its source passages."""
import asyncio
from html import escape
import json
from pathlib import Path
import re


STYLE = """
:root{color-scheme:light;--ink:#172c33;--muted:#607078;--line:#dce5e5;--accent:#136d67;--paper:#fff;--bg:#f2f6f5}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.8 system-ui,-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}
header{background:#142f35;color:#f4f8f6;padding:40px max(24px,calc((100vw - 1180px)/2)) 32px}header p{max-width:880px;color:#d1dfdf}
.eyebrow{font-size:12px;letter-spacing:.18em;color:#81cbc0}h1{font-size:32px;line-height:1.4;margin:12px 0}h2{font-size:19px;margin:0 0 14px}h3{font-size:16px;margin:0}
main{max-width:1228px;margin:auto;padding:24px}.notice{padding:14px 18px;background:#fff5da;border:1px solid #e9d5a0;border-radius:10px;margin-bottom:20px}
.grid{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(300px,1fr);gap:24px;align-items:start}.card{padding:24px;background:var(--paper);border:1px solid var(--line);border-radius:12px;margin-bottom:16px}
.status,.pill{display:inline-block;padding:3px 10px;border-radius:20px;font-size:12px}.status{background:#d9eee7;color:#185944}.pill{background:#eaf1f0;color:var(--accent);margin-right:6px}
.muted{color:var(--muted);font-size:13px}.stats{display:flex;gap:20px;flex-wrap:wrap;margin-top:20px}.stats strong{font-size:23px}.stats span{display:block;font-size:12px;color:#b1ccca}
ul{padding-left:20px;margin-bottom:0}li{margin-bottom:14px}li:last-child{margin-bottom:0}a{color:var(--accent)}a.cite{background:#e7f2ee;border-radius:4px;text-decoration:none;padding:0 5px;font-size:12px;font-weight:650;white-space:nowrap}
code{background:#edf3f1;border-radius:4px;padding:1px 4px;font-size:.9em;overflow-wrap:anywhere}nav{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:20px;font-size:13px}section{scroll-margin-top:20px}
details{scroll-margin-top:20px}summary{cursor:pointer;line-height:1.5;overflow-wrap:anywhere}details[open] summary{margin-bottom:15px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7f6;border:1px solid var(--line);border-radius:8px;padding:14px;font:12px/1.7 ui-monospace,monospace;max-height:520px;overflow:auto}
.source{overflow-wrap:anywhere;font-size:12px;color:var(--muted)}.pending{border-left:3px solid #ceac64;padding-left:12px;list-style:none;margin-left:-20px}.step{padding:9px 0;border-bottom:1px solid var(--line)}.step:last-child{border:0}
.warning{color:#9c442c}footer{padding:10px 0 30px;color:var(--muted);font-size:12px}@media(max-width:850px){.grid{grid-template-columns:1fr}h1{font-size:27px}main{padding:16px}.card{padding:20px}}
"""


def render_html(report):
    validation = report["validation"]
    evidence = {row["id"]: row for row in report["evidence"]}
    cited = validation["cited"]

    def sentence(text):
        # Source/model strings never become markup, scripts or external links.
        safe = escape(text)
        safe = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", safe)
        safe = re.sub(r"\*\*([^*\n]+)\*\*", r"<strong>\1</strong>", safe)
        return re.sub(r"\[(E\d+)\]", lambda m: (
            f'<a class="cite" href="#source-{m[1]}">{m[1]}</a>' if m[1] in evidence else m[0]), safe)

    valid = report["status"] == "draft_for_review"
    sections = "".join(
        f'<section class="card" id="section-{i}"><h2>{escape(name)}</h2><ul>' + "".join(
            f'<li class="{"pending" if item["pending"] else "finding"}">{sentence(item["text"])}</li>'
            for item in items) + '</ul></section>'
        for i, (name, items) in enumerate(validation["sections"].items())) if valid else (
            '<section class="card"><h2>调查未完成</h2><p>本次运行未形成通过来源检查的交接草稿。已取得的证据仍可检查。</p>'
            f'<p class="warning">状态：{escape(report["result"]["status"])}</p></section>')
    sources = ""
    # Include only observed passages; put citations first, then other retrieved text.
    ordered = cited + [ident for ident in evidence if ident not in cited]
    for ident in ordered:
        if ident not in evidence:
            continue
        row = evidence[ident]
        sources += (
            f'<details class="card" id="source-{ident}" {"open" if ident in cited else ""}>'
            f'<summary><span class="pill">{ident}</span>{escape(row["title"])}</summary>'
            f'<p class="source">{escape(row["source"])}<br>原文字符 {row["offset"]}–{row["offset"] + len(row["text"])}'
            f'</p><pre>{escape(row["text"])}</pre></details>')
    steps = []
    for event in report["events"]:
        if event["role"] == "tool":
            label = event["name"] + " " + event["arguments_json"]
            steps.append(f'<div class="step">{escape(label)}</div>')
        elif event["role"] == "protocol_error":
            steps.append('<div class="step warning">模型返回格式错误，已记录。</div>')
    notice = report.get("demonstration_notice", report["note"])
    footer = ("公开基准演示文档属于模拟企业；演示、回归检查与真实用户验收分别记录。"
              if report["mode"].startswith("offline") else
              "资料由调用方提供；本工具未连接现场监控，交接内容与处置条件须人工复核。")
    calls = report.get("generation_calls", report["outer_calls_this_run"])
    navigation = '<nav aria-label="简报导航">' + ''.join(
        f'<a href="#section-{i}">{escape(name)}</a>' for i, name in enumerate(validation["sections"])) + '</nav>' if valid else ''
    title = "故障调查交接单"
    errors = '<ul>' + ''.join(f'<li>{escape(error)}</li>' for error in validation["errors"]) + '</ul>' if validation["errors"] else '结构与引用片段来源检查通过。结论含义、当前适用性与处置授权仍须人工核查。'
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>{title} · Minimal RAG</title><style>{STYLE}</style></head><body>
<header><div class="eyebrow">MINIMAL RAG / SUPPORT → ONCALL</div><h1>{title}</h1>
<p>{escape(report["question"])}</p><span class="status">{"待人工复核" if valid else "调查未完成"}</span>
<div class="stats"><div><strong>{len(cited)}</strong><span>引用的原文片段</span></div><div><strong>{calls}</strong><span>模型外层调用</span></div>
<div><strong>{report["budget"]["used"]} / {report["budget"]["limit"]}</strong><span>{"回放步骤 / 上限" if report["mode"].startswith("offline") else "记录时本批已用 / 上限"}</span></div></div></header>
<main><div class="notice">{escape(notice)}</div>{navigation}<div class="grid"><div>{sections}
<details class="card"><summary>检索与读取记录 · {len(steps)} 步</summary>{''.join(steps)}</details>
<details class="card"><summary>检查范围与运行状态</summary><p>{errors}</p><p class="muted">耗时 {report["seconds"]} 秒。预算计量的是外层适配器调用；不是 token 或订阅额度上限。未执行运维操作。</p></details>
</div><aside><h2>检查原文证据</h2><p class="muted">引用定位到实际返回的片段。片段指纹不表示整篇文档版本或当前事实已获验证。</p>{sources}</aside></div>
<footer>{footer}完整运行记录见同名 JSON 文件。</footer></main></body></html>'''


def save_checkpoint(report, directory):
    path = Path(directory) / "progress.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_failure(error, directory, *, stage):
    """Keep the last checkpoint and make a failed CLI run visibly terminal."""
    try:
        state = json.loads((Path(directory) / "progress.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    if not isinstance(state, dict):
        state = {}
    interrupted = isinstance(error, (KeyboardInterrupt, asyncio.CancelledError))
    status = "interrupted" if interrupted else "failed"
    save_checkpoint({**state, "last_status": state.get("status"), "status": status,
                     "execution_status": status, "stage": stage,
                     "exception": {"type": type(error).__name__, "message": str(error)}}, directory)


def save_report(report, directory, *, prepared=False):
    directory = Path(directory)
    if not prepared:
        directory.mkdir(parents=True, exist_ok=False)
    if any((directory / name).exists() for name in ("report.json", "report.html")):
        raise FileExistsError("previous reports are never overwritten")
    (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (directory / "report.html").write_text(render_html(report), encoding="utf-8")
    return directory / "report.html"
