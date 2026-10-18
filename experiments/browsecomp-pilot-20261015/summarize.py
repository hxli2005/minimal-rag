"""Publish only IDs and aggregate measurements; keep raw evidence under data/."""
import argparse
from collections import Counter
import hashlib
import json
import math

from minimal_rag.codex_adapter import token_usage
from common import DATA, HERE, OUT, ROOT, export, now, parse_judge_response, qrels, read, request_count, sample, save


def wilson(k, n):
    z, p = 1.959963984540054, k / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return [100*(center-half), 100*(center+half)]


def summarize(limit=None):
    ids, refs = sample(limit), qrels()
    rows, generation_runs, judge_runs, router_responses = [], [], [], []
    all_counts, providers = Counter(), set()
    judge_requests = 0
    accepted_zero_reasoning = 0
    for qid in ids:
        trace = read(OUT / 'traces' / f'{qid}.json')
        attempts = [read(path) for path in sorted((OUT / 'attempts').glob(f'{qid}-*.json'))]
        if trace:
            attempts.append(trace)
        counts = Counter()
        calls = 0
        for attempt in attempts:
            generation_runs.extend(attempt['model_runs'])
            calls += len(attempt['model_runs'])
            counts.update(export(attempt)['tool_call_counts'])
        all_counts.update(counts)
        result = export(trace) if trace else {'status': 'not_run', 'retrieved_docids': []}
        judgement = read(OUT / 'judgements' / f'{qid}.json', {})
        for attempt in judgement.get('attempts', []):
            judge_requests += 1
            judge_runs.extend(attempt.get('model_runs', []))
            if 'response' in attempt:
                raw = attempt['response']
                router_responses.append(raw)
                providers.add(raw.get('provider', 'unknown'))
        correct = judgement.get('judgement', {}).get('correct')
        accepted = next((a for a in judgement.get('attempts', []) if a.get('status') == 'completed'), {})
        if accepted.get('reasoning_tokens') == 0 and not accepted.get('reasoning_present'):
            accepted_zero_reasoning += 1
        retrieved = set(result['retrieved_docids'])
        evidence = refs[qid]
        hits = len(retrieved & set(evidence))
        category = None
        if correct is not True:
            status = result['status']
            if status in ['budget_exhausted', 'global_budget_exhausted', 'timeout'] or (trace and len(trace['model_runs']) >= 30):
                category = '预算用尽或超时'
            elif status != 'completed':
                category = '基础设施失败或未完成'
            elif correct is None:
                category = '裁判未判定'
            elif hits == 0:
                category = '证据从没检索到'
            elif hits < len(evidence):
                category = '只检索到部分证据但答错'
            else:
                category = '全部标注证据检索到了但答错'
        rows.append({'query_id': qid, 'status': result['status'], 'correct': correct,
                     'retrieval_recall': hits / len(evidence), 'evidence_hits': hits,
                     'evidence_total': len(evidence), 'tool_call_counts_all_attempts': dict(counts),
                     'generation_calls_all_attempts': calls, 'failure_category': category})
    n, k = len(ids), sum(row['correct'] is True for row in rows)
    completed = sum(row['status'] == 'completed' for row in rows)
    graded = sum(row['correct'] is not None for row in rows)
    usage, judge_usage = token_usage(generation_runs), token_usage(judge_runs)
    router_usage = Counter()
    for raw in router_responses:
        router_usage.update({key: value for key, value in raw.get('usage', {}).items() if isinstance(value, (int, float))})
    costs = [r.get('usage', {}).get('cost') for r in router_responses]
    smoke_response = read(OUT / 'judgements' / f'{sample()[0]}.json', {})
    first_attempt = next((a for a in smoke_response.get('attempts', []) if a.get('status') == 'completed'),
                         next((a for a in smoke_response.get('attempts', []) if 'response' in a), {}))
    first_message = first_attempt.get('response', {}).get('choices', [{}])[0].get('message', {})
    first_parsed = first_attempt.get('parsed') or parse_judge_response(first_message.get('content', ''))
    judge_config = read(OUT / 'judge-config.json', {})
    return {'created': now(), 'denominator': n, 'correct': k, 'accuracy_percent': 100*k/n,
            'score_status': 'no_valid_judgements' if graded == 0 else ('smoke_only' if limit else 'pilot'),
            'accuracy_valid_for_decision': n == 30 and graded == 30,
            'wilson_95_percent': wilson(k,n), 'completed': completed, 'graded': graded,
            'completed_accuracy_percent': 100*k/completed if completed else None,
            'status_counts': dict(Counter(row['status'] for row in rows)),
            'failure_categories': dict(Counter(row['failure_category'] for row in rows if row['failure_category'])),
            'recall_percent': 100*sum(row['retrieval_recall'] for row in rows)/n,
            'mean_search_calls': all_counts['search']/n, 'mean_read_calls': all_counts['read']/n,
            'mean_generation_requests': sum(row['generation_calls_all_attempts'] for row in rows)/n,
            'generation_requests': sum(row['generation_calls_all_attempts'] for row in rows),
            'generation_ledger_requests': request_count(), 'generation_usage': usage,
            'generation_missing_usage_calls': sum(r.get('usage') is None for r in generation_runs),
            'judge_requests': judge_requests, 'judge_sol_usage': judge_usage,
            'accepted_judgements_with_zero_reasoning': accepted_zero_reasoning,
            'judge_requests_without_response': judge_requests - len(router_responses) if judge_config.get('mode') == 'openrouter' else 0,
            'judge_openrouter_usage': dict(router_usage),
            'judge_cost_usd': sum(costs) if costs and all(v is not None for v in costs) else None,
            'judge_cost_estimate_usd': (router_usage['prompt_tokens']*.08 + router_usage['completion_tokens']*.28)/1e6 if router_responses else None,
            'judge_providers': sorted(providers), 'judge': judge_config,
            'smoke': {'parse_correct': first_parsed.get('correct') is not None,
                      'accepted': first_attempt.get('status') == 'completed',
                      'reasoning_present': first_attempt.get('reasoning_present'),
                      'reasoning_chars': len(first_message.get('reasoning') or ''),
                      'reasoning_tokens': first_attempt.get('response', {}).get('usage', {}).get('completion_tokens_details', {}).get('reasoning_tokens'),
                      'provider': first_attempt.get('response', {}).get('provider')},
            'cases': rows,
            'limitations': ['30题固定开发样本，非官方成绩；样本点估计不能用于榜单名次判断。',
                           '失败、超时、未运行和未判定保留在固定分母；未判定是保守未答对，不冒充已判错误。',
                           'SDK内部HTTP重试不可观测；请求数是应用层计数。',
                           '召回按最终尝试检索到的全部docid对官方evidence qrels计算，再逐题平均；工具成本包括恢复前尝试。',
                           'SQLite分块FTS5与官方Lucene BM25不同；模型、工具、检索和预算均非受控单因素对照。']}


def report(s):
    accuracy = s['accuracy_percent']
    tier = '≥90%，建议全量验证' if accuracy >= 90 else ('85%–<90%，先分析错题' if accuracy >= 85 else '<85%，倾向回头主攻 EnterpriseRAG-Bench')
    interval = '–'.join(f'{v:.1f}%' for v in s['wilson_95_percent'])
    total_input = s['generation_usage'].get('input_tokens', 0)
    proxy = s['judge'].get('mode') == 'sol'
    label = 'sol 代理裁判，不能和榜单比较' if proxy else 'Qwen3-32B（OpenRouter/DeepInfra fp8）裁判'
    cost = f"${s['judge_cost_usd']:.6f}" if s['judge_cost_usd'] is not None else '未取得实际账单费用'
    groups = {'证据从没检索到': '未检到证据', '只检索到部分证据但答错': '部分证据答错',
              '全部标注证据检索到了但答错': '全部证据答错', '预算用尽或超时': '预算/超时',
              '基础设施失败或未完成': '未完成', '裁判未判定': '未判定'}
    failures = '、'.join(f'{label}{s["failure_categories"].get(key,0)}题' for key,label in groups.items())
    wrong_ids = '、'.join(row['query_id'] for row in s['cases'] if row['correct'] is False)
    if s['graded'] < 30:
        recommendation = '先解决未完成或未判定原因，再作投入决策；当前保守准确率不能直接用于判定能力。'
    elif accuracy >= 90:
        recommendation = f'建议冻结当前配置跑全量830题，检验90%水平能否保持；按本次均值粗估约{round(s["mean_generation_requests"]*830):,}次sol生成请求。先验证整体水平，再决定是否改检索，公开说明这30题已作为开发样本。'
    elif s['failure_categories'].get('预算用尽或超时', 0):
        recommendation = '先区分预算限制、检索缺口与证据利用失败，不直接把低分归因为检索；暂不投入全量。'
    else:
        recommendation = '先按错因判断是否值得改检索；暂不投入全量，EnterpriseRAG-Bench 继续作为主目标。'
    text = f'''准确率 **{s['correct']}/30 = {accuracy:.2f}%**，Wilson 95% 区间 **{interval}**，落在 **{tier}** 档；{label}，仅用于方向判断。

| 系统 / 口径 | 准确率 | 证据召回 | 平均 search | 平均 read | 平均生成请求 | 输入 token | sol 请求总数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 本次固定 30 题 | {accuracy:.2f}% | {s['recall_percent']:.2f}% | {s['mean_search_calls']:.2f} | {s['mean_read_calls']:.2f} | {s['mean_generation_requests']:.2f} | {total_input:,} | {s['generation_ledger_requests']} |
| GPT-5＋BM25（brief 榜单参照） | 57.59% | — | — | — | — | — | — |
| GPT-5.5＋BM25＋首轮扩展（全量参照） | 90.48% | 96.58% | 33.7 | — | — | — | — |
| GPT-5＋Mixedbread＋get_doc（全量参照） | 90.48% | 83.03% | 11.5 | — | — | — | — |
| AI21，8 agents（全量参照） | 95.18% | 93.17% | 209.3 | — | — | — | — |

- 失败分组：{failures}。错题ID：{wrong_ids or '无'}。检索到文档不代表读到关键正文，证据缺口也不等于唯一错因。
- 运行：完成及判分 {s['completed']}/{s['graded']} 题，固定分母30；生成用量缺失 {s['generation_missing_usage_calls']} 次。请求为应用层计数，SDK内部重试不可见；输入含缓存token {s['generation_usage'].get('cached_input_tokens',0):,}。
- 裁判：实际供应商 {', '.join(s['judge_providers']) or 'Codex'}，调用 {s['judge_requests']} 次（含修复失败），已返回费用 {cost}；{s['judge_requests_without_response']} 次无用量，费用未知。官方空think前缀续写通过双题冒烟，评分正文不变，{s['accepted_judgements_with_zero_reasoning']}/{s['graded']} 个有效判决均为0 reasoning token且内容为空。
- 可比性：SQLite分块BM25、30轮预算、search/read与表中全量系统不同；表格只作参照，不能证明模型或工具的独立贡献。
- 下一步：{recommendation}

口径：{label}、固定 30 题开发样本、非官方成绩。样本与逐题数字见 [summary](../browsecomp-pilot-20261015/summary.json)，[样本 ID](../browsecomp-pilot-20261015/sample.json)；原始轨迹仅存 `data/benchmarks/browsecomp-plus/pilot-20261015/`。外部参照沿用 [基准选型报告](benchmark-selection-20261014.md)。关闭思考依据：[Qwen模板等价校验](../browsecomp-pilot-20261015/judge-template-equivalence.json)、[Qwen官方说明](https://github.com/QwenLM/Qwen3/blob/main/docs/source/inference/transformers.md)、[OpenRouter续写接口](https://openrouter.ai/docs/api-reference/overview#assistant-prefill)。
'''
    (ROOT / 'experiments/reports/browsecomp-pilot-20261015.md').write_text(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    summary = summarize(2 if args.smoke else None)
    name = 'smoke-summary.json' if args.smoke else 'summary.json'
    save(OUT / name, summary)
    if args.publish:
        save(HERE / name, summary)
        if not args.smoke:
            report(summary)
            files = sorted((OUT / 'traces').glob('*.json')) + sorted((OUT / 'judgements').glob('*.json'))
            save(HERE / 'evidence-manifest.json', {'official_revision': '046949032b0328319cc9a02663a759ec601d9402',
                 'data_revisions': read(DATA / 'index.json')['data_revisions'],
                 'files': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}})
    print(json.dumps({key: summary[key] for key in ['denominator','correct','accuracy_percent','wilson_95_percent','completed','graded','recall_percent','generation_ledger_requests','judge_requests','judge_cost_usd']}, ensure_ascii=False))
