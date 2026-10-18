"""ID-only results, paired bootstrap, frozen selection and official submission."""
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import hashlib
import json
import numpy as np
from minimal_rag.codex_adapter import token_usage
from shared import DATA, HERE, ROOT, OUT, LIMITS, count, directory, export, ids, now, pilot, read, save
from summarize import wilson


def rows(group, selected):
    folder, refs = directory(group), pilot.qrels()
    ledger = Counter()
    for stage in LIMITS:
        path = OUT / f'{stage}-requests.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                entry = json.loads(line)
                if entry['group'] == group:
                    ledger[entry['query_id']] += 1
    result = []
    for qid in selected:
        trace = read(folder / 'traces' / f'{qid}.json')
        attempts = [read(p) for p in (folder / 'attempts').glob(f'{qid}-*.json')]
        if trace:
            attempts.append(trace)
        runs = [run for attempt in attempts for run in attempt['model_runs']]
        tools = Counter()
        for attempt in attempts:
            tools.update(export(attempt)['tool_call_counts'])
        official = export(trace) if trace else {'status': 'not_run', 'retrieved_docids': []}
        judge = read(folder / 'judgements' / f'{qid}.json', {})
        judged = judge.get('judgement', {})
        accepted = next((a for a in judge.get('attempts', []) if a.get('status') == 'completed'), {})
        if judged.get('correct') is not None:
            assert accepted.get('reasoning_tokens') == 0 and not accepted.get('reasoning_present')
            assert accepted['response']['provider'] == 'DeepInfra'
        usage = token_usage(runs)
        calls = ledger[qid]
        if group == 'A' and qid in ids('pilot30'):
            calls += len(trace['model_runs']) if trace else 0
        hits = len(set(official['retrieved_docids']) & set(refs[qid]))
        category, status = None, official['status']
        if judged.get('correct') is not True:
            if status in ('budget_exhausted', 'stage_budget_exhausted') or (trace and len(trace['model_runs']) >= 30):
                category = '预算用尽'
            elif status != 'completed':
                category = '未运行' if status == 'not_run' else '基础设施失败'
            elif judged.get('correct') is None:
                category = '裁判未判定'
            else:
                category = '证据从没检索到' if hits == 0 else ('部分检索到' if hits < len(refs[qid]) else '全部检索到但答错')
        responses = [a['response'] for a in judge.get('attempts', []) if 'response' in a]
        result.append({'query_id': qid, 'status': status, 'correct': judged.get('correct'),
             'terminal': bool(trace and trace.get('result') and not trace.get('recoverable')),
             'confidence': judged.get('confidence'), 'recall': hits / len(refs[qid]),
             'evidence_hits': hits, 'evidence_total': len(refs[qid]), 'search': tools['search'], 'read': tools['read'],
             'generation_requests': calls, 'input_tokens': usage.get('input_tokens', 0),
             'cached_input_tokens': usage.get('cached_input_tokens', 0),
             'uncached_input_tokens': usage.get('input_tokens', 0)-usage.get('cached_input_tokens', 0),
             'missing_usage_calls': max(0, calls-len(runs)) + sum(r.get('usage') is None for r in runs),
             'seconds': sum(a.get('seconds', 0) for a in attempts), 'failure_category': category,
             'judge_requests': len(judge.get('attempts', [])),
             'judge_response_attempts': sum('response' in a for a in judge.get('attempts', [])),
             'judge_backfill_attempts': sum(bool(a.get('backfill')) for a in judge.get('attempts', [])),
             'judge_cost_usd_returned': sum(r.get('usage', {}).get('cost') or 0 for r in responses),
             'judge_requests_without_cost': len(judge.get('attempts', []))-sum(r.get('usage', {}).get('cost') is not None for r in responses)})
    return result


def calib_err(confidence, correct, p='2', beta=100):
    # Exact arithmetic/loop boundaries from official 0469490 evaluate_run.py.
    idxs = np.argsort(confidence)
    confidence, correct = confidence[idxs], correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    bins[-1] = [bins[-1][0], len(confidence)]
    cerr, total_examples = 0, len(confidence)
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0]:bins[i][1]]
        bin_correct = correct[bins[i][0]:bins[i][1]]
        num_examples_in_bin = len(bin_confidence)
        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))
            if p == '2':
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == '1':
                cerr += num_examples_in_bin / total_examples * difference
            elif p in ('infty', 'infinity', 'max'):
                cerr = np.maximum(cerr, difference)
            else:
                raise ValueError(p)
    return np.sqrt(cerr) if p == '2' else cerr


def aggregate(values):
    n, k = len(values), sum(r['correct'] is True for r in values)
    if not n:
        return {'denominator': 0}
    result = {'denominator': n, 'correct': k, 'accuracy_percent': k/n*100,
              'wilson_95_percent': wilson(k,n), 'graded': sum(r['correct'] is not None for r in values),
              'completed': sum(r['status'] == 'completed' for r in values),
              'terminal': sum(r['terminal'] for r in values),
              'recall_percent': sum(r['recall'] for r in values)/n*100,
              'status_counts': dict(Counter(r['status'] for r in values)),
              'failure_categories': dict(Counter(r['failure_category'] for r in values if r['failure_category'])),
              'calibration_error_percent': None}
    for key in ['search','read','generation_requests','input_tokens','cached_input_tokens','uncached_input_tokens','seconds']:
        result['mean_'+key] = sum(r[key] for r in values)/n
    for key in ['generation_requests','missing_usage_calls','judge_requests','judge_response_attempts','judge_backfill_attempts','judge_cost_usd_returned','judge_requests_without_cost']:
        result[key] = sum(r[key] for r in values)
    if n >= 100 and all(r['confidence'] is not None and r['correct'] is not None for r in values):
        result['calibration_error_percent'] = 100*float(calib_err(np.array([r['confidence']/100 for r in values]),
                                                                np.array([float(r['correct']) for r in values]), beta=100))
    return result


def paired(left, right):
    right_by_id = {r['query_id']: r for r in right}
    def usable(row):
        return row['terminal'] and (row['correct'] is not None or row['status'] != 'completed')
    pairs = [(r,right_by_id[r['query_id']]) for r in left
             if r['query_id'] in right_by_id and usable(r) and usable(right_by_id[r['query_id']])]
    result = {'planned': len(left), 'paired': len(pairs), 'bootstrap_samples': 10000, 'seed': 20261015,
              'wins': sum(a['correct'] is True and b['correct'] is not True for a,b in pairs),
              'losses': sum(b['correct'] is True and a['correct'] is not True for a,b in pairs)}
    result['ties'] = len(pairs)-result['wins']-result['losses']
    if not pairs:
        return result
    samples = np.random.default_rng(20261015).integers(0,len(pairs),size=(10000,len(pairs)))
    for key in ['accuracy_pp','recall_pp','search','uncached_input_tokens']:
        if key == 'accuracy_pp':
            delta = np.array([100*(int(a['correct'] is True)-int(b['correct'] is True)) for a,b in pairs])
        elif key == 'recall_pp':
            delta = np.array([100*(a['recall']-b['recall']) for a,b in pairs])
        else:
            delta = np.array([a[key]-b[key] for a,b in pairs])
        result[key] = {'mean_difference': float(delta.mean()),
                       'bootstrap_95': np.percentile(delta[samples].mean(axis=1),[2.5,97.5]).tolist()}
    result['query_ids'] = [a['query_id'] for a,b in pairs]
    return result


def selection():
    a, lean = rows('A',ids('dev100')), rows('A-lean',ids('dev100'))
    sa, sl, comparison = aggregate(a), aggregate(lean), paired(lean,a)
    complete = (comparison['paired'] == 100 and sa['missing_usage_calls'] == sl['missing_usage_calls'] == 0)
    reduction = 1-sl['mean_uncached_input_tokens']/sa['mean_uncached_input_tokens']
    passed = complete and reduction >= .4 and comparison['losses']-comparison['wins'] <= 2
    value = {'created': now(), 'selected': 'A-lean' if passed else 'A', 'passed': passed,
             'complete_pairs_and_usage': complete, 'uncached_reduction_fraction': reduction if complete else None,
             'A': sa, 'A-lean': sl, 'paired_lean_minus_A': comparison,
             'reason': 'Both predefined gates passed' if passed else 'Gate not passed; preserve A',
             'per_query': [{'query_id': x['query_id'], 'A': x, 'A-lean': y} for x,y in zip(a,lean)]}
    save(HERE / 'selection.json', value)
    return value


def snapshot(publish=False):
    chosen = read(HERE / 'selection.json', {}).get('selected')
    groups = {'A-dev100': rows('A',ids('dev100')), 'A-lean-dev100': rows('A-lean',ids('dev100')),
              'B30': rows('B',ids('pilot30')), 'B200': rows('B',ids('b200'))}
    groups['B230'] = groups['B30'] + groups['B200']
    comparisons = {'A-lean_minus_A_dev100': paired(groups['A-lean-dev100'], groups['A-dev100']),
                   'A_minus_B30': paired(rows('A',ids('pilot30')),groups['B30'])}
    if chosen:
        groups['full830'] = rows(chosen,ids('all830'))
        groups['heldout730'] = rows(chosen,ids('remaining730'))
        for label, sample_names in [('30',['pilot30']),('200',['b200']),('230',['pilot30','b200'])]:
            selected = sum([ids(name) for name in sample_names], [])
            comparisons['frozen_minus_B'+label] = paired(rows(chosen,selected),rows('B',selected))
    s = {'created': now(), 'selected': chosen, 'stage_requests': {k:count(k) for k in LIMITS},
         'stage_limits': LIMITS, 'groups': {key: aggregate(value) for key,value in groups.items()},
         'comparisons': comparisons, 'rows': groups}
    save(OUT / 'summary.json', s)
    if publish:
        save(HERE / 'summary.json', s)
        write_report(s)
    return s


def write_report(s):
    def metric_row(name, a):
        return f"| {name} ({a['graded']}/{a['denominator']}判分) | {a['accuracy_percent']:.2f}% | {a['recall_percent']:.2f}% | {a['mean_search']:.2f} | {a['mean_read']:.2f} | {a['mean_uncached_input_tokens']:,.0f} |"
    def pair_text(p):
        if not p or not p.get('paired'):
            return '尚无有效配对'
        m = p['accuracy_pp']
        lo,hi = m['bootstrap_95']
        return f"{p['paired']}/{p['planned']}题配对差{m['mean_difference']:+.2f}pp，95%区间[{lo:+.2f}, {hi:+.2f}]，胜/负/平={p['wins']}/{p['losses']}/{p['ties']}"
    full = s['groups'].get('full830')
    selection_result = read(HERE / 'selection.json', {})
    complete = full and full['terminal'] == 830 and full['graded'] == full['completed']
    full_text = (f"全量830题保守准确率 **{full['accuracy_percent']:.2f}%**，Wilson95%区间 **[{full['wilson_95_percent'][0]:.2f}, {full['wilson_95_percent'][1]:.2f}]%**"
                 if complete else '全量830题成绩尚未完成')
    cost = selection_result.get('uncached_reduction_fraction')
    if cost is not None:
        lean_text = f"选择{s['selected']}，dev100降本候选每题未缓存输入变化{-100*cost:+.2f}%"
    else:
        a,lean=s['groups']['A-dev100'],s['groups']['A-lean-dev100']
        recorded=1-lean['mean_uncached_input_tokens']/a['mean_uncached_input_tokens']
        lean_text=f"A-lean dev100已测93→96题正确，已记录未缓存输入下降{recorded*100:.1f}%；基线用量有缺失，不是精确节省率，未通过原40%采用门槛"
    p = s['comparisons'].get('frozen_minus_B230') or s['comparisons']['A_minus_B30']
    text = f"{full_text}；对照：{pair_text(p)}；{lean_text}。\n\n"
    text += '| 系统 | 准确率 | 召回率 | 平均search | 平均read | 每题未缓存输入token |\n|---|---:|---:|---:|---:|---:|\n'
    for name in ['full830','B230','A-dev100','A-lean-dev100']:
        if name in s['groups']:
            text += metric_row((s['selected']+' 全量') if name == 'full830' else name, s['groups'][name])+'\n'
    text += '| GPT-5＋BM25（brief历史参照） | 57.59% | — | — | — | — |\n'
    text += '| GPT-5.5＋BM25＋查询扩展（单agent历史参照） | 90.48% | 96.58% | 33.7 | — | — |\n'
    text += '| GPT-5＋Mixedbread＋get_doc（单agent历史参照） | 90.48% | 83.03% | 11.5 | — | — |\n'
    text += '| AI21，8 agents（总榜历史参照） | 95.18% | 93.17% | 209.3 | — | — |\n\n'
    position = '仅比较历史参照，不能认定当前榜单排名；本系统Wilson区间与他方点估计不构成配对显著性检验'
    text += f'- 榜单：{position}；参照为brief提供的历史榜单，搜索次数不同。'
    if full:
        held = s['groups']['heldout730']
        text += f" 其余730题已判{held['graded']}题，固定分母保守准确率{held['accuracy_percent']:.2f}%。"
    text += '\n- 项目贡献：工具/返回文本/提示组合的配对差，核心循环相同，不能识别循环的独立贡献。'
    for key in ['frozen_minus_B30','frozen_minus_B200','frozen_minus_B230']:
        if key in s['comparisons']:
            text += f" {key.removeprefix('frozen_minus_B')}题：{pair_text(s['comparisons'][key])}。"
    if not s['selected']:
        text += ' A对B30：'+pair_text(s['comparisons']['A_minus_B30'])+'。'
    text += '\n- 降本：'+lean_text+'；仅测试800字符这一候选，调用与token均包含失败尝试；未返回用量的请求不假定免费。'
    if full:
        calibration=full['calibration_error_percent']
        text += f" 全量每题输入{full['mean_input_tokens']:,.0f}、缓存{full['mean_cached_input_tokens']:,.0f}；校准误差{str(calibration)+'%' if calibration is not None else '不可用'}（该本地汇总仍有缺失置信度；官方复核另行报告）。"
    failures = full['failure_categories'] if full else s['groups']['B30']['failure_categories']
    text += '\n- 错题/缺失状态：'+json.dumps(failures,ensure_ascii=False)+'；未运行和未判定单列，不能视为能力错误。'
    text += '\n- 下一步：'+('仅准备提交文件，先由维护方核实口径；不凭开发样本点估计宣称SOTA。' if complete else '按冻结计划从断点完成；若触达预算或额度限制，保留已完成结果，不改配置挤预算。')+'\n\n'
    text += '口径：Qwen3-32B（OpenRouter/DeepInfra fp8）裁判；dev100开发样本已含在全量中；B是官方标准工具配本项目检索；未经维护方核实。表中未完成组使用固定分母保守值，不代表已完成成绩。应用层请求账本：'+json.dumps(s['stage_requests'])+'，SDK内部重试不可见。详见[逐题和配对表](../browsecomp-full-20261015/summary.json)、[执行记录](../browsecomp-full-20261015/notes.md)。\n'
    (ROOT / 'experiments/reports/browsecomp-full-20261015.md').write_text(text)


def submission():
    s = snapshot(publish=True)
    if 'full830' not in s['groups']:
        return False
    full = s['groups']['full830']
    if full['terminal'] != 830 or full['graded'] != full['completed'] or full['calibration_error_percent'] is None:
        return False
    value = {'LLM': 'gpt-6.1-sol (effort high)',
             'Retriever': 'BM25 (SQLite FTS5, 1600-char chunks) + search/read tools',
             'Accuracy (%)': round(full['accuracy_percent'],2), 'Recall (%)': round(full['recall_percent'],2),
             'Search Calls': full['mean_search'], 'avg_tool_stats': {'search':full['mean_search'],'read':full['mean_read']},
             'Calibration Error (%)': round(full['calibration_error_percent'],2), 'Link': '',
             'Evaluation Date': datetime.now(ZoneInfo('America/Los_Angeles')).date().isoformat(),
             'per_query_metrics': [{'query_id':r['query_id'], 'correct':r['correct'] is True,
                                    'recall':round(100*r['recall'],2)} for r in s['rows']['full830']]}
    save(OUT / 'submission.json',value)
    save(HERE / 'submission.json',value)
    paths = list(directory(s['selected']).glob('traces/*.json'))+list(directory(s['selected']).glob('judgements/*.json'))
    save(HERE / 'evidence-manifest.json', {'created':now(),'files':
         {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}})
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--select',action='store_true')
    parser.add_argument('--submission',action='store_true')
    args = parser.parse_args()
    if args.select:
        print(json.dumps({k:v for k,v in selection().items() if k != 'per_query'}))
    if args.submission:
        print({'submission_ready':submission()})
    s = snapshot(args.publish)
    print(json.dumps({'stage_requests':s['stage_requests'],'groups':
                     {k:{f:v[f] for f in ['denominator','completed','graded','correct','generation_requests']} for k,v in s['groups'].items()}}))
