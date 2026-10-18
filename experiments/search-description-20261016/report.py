"""Compare complete, paired Enterprise runs with only search wording changed."""
from collections import Counter
import json
from pathlib import Path
import random
import statistics

from minimal_rag.paths import ROOT
from minimal_rag.codex_adapter import token_usage
from minimal_rag_experiments.hotpot_benchmark import save, sha256

HERE = Path(__file__).resolve().parent
OLD = ROOT / 'runs/bm25-unified-20261016/enterprise-test'
NEW = ROOT / 'runs/search-description-20261016/enterprise-test'


def usage(folder):
    result = {}
    for path in (folder / 'traces').glob('*.json'):
        t = json.loads(path.read_text())
        calls = Counter(e['name'] for e in t['events'] if e.get('role') == 'tool')
        searches = [json.loads(e['arguments_json']) for e in t['events'] if e.get('role') == 'tool' and e['name'] == 'search']
        result[t['id']] = {'search': calls['search'], 'read': calls['read'], 'model_calls': len(t['model_runs']),
                          'first_query': searches[0]['query'] if searches else '',
                          'first_query_words': len(searches[0]['query'].split()) if searches else 0,
                          'budget12': len(t['model_runs']) >= 12}
    return result


def paired(values):
    rng = random.Random(20261016)
    draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(10000))
    return {'mean_delta': statistics.mean(values), 'paired_bootstrap_95': [draws[249], draws[9749]]}


def main():
    old = json.loads((OLD / 'summary.json').read_text())
    new = json.loads((NEW / 'summary.json').read_text())
    if not new['complete']:
        print(json.dumps({'completed': new['completed'], 'graded': new['metrics']['graded'], 'requested': new['requested']}))
        return 1
    assert old['complete'] and old['requested'] == new['requested'] == 400
    assert {k: v for k, v in old['config'].items() if k != 'source_sha256'} == {k: v for k, v in new['config'].items() if k != 'source_sha256'}
    assert {k for k in old['config']['source_sha256'] if old['config']['source_sha256'][k] != new['config']['source_sha256'][k]} == {'indexed_rag.py'}
    a, b = ({r['id']: r for r in s['cases']} for s in (old, new))
    assert a.keys() == b.keys()
    old_use, new_use = usage(OLD), usage(NEW)
    assert a.keys() == old_use.keys() == new_use.keys()
    ids = sorted(a)
    wins = [q for q in ids if b[q]['correct'] > a[q]['correct']]
    losses = [q for q in ids if b[q]['correct'] < a[q]['correct']]
    correct = {'baseline': int(sum(a[q]['correct'] for q in ids)), 'candidate': int(sum(b[q]['correct'] for q in ids)),
               'wins': wins, 'losses': losses, 'ties': 400 - len(wins) - len(losses),
               **paired([100 * (b[q]['correct'] - a[q]['correct']) for q in ids])}
    behavior = {}
    for metric in ('search', 'read', 'model_calls'):
        before, after = sum(old_use[q][metric] for q in ids), sum(new_use[q][metric] for q in ids)
        behavior[metric] = {'baseline_total': before, 'candidate_total': after,
                            'baseline_mean': before / 400, 'candidate_mean': after / 400,
                            'change_pct': (after / before - 1) * 100,
                            **paired([new_use[q][metric] - old_use[q][metric] for q in ids])}
    behavior['first_query_words_mean'] = {'baseline': statistics.mean(old_use[q]['first_query_words'] for q in ids),
                                           'candidate': statistics.mean(new_use[q]['first_query_words'] for q in ids)}
    behavior['budget12_cases'] = {'baseline': sum(old_use[q]['budget12'] for q in ids), 'candidate': sum(new_use[q]['budget12'] for q in ids)}
    recovery = json.loads((NEW / 'recovery.json').read_text()) if (NEW / 'recovery.json').exists() else None
    overhead = Counter()
    if recovery:
        for attempt in recovery['interrupted_generation_attempts']:
            trace = json.loads((ROOT / attempt['path']).read_text())
            overhead['generation'] += len(trace['model_runs'])
            overhead.update(e['name'] for e in trace['events'] if e.get('role') == 'tool')
            overhead.update(token_usage(trace['model_runs']))
    actual_requests = dict(new['requests'])
    actual_requests['generation'] += overhead['generation']
    result = {'change': json.loads((HERE / 'change.json').read_text()), 'requested': 400, 'complete': True,
              'decision': 'Keep the revised search description: fewer searches and model calls with higher observed accuracy. Accuracy improvement remains uncertain because the paired 95% interval crosses zero.',
              'accuracy': correct, 'behavior': behavior, 'baseline_metrics': old['metrics'], 'candidate_metrics': new['metrics'],
              'baseline_requests': old['requests'], 'candidate_requests': new['requests'],
              'recovery': recovery, 'interrupted_generation_overhead': dict(overhead),
              'actual_requests_including_interruption': actual_requests,
              'by_category': {c: {'baseline': old['by_category'][c], 'candidate': new['by_category'][c]} for c in old['by_category']},
              'baseline_summary': str((OLD / 'summary.json').relative_to(ROOT)), 'baseline_summary_sha256': sha256(OLD / 'summary.json'),
              'candidate_summary': str((NEW / 'summary.json').relative_to(ROOT)), 'candidate_summary_sha256': sha256(NEW / 'summary.json'),
              'candidate_config': new['config'], 'limits': 'Same local semantic judge, original first valid verdicts; application requests include failures, not SDK-internal HTTP retries.'}
    save(HERE / 'summary.json', result)
    save(ROOT / 'evidence/search-description-20261016.json', result)
    acc, searches = correct, behavior['search']
    lines = ['# 仅修改搜索工具说明：Enterprise test400对比', '',
             f"准确率：{acc['baseline']}/400（{acc['baseline']/4:.2f}%）→ **{acc['candidate']}/400（{acc['candidate']/4:.2f}%）**，变化{acc['mean_delta']:+.2f}pp；配对{len(wins)}胜{len(losses)}负，95%区间{acc['paired_bootstrap_95']}pp。", '',
             f"搜索次数：{searches['baseline_total']}→ **{searches['candidate_total']}**（{searches['change_pct']:+.2f}%）；每题{searches['baseline_mean']:.3f}→{searches['candidate_mean']:.3f}次。", '',
             '**建议保留新版搜索说明。** 本轮搜索和模型调用明显减少，观察到的准确率也更高；但正确率差的95%区间跨过零，不能据单次实验确认稳定提分。没有追加第二版说明或额外实验。', '',
             '## 唯一改动', '',
             '只改`indexed_tools`中`search`说明：允许完整问题或保留对象、时间、地区、条件的查询；解释过短关键词和过严all_terms/短语限制的风险；由模型根据结果相关性选择换词、放宽过滤或探索分页。没有固定搜索步骤、次数、强制首搜或新增执行分支。', '',
             'AST核对确认只去掉search说明字符串后源码完全一致。BM25、索引、工具schema、默认值、返回字段、read说明和实现、核心循环、系统提示、适配器、评分代码都未改。138项测试通过。完整前后说明见[change.json](../search-description-20261016/change.json)。', '',
             '## 同口径结果', '', '| 指标 | 原说明 | 新说明 | 变化 |', '| --- | ---: | ---: | ---: |',
             f"| 正确率 | {acc['baseline']/4:.2f}% | {acc['candidate']/4:.2f}% | {acc['mean_delta']:+.2f}pp |"]
    for key, label in [('search', '搜索次数'), ('read', '阅读次数'), ('model_calls', '生成模型调用')]:
        row = behavior[key]; lines.append(f"| {label} | {row['baseline_total']} | {row['candidate_total']} | {row['change_pct']:+.2f}% |")
    for key, label in [('input_tokens', '生成输入token（含缓存）'), ('output_tokens', '生成输出token')]:
        before, after = old['metrics'][key], new['metrics'][key]
        lines.append(f"| {label} | {before:,} | {after:,} | {(after/before-1)*100:+.2f}% |")
    lines += [f"| 用满12轮题数 | {behavior['budget12_cases']['baseline']} | {behavior['budget12_cases']['candidate']} | — |", '',
              f"首条查询平均词数从{behavior['first_query_words_mean']['baseline']:.2f}增至{behavior['first_query_words_mean']['candidate']:.2f}。参考文档观察召回从{old['metrics']['retrieved_document_recall']*100:.2f}%增至{new['metrics']['retrieved_document_recall']*100:.2f}%。semantic题正确数58→68，贡献净增10题；这是本轮实际行为和结果，不是固定搜索流程。", '',
              f"新组生成和判分均完成400/400，运行错误{new['metrics']['errors']}，工具错误{new['metrics']['tool_errors']}；生成请求{new['requests']['generation']}、裁判请求{new['requests']['judge']}。", '',
              f"上表比较最终400份答案的调用行为。用户手动停止产生的5份未完成尝试已单独保留；恢复时只重新完成这5题，64份完整答案和46份有效判决没有重跑。中断额外消耗{overhead['generation']}次生成调用、{overhead['search']}次搜索、{overhead['read']}次阅读；计入中断后的实际生成调用为{actual_requests['generation']}，裁判调用为{actual_requests['judge']}（包含保留的失败裁判尝试）。", '',
              '相同400题、完整原索引、sol high生成、4000字符摘录、12轮、生成并发5、独立裁判并发1；沿用sol medium裁判和首次有效判决。两份config逐字段比对，只有indexed_rag.py的来源摘要不同。没有调整其他基准或重写原分数。', '',
              '## 分题型', '', '| 类型 | 题数 | 原说明正确 | 新说明正确 |', '| --- | ---: | ---: | ---: |']
    for c in old['by_category']:
        before, after = old['by_category'][c], new['by_category'][c]; n = before['n']
        lines.append(f"| {c} | {n} | {round(before['correct']*n)} | {round(after['correct']*n)} |")
    lines += ['', '## 逐题胜负', '', '改善：' + '、'.join(wins), '', '回退：' + '、'.join(losses), '',
              '所有逐题答案、search/read轨迹和判决保存在`runs/search-description-20261016/enterprise-test/`；对照位于`runs/bm25-unified-20261016/enterprise-test/`。汇总见[summary.json](../search-description-20261016/summary.json)，命令见[brief](../search-description-20261016/brief.md)。区间为按题配对bootstrap 10,000次，分数为本地裁判口径。']
    (ROOT / 'experiments/reports/search-description-20261016.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'accuracy': correct, 'behavior': behavior, 'requests': new['requests']}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
