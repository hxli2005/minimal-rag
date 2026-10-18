"""Summarize completed paired outcomes without another judge or generation."""
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from minimal_rag.codex_adapter import token_usage
from minimal_rag_experiments.evidence_summary import publish

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'runs/complete-answer-20261013'


def completeness(row):
    if row['result']['status']!='finished':return 0
    facts=row['judgement']['facts_supported']
    return statistics.mean(facts) if facts else 1


def main():
    result=json.loads((OUTPUT/'summary.json').read_text())
    if not result['complete']:raise RuntimeError('All48 planned outcomes must be terminal first')
    cohort=json.loads((ROOT/'experiments/overnight-20261012/cohort.json').read_text())['datasets']['enterprise']
    fact_counts={c['id']:len(c['facts']) for c in cohort}
    by_arm={arm:{r['id']:r for r in result['cases'][arm]} for arm in ['concise','complete']}
    pairs=[]
    for c in cohort:
        a,b=[by_arm[arm][c['id']] for arm in ['concise','complete']]
        pairs.append({'id':c['id'],'category':c['category'],'correct_delta':b['metric']-a['metric'],
            'completeness_delta':completeness(b)-completeness(a),
            'concise_chars':len(a['result'].get('answer','')),'complete_chars':len(b['result'].get('answer',''))})
    result['paired']['completeness_gain']=statistics.mean(p['completeness_delta'] for p in pairs)
    result['paired']['coverage_wins']=sum(p['completeness_delta']>0 for p in pairs)
    result['paired']['coverage_losses']=sum(p['completeness_delta']<0 for p in pairs)
    result['paired']['coverage_ties']=sum(p['completeness_delta']==0 for p in pairs)
    rng=np.random.default_rng(20261013)
    indices=rng.integers(0,24,(20000,24))
    for metric,key in [('correct_delta','accuracy'),('completeness_delta','completeness')]:
        samples=np.array([p[metric] for p in pairs])[indices].mean(axis=1)
        result['paired'][key+'_paired_bootstrap95']=np.quantile(samples,[.025,.975]).tolist()
    for arm,group in result['groups'].items():
        rows=list(by_arm[arm].values())
        facts=[r.get('judgement',{}).get('facts_supported',[False]*fact_counts[r['id']]) for r in rows]
        group['completeness_micro']=sum(sum(f) for f in facts)/sum(map(len,facts)) if sum(map(len,facts)) else 1
        group['fully_supported_cases']=sum(completeness(r)==1 for r in rows)
        group['correct_times_completeness']=statistics.mean(r['metric']*completeness(r) for r in rows)
        group['result_status']=dict(Counter(r['result']['status'] for r in rows))
        judges=[run for r in rows for attempt in r['judge_attempts'] for run in attempt['runs']]
        group['judge_calls']=len(judges)
        group['judge_tokens']=token_usage(judges)
        group['search_calls']=sum(e.get('role')=='tool' and e.get('name')=='search' for r in rows for e in r['events'])
        group['read_calls']=sum(e.get('role')=='tool' and e.get('name')=='read' for r in rows for e in r['events'])
        group['answer_chars_min']=min(len(r['result'].get('answer','')) for r in rows)
        group['answer_chars_max']=max(len(r['result'].get('answer','')) for r in rows)
    result['token_change']={k:result['groups']['complete']['generation_tokens'][k]/result['groups']['concise']['generation_tokens'][k]-1 for k in ['input_tokens','output_tokens']}
    result['limits'].append('Paired bootstrap resamples questions only, not repeat-generation or judge variation; a zero-width interval for ties does not establish zero population effect.')
    manifest=OUTPUT/'executed-sources.json'
    if manifest.exists():
        frozen=json.loads(manifest.read_text())
        result['sources']=frozen['sources']
        result['execution_source_commit']=frozen['commit']
    result['analysis_source']={'path':str(Path(__file__).relative_to(ROOT)),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result['decision']='No promotion: one complete-coverage sentence did not improve accuracy or completeness; keep it in experiments.'
    result['recommendation']='Freeze identical observed evidence for24 paired answer-only comparisons before any more retrieval or verbosity changes.'
    result['closed_ledgers']=json.loads((OUTPUT/'closed-ledger-check.json').read_text())

    result['paired_details']=pairs
    (OUTPUT/'final-summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    # Pair IDs and per-question changes remain outside Git alongside the native traces.
    public={k:v for k,v in result.items() if k!='paired_details'}
    publish(ROOT/'evidence/complete-answer-20261013.json',public)
    print(json.dumps({k:v for k,v in public.items() if k not in ['cases','sources','protocol']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
