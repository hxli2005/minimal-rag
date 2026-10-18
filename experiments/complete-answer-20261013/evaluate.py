"""Matched dev24 answer coverage experiment; only one sentence changes."""
import hashlib
import argparse
import asyncio
import json
import statistics
import sys
import tempfile
from pathlib import Path

from minimal_rag.codex_adapter import SYSTEM_PROMPT, rag_codex_config, token_usage
from minimal_rag.indexed_rag import IndexedKnowledgeBase, indexed_tools
from minimal_rag.loop import Tool, run_agent
from minimal_rag_experiments.rag_judge import PROMPT, assess
from minimal_rag_experiments.rag_optimization_eval import NATIVE_FORMAT
from minimal_rag_experiments.evidence_summary import publish

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/overnight-20261012'))
import budget
OUTPUT = ROOT/'runs/complete-answer-20261013'
COHORT = ROOT/'experiments/overnight-20261012/cohort.json'
CASES = json.loads(COHORT.read_text())['datasets']['enterprise']
COMPLETE_PROMPT = SYSTEM_PROMPT + " 请完整覆盖问题涉及的相关事实。"


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def summary(reuse):
    groups, rows_by_arm = {}, {}
    for arm in ['concise','complete']:
        folder = ROOT/'runs/tool-capabilities-20261013/baseline' if arm=='concise' and reuse else OUTPUT/arm
        rows = [json.loads(p.read_text()) for p in folder.glob('*.json')]
        rows = [r for r in rows if r.get('terminal')]
        rows_by_arm[arm] = rows
        facts = [r.get('judgement',{}).get('facts_supported',[False]*len(next(c for c in CASES if c['id']==r['id'])['facts'])) for r in rows]
        lengths = [len(r.get('result',{}).get('answer','')) for r in rows]
        groups[arm] = {'planned':24,'completed':len(rows),'correct':sum(r.get('metric',0) for r in rows),
            'accuracy':sum(r.get('metric',0) for r in rows)/24,
            'completeness_macro':sum(0 if r.get('result',{}).get('status')!='finished' else statistics.mean(f) if f else 1 for r,f in zip(rows,facts))/24,
            'answer_chars_median':statistics.median(lengths) if lengths else None,
            'answer_chars_mean':statistics.mean(lengths) if lengths else None,
            'generation_calls':sum(len(r['model_runs']) for r in rows),
            'generation_tokens':token_usage([run for r in rows for run in r['model_runs']]),
            'baseline_reused':arm=='concise' and reuse}
    ledger = [json.loads(l) for l in budget.LEDGER.read_text().splitlines()] if budget.LEDGER.exists() else []
    result = {'groups':groups,'new_calls':len(ledger),'new_limit':budget.LIMIT,'complete':all(g['completed']==24 for g in groups.values()),
        'metric_definitions':{'accuracy':'correct count / all24 planned outcomes','completeness_macro':'mean over24 of supported reference-fact fraction; failed outcomes zero','length':'Unicode characters in raw answer, including citations/whitespace; failed answer empty'},
        'limits':['Known dev24 with frozen references and local sol judge, not official score.','Default prompt has no explicit brevity instruction; concise is the CLI baseline label.','Length is diagnostic, not proof of fact support.']}
    if result['complete']:
        a,b=[{r['id']:r for r in rows_by_arm[arm]} for arm in ['concise','complete']]
        diffs=[b[c['id']]['metric']-a[c['id']]['metric'] for c in CASES]
        result['paired']={'wins':sum(d>0 for d in diffs),'losses':sum(d<0 for d in diffs),'ties':sum(d==0 for d in diffs),'accuracy_gain':statistics.mean(diffs)}
        requests=[run for arm,rows in rows_by_arm.items() if not (arm=='concise' and reuse) for r in rows for run in [*r['model_runs'],*[run for attempt in r['judge_attempts'] for run in attempt['runs']]]]
        if sorted(r['overnight_call'] for r in requests)!=list(range(1,len(ledger)+1)) or len(ledger)>budget.LIMIT or any(r['model']!='gpt-6.1-sol' for r in ledger):
            raise RuntimeError('New-round request reconciliation failed')
        result['new_tokens']=token_usage(requests)
        result['request_status']=dict(__import__('collections').Counter(r['status'] for r in requests))
        result['protocol']=json.loads((OUTPUT/'protocol.json').read_text())
        result['sources']=[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [COHORT,OUTPUT/'protocol.json',OUTPUT/'calls.jsonl',Path(__file__),ROOT/'src/minimal_rag/applications/qa.py',ROOT/'src/minimal_rag/indexed_rag.py']]

        manifest=OUTPUT/'executed-sources.json'
        if manifest.exists():
            result['sources']=json.loads(manifest.read_text())['sources']
        result['cases']=rows_by_arm
    save(OUTPUT/'summary.json',result)
    return result


async def one(client, case, arm):
    path=OUTPUT/arm/(case['id']+'.json')
    report=json.loads(path.read_text()) if path.exists() else {'id':case['id'],'events':[],'model_runs':[],'judge_attempts':[]}
    if report.get('terminal'):return
    kb=IndexedKnowledgeBase(ROOT/'data/benchmarks/enterprise/corpus.sqlite',check_same_thread=False)
    with tempfile.TemporaryDirectory(prefix='rag-complete-answer-') as cwd:
        try:
            if 'result' not in report:
                if report['model_runs']:
                    raise RuntimeError('Partial generation retained: inspect failure before resuming; never silently regenerate')
                model=budget.BudgetModel(client,cwd,label=f'{arm}/{case["id"]}/generate',effort='high',timeout=180,max_calls=6,
                    system_prompt=COMPLETE_PROMPT if arm=='complete' else SYSTEM_PROMPT)
                report['model_runs']=model.runs
                tools={}
                for name,tool in indexed_tools(kb).items():
                    async def execute(_fn=tool.execute,**kwargs):return await asyncio.to_thread(_fn,**kwargs)
                    tools[name]=Tool(tool.description,tool.arguments,execute)
                report['result']=await asyncio.wait_for(run_agent(model,tools,NATIVE_FORMAT+case['question'],6,
                    lambda e:(report['events'].append(e),save(path,report))),900)
                save(path,report)
            if report['result']['status']!='finished':report.update(metric=0,terminal=True)
            else:
                grade,_=await assess(lambda:budget.BudgetModel(client,cwd,label=f'{arm}/{case["id"]}/judge',effort='medium',timeout=150,max_calls=1,system_prompt=PROMPT),
                    {'question':case['question'],'reference_answer':case['gold'],'reference_facts':case['facts'],'candidate_answer':report['result']['answer']},
                    len(case['facts']),report['judge_attempts'],lambda:save(path,report))
                report.update(judgement=grade,metric=float(grade['correct']),terminal=True)
            save(path,report)
        finally:
            kb.close();save(path,report)


async def main(args):
    budget.LEDGER,budget.LIMIT=OUTPUT/'calls.jsonl',args.call_limit
    from openai_codex import AsyncCodex
    protocol={'model':'gpt-6.1-sol','call_limit':args.call_limit,'cases':[c['id'] for c in CASES],
        'arms':['concise','complete'],'baseline_reused':args.reuse_baseline,'system_prompts':{'concise':SYSTEM_PROMPT,'complete':COMPLETE_PROMPT},
        'single_change':'Append one complete-coverage sentence only','retrieval':'unchanged full SQLite BM25','context_chars':4000,'max_generation_calls':6,'generation_effort':'high','judge':'unchanged sol medium and frozen facts','ordering':'interleaved per case when both arms are fresh'}
    if not args.live:
        print(json.dumps({'planned_protocol':protocol,'offline_snapshot':summary(args.reuse_baseline)},ensure_ascii=False));return
    existing=OUTPUT/'protocol.json'
    if existing.exists() and json.loads(existing.read_text())!=protocol:raise RuntimeError('Do not mix protocols in an existing run')
    save(existing,protocol)
    async with AsyncCodex(rag_codex_config('/private/tmp')) as client:
        account=await client.account(refresh_token=False)
        if getattr(getattr(account.account,'root',account.account),'type',None)!='chatgpt':raise RuntimeError('Existing subscription required, no API-key fallback')
        for case in CASES:
            for arm in (['complete'] if args.reuse_baseline else ['concise','complete']):
                await one(client,case,arm)
                s=summary(args.reuse_baseline);print(json.dumps({'case':case['id'],'arm':arm,'calls':s['new_calls'],'completed':{a:g['completed'] for a,g in s['groups'].items()}},ensure_ascii=False),flush=True)
    result=summary(args.reuse_baseline)
    if result['complete']:publish(ROOT/'evidence/complete-answer-20261013.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--live',action='store_true');p.add_argument('--reuse-baseline',action='store_true');p.add_argument('--call-limit',required=True,type=int);args=p.parse_args()
    if not 1<=args.call_limit<=336:p.error('call-limit must be within the explicitly approved new allowance, at most336')
    asyncio.run(main(args))
