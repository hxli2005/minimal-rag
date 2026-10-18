"""One-off offline receipts for the actual runner and historical replay; no model calls."""
import copy
import json
import os
import tempfile
from datetime import datetime,timedelta
from night import *


def check_stopped(fn,expected):
    try:fn()
    except Stopped as error:assert str(error)==expected,(str(error),expected)
    else:raise AssertionError('expected stop: '+expected)


def main():
    plan=read(NIGHT/'night-plan.json')
    old=read(OLD/'workflow-state.json'); backfill=read(OLD/'backfill-state.json')
    assert old['status'] in ('completed','completed_with_budget_or_evaluation_limitations')
    assert {'b200','report'}<=set(old['completed_phases']) and backfill['status']=='completed'
    assert set(backfill['groups'])=={'A','A-lean','B'}
    for pid in [old['pid'],read(OLD/'active-phase-process.json')['pid']]:
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise AssertionError('previous process alive')
    assert subprocess.check_output(['git','status','--porcelain','--','src'],text=True)==''
    assert subprocess.check_output(['git','log','-1','--format=%H','--','experiments/reports/browsecomp-full-20261015.md'],text=True).strip()
    sizes={k:sum(map(len,samples(k).values())) for k in ('panel','dev','test')}
    assert sizes=={'panel':60,'dev':200,'test':400}
    for stage in sizes:
        for ids in samples(stage).values():assert len(ids)==len(set(ids))
    assert set(samples('dev')['enterprise']).isdisjoint(samples('test')['enterprise'])
    a=rows('v0','browsecomp','dev',samples('dev')['browsecomp'])
    lean=rows('v1','browsecomp','dev',samples('dev')['browsecomp'])
    known=compare({'enterprise':lean},{'enterprise':a},'test')
    assert known['delta_pp']==3 and np.allclose(known['bootstrap_95_pp'],[-1,8]),known
    assert known['per_benchmark']['enterprise']['wins']==4 and known['per_benchmark']['enterprise']['losses']==1
    historical=read(OLD/'A-lean/config-dev.json')
    expected={'model':'gpt-6.1-sol','effort':'high','request_timeout':180,'case_timeout':1800,'max_turns':30,'max_calls':30,'excerpt_chars':800,'retrieval':'SQLite FTS5 BM25','query_template_change':'get_document -> read'}
    assert all(historical[k]==v for k,v in expected.items())
    from minimal_rag_experiments import rag_optimization_eval as e
    base=read(ROOT/'data/benchmarks/sota-20261014/enterprise-dev/config.json')
    assert e.VARIANTS['v6-e800']==dict(e.VARIANTS['v6'],excerpt_chars=800)
    assert e.NATIVE_FORMAT==base['format_request'] and e.JUDGE_PROMPT==base['judge_prompt']
    assert e.VARIANTS['v6']['prompt']==base['settings']['prompt']
    save(HERE/'frozen.json',{'night_plan':plan,'v1':read(HERE/'v1.json'),'browsecomp_reuse_checked':expected,
        'browsecomp_non_generation_fields_preserved':{k:v for k,v in historical.items() if k not in expected},
        'enterprise':{'model':'gpt-6.1-sol','effort':'high','max_turns':12,'max_calls':12,'timeout':180,'case_timeout':900,'variant':e.VARIANTS['v6-e800'],
                      'format_request':e.NATIVE_FORMAT,'short_style':'concise','judge_model':'gpt-6.1-sol','judge_effort':'medium','judge_prompt':e.JUDGE_PROMPT,'corpus':str(ROOT/'data/benchmarks/enterprise/corpus.sqlite')},
        'allowed_differences':['historical -> current adapter including final_response_required','cascade+preview -> BM25','4000 -> 800 excerpt'],
        'baseline_parameters':base})
    with tempfile.TemporaryDirectory() as directory:
        p=Path(directory);save(p/'night-plan.json',plan);l=Ledger(p)
        check_stopped(lambda:l.reserve('v1','hotpot','hotpot','1','generation'),'stage_budget')
        stamp=datetime.fromisoformat(plan['start_at']).timestamp()
        for stage,cap in [('panel',600),('dev',1400),('test',2800)]:
            l.entries=[dict(provider='sol',candidate='v1',stage=stage,cleanup=False)]*cap
            check_stopped(lambda:l.reserve('v1',stage,'enterprise','x','generation',at=stamp),'stage_budget')
        l.entries=[dict(provider='sol',candidate='v0',stage='dev',cleanup=False)]*9000
        check_stopped(lambda:l.reserve('v1','dev','enterprise','x','generation',at=stamp),'exploration_budget')
        l.reserve('v1','dev','enterprise','x','judge',cleanup=True,at=stamp)
        assert l.counts()['total']==9001
        l.entries=[dict(provider='sol',candidate='v0',stage='dev',cleanup=True)]*10000
        check_stopped(lambda:l.reserve('v1','dev','enterprise','x','judge',cleanup=True,at=stamp),'night_budget')
        l.entries=[]
        check_stopped(lambda:l.reserve('v1','dev','enterprise','x','generation',at=l.deadline-1799),'generation_cutoff')
        check_stopped(lambda:l.reserve('v1','dev','enterprise','x','judge',cleanup=True,at=l.deadline),'deadline')
        l=Ledger(p); before=l.counts();l.reserve('v1','dev','enterprise','y','generation',at=stamp)
        resumed=Ledger(p);assert resumed.counts()['total']==before['total']+1 and resumed.deadline==l.deadline
        s=initial_state()
        begin_candidate(s,dict(version='v1',parent='v0',expected_requests=1,expected_seconds=1),l,at=stamp)
        finish_candidate(s,'v1',False,'offline');assert s['current_best']=='v0' and s['status']=='awaiting_candidate'
        begin_candidate(s,dict(version='v2',parent=s['current_best'],expected_requests=1,expected_seconds=1),l,at=stamp)
        finish_candidate(s,'v2',True,'offline')
        assert s['current_best']=='v2' and s['night_anchor']=='v0' and s['status']=='awaiting_candidate'
        assert can_start(s,l,at=l.deadline-5399)=='insufficient_time'
        begin_candidate(s,dict(version='v3',parent=s['current_best'],expected_requests=1,expected_seconds=1),l,at=stamp)
        assert s['candidates'][-1]['parent']=='v2' and can_start(s,l,at=stamp)=='candidate_limit'
        assert l.plan==plan
    synthetic={d:[{'id':str(i),'terminal':True,'correct':True,'uncached':100,'missing_usage':0} for i in range(100)] for d in PRIMARY}
    changed=copy.deepcopy(synthetic)
    changed['financebench']=[{'correct':False,'uncached':1e15,'missing_usage':1}]
    changed['hotpotqa']=None
    assert compare(changed,synthetic,'dev')==compare(synthetic,synthetic,'dev')
    best=copy.deepcopy(synthetic);next_=copy.deepcopy(synthetic)
    for d in PRIMARY:
        for r in best[d]:r['uncached']=108
        for r in next_[d]:r['uncached']=116.64
    assert compare(next_,best,'dev')['passed'] and not compare(next_,synthetic,'dev')['passed']
    for d in PRIMARY:
        for r in best[d][:2]:r['correct']=False
        for r in next_[d][:4]:r['correct']=False
    assert compare(next_,best,'panel')['passed'] and not compare(next_,synthetic,'panel')['passed']
    missing=copy.deepcopy(synthetic);missing['enterprise'][0]['missing_usage']=1
    assert compare(missing,synthetic,'dev')['status']=='undetermined'
    assert compare(synthetic,missing,'dev')['cost_ratio_kind']=='upper_bound'
    # The auxiliary reservations still use the same sol budget.
    with tempfile.TemporaryDirectory() as directory:
        save(Path(directory)/'night-plan.json',plan);l=Ledger(directory)
        l.reserve('final','finance','financebench','1','generation',at=stamp)
        assert l.counts()['total']==1 and l.counts()['exploration']==1
        l.entries=[dict(provider='sol',candidate='v0',stage='finance',cleanup=False)]*500
        check_stopped(lambda:l.reserve('final','finance','financebench','2','generation',at=stamp),'stage_budget')
    save(HERE/'offline-validation.json',{'passed':True,'created':now(),'model_requests':0,'sample_sizes':sizes,
        'known_Alean_minus_A':known,'checks':['previous_workflow_final_and_exited','reuse_configuration_and_prompt','v6-e800_only_excerpt_diff',
          'stage_caps_600_1400_2800','total_10000_exploration_9000_and_cleanup_reserve','deadline_and_generation_cutoff','resume_preserves_counts_and_deadline',
          'rejection_returns_old_base_adoption_returns_new_base','anchor_never_changes','cumulative_loss_and_cost_rejected','auxiliary_metrics_isolated_but_requests_counted','missing_usage_conservative'],
        'boundary':'Offline state/worker wiring checks; heartbeat wakeup and online model execution are separate evidence.'})
    save(NIGHT/'night-state.json',read(NIGHT/'night-state.json',initial_state()))
    print('offline validation passed')


def replay():
    from minimal_rag.indexed_rag import IndexedKnowledgeBase
    kb=IndexedKnowledgeBase(ROOT/'data/benchmarks/enterprise/corpus.sqlite',800)
    aliases=read(ROOT/'data/benchmarks/enterprise/duplicate-id-aliases.json',{})
    cs=cases('enterprise');records=[]
    for qid,c in cs.items():
        trace=read(ROOT/f'data/benchmarks/sota-20261014/enterprise-dev/traces/{qid}.json')
        for event in trace['events']:
            if event.get('role')!='tool' or event['name']!='search':continue
            args=json.loads(event['arguments_json']);args['limit']=20
            result=kb.search(**args)
            historical=event['content']
            oldids={aliases.get(h['document_id'],h['document_id']) for h in historical.get('hits',[])+historical.get('other_matches',[])}
            newids={aliases.get(h['document_id'],h['document_id']) for h in result['hits']}
            gold=set(c['gold_ids'])
            records.append({'id':qid,'gold_count':len(gold),'historical_gold_hits':len(gold&oldids),'bm25_gold_hits':len(gold&newids),
                'historical_chars':len(json.dumps(historical,ensure_ascii=False)),'bm25_chars':len(json.dumps(result,ensure_ascii=False)),
                'historical_doc_count':len(oldids),'bm25_doc_count':len(newids)})
    kb.close()
    total=sum(r['gold_count'] for r in records)
    summary={'search_calls':len(records),'cases':len(cs),'model_calls':0,
             'historical_gold_coverage_per_search':sum(r['historical_gold_hits'] for r in records)/total,
             'bm25_top20_gold_coverage_per_search':sum(r['bm25_gold_hits'] for r in records)/total,
             'mean_historical_return_chars':np.mean([r['historical_chars'] for r in records]),
             'mean_bm25_top20_return_chars':np.mean([r['bm25_chars'] for r in records]),
             'note':'Historical actual hits+previews versus BM25 top20 for the same queries; diagnostic only, not an adoption gate.'}
    save(HERE/'offline-replay.json',{'summary':summary,'per_search':records});print(json.dumps(summary))

if __name__=='__main__':
    main();replay()
