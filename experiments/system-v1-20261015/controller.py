"""Execute a frozen candidate; the chat/heartbeat owns evidence-based next proposals."""
import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from night import *

async def phase(version,stage,dataset,limit=None,judge_only=False):
    retries_path = NIGHT/'routing-recoveries.json'
    retries = read(retries_path,{})
    key = f'{version}-{stage}-{dataset}'
    while True:
        command=[sys.executable,'-u',str(HERE/'worker.py'),version,stage,dataset]
        if limit: command += ['--limit',str(limit)]
        if judge_only: command += ['--judge-only']
        process=await asyncio.create_subprocess_exec(*command,pass_fds=(int(os.environ['NIGHT_LOCK_FD']),))
        save(NIGHT/'active-process.json',{'pid':process.pid,'stage':stage,'dataset':dataset,'version':version,'started':now()})
        try:
            await asyncio.wait_for(process.wait(),max(1,Ledger().deadline-time.time()))
        except (TimeoutError,asyncio.CancelledError):
            process.terminate()
            try: await asyncio.wait_for(process.wait(),15)
            except TimeoutError: process.kill();await process.wait()
            raise Stopped('deadline' if time.time()>=Ledger().deadline else 'interrupted')
        state=read(NIGHT/f'{key}-state.json')
        reason=state['status']
        if reason=='routing_timeout' and retries.get(version,0)<2:
            retries[version]=retries.get(version,0)+1;save(retries_path,retries)
            note(f'路由启动超时：{key}用新进程恢复{retries[version]}/2（本候选累计），保留账本和尝试。')
            await asyncio.sleep(60);continue
        if reason=='rate_limited':
            await asyncio.sleep(60);continue
        if reason!='completed': raise Stopped(reason)
        return

def record_gate(version,best,stage,value):
    record={'version':version,'current_best':best,'night_anchor':'v0','stage':stage,'comparisons':value,
            'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'updated':now(),
            'counts':Ledger().counts()}
    save(HERE/f'{version}-{stage}-gate.json',record)
    with (HERE/'versions.md').open('a') as f:
        f.write(f'\n- {now()} 夜间20261015T102540，{version} {stage}：'+
            '；'.join(f'{name}={v["status"]}, Δ={v.get("delta_pp","见分项")}pp, CI={v.get("bootstrap_95_pp")}, cost={v.get("cost_ratio")} ({v.get("cost_ratio_kind","panel不判成本")})' for name,v in value.items())+
            f'；完整结果见 experiments/system-v1-20261015/{version}-{stage}-gate.json。\n')
    update_version_row(version,best,stage)
    commit(f'Record {version} {stage} dual-baseline gate')


def update_version_row(version,best,stage):
    cfg=read(HERE/f'{version}.json')
    state=read(NIGHT/'night-state.json')
    status=next(c['status'] for c in state['candidates'] if c['version']==version)
    def metric(dataset,split):
        ids=samples('test' if split=='test' else 'dev').get(dataset,[])
        rs=rows(version,dataset,split,ids)
        return f'{sum(bool(r["correct"]) for r in rs)/len(rs)*100:.1f}' if all(r['correct'] is not None and r['terminal'] for r in rs) else '—'
    g=read(HERE/f'{version}-{stage}-gate.json')['comparisons']
    outcomes='；'.join(f'{k}: {v["status"]} Δ{v.get("delta_pp","分项")} CI{v.get("bootstrap_95_pp","—")}' for k,v in g.items())
    costs=' / '.join(str(v.get('cost_ratio','—')) for v in g.values())
    line=f'| {version} | {best} / v0 | {cfg.get("change",cfg.get("benefit",version))} | {metric("enterprise","dev")} / {metric("enterprise","test")} | {metric("browsecomp","dev")} | {stage}: {outcomes} | {costs} | {status} |'
    path=HERE/'versions.md';lines=path.read_text().splitlines()
    for i,old in enumerate(lines):
        if old.startswith(f'| {version} |'):
            lines[i]=line;break
    else:
        i=next(i for i,l in enumerate(lines) if l.startswith('v0主评分来源'))
        lines.insert(i,line+'\n')
    path.write_text('\n'.join(lines)+'\n')


def demonstrated_benefit(cfg,dev,test):
    if cfg.get('simplifies_implementation'):return 'simpler implementation'
    if dev['bootstrap_95_pp'][0]>0:return 'positive paired main-dev confidence interval'
    if dev.get('cost_ratio') is not None and test.get('cost_ratio') is not None and max(dev['cost_ratio'],test['cost_ratio'])<=0.95:
        return 'at least 5% lower primary uncached input in both dev and test'
    return None


def projection(version,stage):
    ledger=Ledger(); cfg=read(HERE/f'{version}.json')
    expected_requests=expected_seconds=0
    split='test' if stage=='test' else 'dev'
    for dataset,ids in samples(stage).items():
        if f'{dataset}_{split}' in cfg.get('reuse',{}):continue
        actual=rows(version,dataset,split,ids)
        missing=sum(not r['terminal'] for r in actual)
        observed=[r for r in rows(version,dataset,'dev',samples('dev')[dataset]) if r['terminal']]
        if observed:
            per_calls=sum(r['calls'] for r in observed)/len(observed)
            per_seconds=sum(r['seconds'] for r in observed)/len(observed)
        else:
            per_calls,per_seconds=(8.53,110) if dataset=='browsecomp' else (4.13,55)
        expected_requests+=missing*(per_calls+(0 if dataset=='browsecomp' else 1))
        expected_seconds+=missing*max(per_seconds/5,12 if dataset=='enterprise' else 5)
    used=sum(e['provider']=='sol' and e['candidate']==version and e['stage']==stage for e in ledger.entries)
    receipt={'remaining_requests_plus_20pct':expected_requests*1.2,'remaining_seconds_plus_20pct':expected_seconds*1.2,
             'stage_already_used':used,'stage_limit':ledger.plan['stage_limits'][stage],'updated':now()}
    save(HERE/f'{version}-{stage}-projection.json',receipt)
    if used+expected_requests*1.2>ledger.plan['stage_limits'][stage]:raise Stopped('stage_projection')
    if ledger.counts()['exploration']+expected_requests*1.2>ledger.plan['exploration_limit']:raise Stopped('exploration_projection')
    if ledger.counts()['total']+expected_requests*1.2>ledger.plan['sol_limit']:raise Stopped('exploration_projection')
    if time.time()+expected_seconds*1.2>ledger.deadline-1800:raise Stopped('time_projection')


def smoke(version):
    lengths=[];tools=0
    for qid in samples('panel')['enterprise'][:3]:
        t=read(result_folder(version,'enterprise','dev')/'traces'/f'{qid}.json')
        assert t['result']['status']=='finished'
        assert isinstance(t.get('judgement',{}).get('correct'),bool)
        seen=set()
        for e in t['events']:
            if e['role']!='tool':continue
            tools+=1;c=e['content'];assert not c.get('error')
            for hit in c.get('hits',[c]):
                if hit.get('document_id'):seen.add(hit['document_id'])
                if e['name']=='search': lengths.append(len(hit['text']))
        assert set(t['result']['sources'])<=seen
    expected=read(HERE/f'{version}.json')['enterprise']['excerpt_chars']
    assert tools and lengths and max(lengths)<=expected
    save(HERE/f'{version}-smoke.json',{'passed':True,'n':3,'tool_calls':tools,'max_search_excerpt_chars':max(lengths),
         'judge_valid':True,'citations_seen':True,'score_not_used_for_assembly':True})

async def main(version):
    path=NIGHT/'night-state.json'
    state=read(path,initial_state())
    cfg=read(HERE/f'{version}.json')
    existing=next((c for c in state['candidates'] if c['version']==version),None)
    if existing and existing['status'] in ('adopted','rejected'):return
    if not existing:
        begin_candidate(state,cfg,Ledger())
    else:
        existing['status']='running'
        existing.pop('reason',None)
        state.pop('stop_reason',None)
    best=state['candidates'][-1]['parent']
    state.update(status='running',active_candidate=version,pid=os.getpid(),updated=now());save(path,state)
    root_task=asyncio.current_task()
    for sig in (signal.SIGINT,signal.SIGTERM):asyncio.get_running_loop().add_signal_handler(sig,root_task.cancel)
    try:
        for stage in ('panel','dev','test'):
            prior=read(HERE/f'{version}-{stage}-gate.json')
            if prior and all(v['passed'] for v in prior['comparisons'].values()):continue
            state.update(stage=stage,updated=now());save(path,state)
            projection(version,stage)
            split='test' if stage=='test' else 'dev'
            for dataset,ids in samples(stage).items():
                if f'{dataset}_{split}' in cfg.get('reuse',{}):continue
                if stage=='panel' and dataset=='enterprise' and not (HERE/f'{version}-smoke.json').exists():
                    await phase(version,stage,dataset,limit=3)
                    smoke(version)
                await phase(version,stage,dataset)
                # Retry all missing grades uniformly, preserving the first valid response.
                while any(r['terminal'] and r['correct'] is None for r in rows(version,dataset,split,ids)):
                    reports=[read(result_folder(version,dataset,split)/('judgements' if dataset=='browsecomp' else 'traces')/f'{qid}.json',{}) for qid in ids]
                    if any(sum(bool(a.get('received_response') or 'response' in a) for a in r.get('attempts' if dataset=='browsecomp' else 'judge_attempts',[]))>=3 and r.get('judgement',{}).get('correct') is None for r in reports):break
                    await phase(version,stage,dataset,judge_only=True)
                    if any(r['correct'] is None for r in rows(version,dataset,split,ids)):await asyncio.sleep(60)
            value=gates(version,best,stage);record_gate(version,best,stage,value)
            if not all(v['passed'] for v in value.values()):
                state=finish_candidate(state,version,False,f'{stage}_gate')
                save(path,state);note(f'决定：{version}未通过{stage}，保留{best}，进入下一候选分析；理由见双基线门槛。')
                update_version_row(version,best,stage)
                commit(f'Reject {version} and return to outer optimization loop');return
            if stage=='dev':
                note(f'{version} dev通过，冻结后进入Enterprise已知test400；BrowseComp dev不冒充test。')
                commit(f'Freeze {version} before Enterprise regression test')
                # Attribution is optional; preserve the necessary test chain first.
        value=read(HERE/f'{version}-test-gate.json')['comparisons']['current_best']
        dev=read(HERE/f'{version}-dev-gate.json')['comparisons']['current_best']
        benefit=demonstrated_benefit(cfg,dev,value)
        adopted=bool(benefit)
        state=finish_candidate(state,version,adopted,'all_gates_passed: '+str(benefit))
        save(path,state);note(f'决定：{version}采用={adopted}，current_best={state["current_best"]}，anchor仍为v0；理由：{benefit}。')
        update_version_row(version,best,'test')
        commit(f'Complete {version} candidate and return to outer optimization loop')
    except BaseException as error:
        reason=str(error) or type(error).__name__
        if reason in ('exploration_budget','generation_cutoff'):
            for dataset in samples(state['stage']):
                if f'{dataset}_{"test" if state["stage"]=="test" else "dev"}' in cfg.get('reuse',{}):continue
                try:await phase(version,state['stage'],dataset,judge_only=True)
                except Stopped as cleanup_error:
                    if str(cleanup_error) in ('usage_limit','permission_blocked','deadline'):
                        reason=str(cleanup_error);break
        state['candidates'][-1].update(status='incomplete',reason=reason)
        state.update(status='awaiting_candidate' if reason in ('stage_budget','stage_projection') else 'stopped',stop_reason=reason,updated=now());save(path,state)
        note(f'本轮保存并停止：{reason}；候选保留未完成，不替换current_best。')
        record_gate(version,best,state['stage'],gates(version,best,state['stage']))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('version');a=p.parse_args()
    asyncio.run(main(a.version))
