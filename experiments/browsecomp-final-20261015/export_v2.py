"""Search-only export and a private, unchanged official scoring bundle."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
DATA=ROOT/'data/benchmarks/browsecomp-plus'
OUT=DATA/'final-20261015'
REVISION='046949032b0328319cc9a02663a759ec601d9402'
sys.path.insert(0,str(ROOT/'experiments/system-v1-20261015'))
from night import NIGHT,read,save,result_folder


def export(trace):
    ids=set();calls=Counter()
    for event in trace.get('events',[]):
        if event.get('role')!='tool':continue
        calls[event['name']]+=1
        if event['name']!='search':continue
        content=event['content']
        if isinstance(content,str):
            for hit in json.loads(content):
                if 'docid' in hit:ids.add(str(hit['docid']))
        elif isinstance(content,dict):
            ids.update(str(h['document_id']) for h in content.get('hits',[]))
    result=trace.get('result',{})
    return {'query_id':str(trace['query_id']),'tool_call_counts':dict(calls),
        'status':'completed' if result.get('status')=='finished' else result.get('status','interrupted'),
        'retrieved_docids':sorted(ids),'result':[{'type':'output_text','output':result.get('answer','')}]}


def group(name,folder,ids):
    qrels={}
    for line in (DATA/'official/topics-qrels/qrel_evidence.txt').read_text().splitlines():
        qid,_,docid,_=line.split();qrels.setdefault(qid,set()).add(docid)
    changes=[]
    for qid in ids:
        t=read(folder/'traces'/f'{qid}.json')
        if not t or not t.get('result') or t.get('recoverable'):continue
        new=export(t)
        old=read(folder/'official'/f'{qid}.json')
        if old is None:
            old=dict(new,retrieved_docids=sorted(set(new['retrieved_docids'])|{str(e['content']['document_id']) for e in t['events'] if e.get('role')=='tool' and e['name']=='read' and isinstance(e['content'],dict) and e['content'].get('document_id')}))
        assert {k:v for k,v in new.items() if k!='retrieved_docids'}=={k:v for k,v in old.items() if k!='retrieved_docids'}
        save(OUT/'runs'/name/f'{qid}.json',new)
        gold=qrels[str(qid)]
        changes.append({'id':str(qid),'old_recall':len(set(old['retrieved_docids'])&gold)/len(gold),
                        'search_only_recall':len(set(new['retrieved_docids'])&gold)/len(gold),
                        'removed_read_only_docids':len(set(old['retrieved_docids'])-set(new['retrieved_docids']))})
    n=len(changes)
    return {'requested':len(ids),'terminal_exported':n,'old_recall_pct':sum(r['old_recall'] for r in changes)/n*100,
        'search_only_recall_pct':sum(r['search_only_recall'] for r in changes)/n*100,'per_query':changes}


def prepare():
    state=read(NIGHT/'night-state.json')
    assert state['status']=='frozen','Nightly selection must be frozen first'
    version=state['current_best']
    prior=ROOT/'experiments/browsecomp-full-20261015'
    ids=lambda name:read(prior/f'sample-{name}.json')['query_ids']
    selected=result_folder(version,'browsecomp','dev')
    a=DATA/'full-20261015/A'
    mapping={'version':version,'night_state':str(NIGHT/'night-state.json'),'selected_source':str(selected),
             'selected_group':'A' if selected==a else 'selected','official_revision':REVISION}
    summaries={'A':group('A',a,ids('all830')),'B':group('B',DATA/'full-20261015/B',ids('pilot30')+ids('b200'))}
    if selected!=a:summaries['selected']=group('selected',selected,ids('all830'))
    save(OUT/'selected.json',mapping)
    save(HERE/'export-summary.json',{'selected':mapping,'groups':summaries})
    bundle=OUT/'gpu-bundle'
    # Verify the local official source against its original pinned archive, not the enclosing project HEAD.
    with tarfile.open(DATA/'official.tar.gz') as tar:
        files=0
        for member in tar.getmembers():
            if not member.isfile():continue
            rel=Path(member.name).relative_to(member.name.split('/')[0])
            assert (DATA/'official'/rel).read_bytes()==tar.extractfile(member).read(),str(rel)
            files+=1
    shutil.copytree(DATA/'official',bundle,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','.git'))
    shutil.copytree(OUT/'runs',bundle/'runs',dirs_exist_ok=True)
    (bundle/'data').mkdir(exist_ok=True)
    shutil.copy2(DATA/'queries.jsonl',bundle/'data/browsecomp_plus_decrypted.jsonl')
    (bundle/'README-gpu.md').write_text('同机双H800，CUDA_VISIBLE_DEVICES=0,1，TP=2；只运行一个官方BF16裁判。\n锁定环境放数据盘；模型需足够空间，本次使用120GB内存盘/dev/shm。\nuv sync --frozen --python 3.10；后续始终用uv项目环境，不以基础镜像版本替代。\n先两题验证并保留首次有效eval，再A830/B230；官方batch64等默认参数不变，记录首批耗时。\n取回evals和环境日志，删除远端私有输入/判分副本，关机并清理付费存储；可保留免费公共依赖。\n')
    save(OUT/'bundle-manifest.json',{'official_revision':REVISION,'official_files_verified':files,
        'archive_sha256':hashlib.sha256((DATA/'official.tar.gz').read_bytes()).hexdigest(),
        'groups':{k:v['terminal_exported'] for k,v in summaries.items()},'selected':mapping})
    print(json.dumps({k:{key:val for key,val in v.items() if key!='per_query'} for k,v in summaries.items()}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.parse_args();prepare()
