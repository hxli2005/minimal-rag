"""Post-hoc fusion grid over cached rankings; same prior dev halves, no models."""
import json
import hashlib
import statistics
from collections import defaultdict

from worker import BASE, OUTPUT, ROOT, read, save


def fuse(sparse, dense, k, weight):
    scores = defaultdict(float)
    for ranking, scale in ((sparse,1.),(dense,weight)):
        for rank, doc in enumerate(dict.fromkeys(ranking),1):
            scores[doc] += scale/(k+rank)
    return sorted(scores,key=lambda doc:(-scores[doc],doc))[:20]


def summarize(rows, partition, category):
    selected=[r for r in rows if r['gold_ids'] and (partition=='all' or r['partition']==partition)
              and (category=='all' or r['category']==category)]
    return {'n':len(selected), **{f'recall{k}':statistics.mean(len(set(r['gold_ids'])&set(r['ranked'][:k]))/len(set(r['gold_ids'])) for r in selected) for k in (5,20)}}


def main():
    # Grid and selection rule fixed here before calculating any candidate result.
    grid=[(k,w) for k in (10,30,60,100) for w in (.5,1.,2.)]
    source_path=ROOT/'runs/qwen3-chunk-scale-20261012/records.json'
    source_bytes=source_path.read_bytes()
    source=json.loads(source_bytes)
    partition={c['question_id']:('selection' if i%2==0 else 'validation') for i,c in enumerate(read(BASE/'enterprise/dev.json'))}
    dense={r['id']:r for r in source['qwen']}; original={r['id']:r for r in source['hybrid']}
    candidates={}
    for k,weight in grid:
        rows=[]
        for case in source['bm25']:
            ranked=fuse(case['candidate_document_ids'],dense[case['id']]['candidate_document_ids'],k,weight)
            if (k,weight)==(60,1.):
                assert ranked==original[case['id']]['ranked_document_ids'],case['id']
            rows.append({'id':case['id'],'category':case['category'],'gold_ids':case['gold_ids'],
                         'partition':partition[case['id']],'ranked':ranked})
        candidates[f'k{k}/dense{weight:g}']={'k':k,'dense_weight':weight,'rows':rows,
            'groups':{f'{p}/{c}':summarize(rows,p,c) for p in ('selection','validation','all') for c in ('semantic','all')}}
    baseline=candidates['k60/dense1']
    eligible=[(name,c) for name,c in candidates.items() if c['groups']['selection/all']['recall20']>=baseline['groups']['selection/all']['recall20']]
    # Semantic@20, then overall@20; prefer fewer deviations from frozen default on ties.
    chosen_name,chosen=max(eligible,key=lambda item:(item[1]['groups']['selection/semantic']['recall20'],
        item[1]['groups']['selection/all']['recall20'],-int(item[1]['k']!=60)-int(item[1]['dense_weight']!=1.)))
    result={'kind':'posthoc_cached_document_fusion_grid','date':'2026-10-13','model_calls':0,'gpu':False,
        'settings':'12 fixed grids: k10/30/60/100 × dense weight0.5/1/2; lexical weight1; same100+100 candidates and20 outputdocs.',
        'selection_rule':'Max selection semantic@20 subject to no selection overall@20 regression vs equalRRF60; then overall@20, fewer changed parameters. No validation-based reselection.',
        'partition':'Exact original dev-index parity halves used before tonight BM25 replay; both now known dev, not untouched.',
        'source':{'path':str(source_path.relative_to(ROOT)),'sha256':hashlib.sha256(source_bytes).hexdigest()},
        'baseline100_rankings_reproduced':True,'chosen':chosen_name,
        'validation_semantic_gain20':chosen['groups']['validation/semantic']['recall20']-baseline['groups']['validation/semantic']['recall20'],
        'decision':'No automatic promotion; original failed gate untouched.', 'default_changed':False,
        'chosen_groups':chosen['groups'],'baseline_groups':baseline['groups'],
        'candidates':{name:{k:v for k,v in c.items() if k!='rows'} for name,c in candidates.items()},
        'chosen_rows':chosen['rows'],
        'limits':['Original failed Qwen gate and full-build decision unchanged.','Post-hoc known sample with injected semantic references; no full-corpus or QA score.','Existing reference omissions limit overall guardrail. No new encodings or model loading.']}
    save(OUTPUT/'rrf-grid.json',result);save(ROOT/'evidence/rrf-grid-20261013.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('chosen_rows','candidates')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
