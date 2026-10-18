"""CPU replay of actual queries for the14 full-corpus semantic natural misses."""
import json
import sqlite3
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/overnight-20261012'))
from offline import ranking, expression, STOP
from worker import BASE, HISTORY, read, save

OUTPUT=ROOT/'runs/tool-capabilities-20261013/semantic'


def natural200(db,query,aliases):
    cursor=db.execute("SELECT c.document_id,c.start,c.text FROM chunks c JOIN documents d ON d.id=c.document_id WHERE chunks MATCH ? AND rank MATCH 'bm25(0,0,2,1)' ORDER BY rank",(expression(query),))
    ids=[]; seen=set()
    for doc,_,_ in cursor:
        canonical=aliases.get(doc,doc)
        if canonical not in seen:
            seen.add(canonical);ids.append(canonical)
        if len(ids)==200:break
    cursor.close();return ids


def main():
    aliases=read(BASE/'enterprise/duplicate-id-aliases.json')
    cases={c['question_id']:c for c in read(BASE/'enterprise/dev.json')}
    natural=[read(p) for p in (ROOT/'runs/overnight-20261012/full-bm25-calibration').glob('qst_*.json')]
    misses=[r for r in natural if r['recall20']<1]
    with sqlite3.connect((BASE/'enterprise/corpus.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.execute('PRAGMA cache_size=-16000')
        for row in misses:
            target=OUTPUT/(row['id']+'.json')
            if target.exists():continue
            case=cases[row['id']];gold={aliases.get(d,d) for d in case['expected_doc_ids']}
            trace=read(HISTORY/'runs/enterprise-dev100-hybrid-minilm-wide/traces'/target.name)
            queries=[];seen_gold=False
            for event in trace['events']:
                if event.get('role')!='tool':continue
                if event.get('name')=='search':
                    q=json.loads(event['arguments_json'])
                    hits=ranking(db,q['query'],2,False,aliases,q.get('scope',''),q.get('all_terms',False),q.get('offset',0))
                    ids=[h['document_id'] for h in hits]
                    queries.append({'arguments':q,'gold_seen_before':seen_gold,'ranked':ids,'gold_recovered':sorted(gold&set(ids))})
                original_ids={aliases.get(h.get('document_id'),h.get('document_id')) for h in event['content'].get('hits',[event['content']])}
                seen_gold=seen_gold or bool(gold&original_ids)
            ranked200=natural200(db,row['question'],aliases)
            texts=[]
            for doc in gold:
                title,text=db.execute('SELECT title,text FROM documents WHERE id=?',(doc,)).fetchone()
                texts.append({'id':doc,'title':title,'text_chars':len(text),'preview':text[:2200]})
            recovered=set().union(*(set(q['gold_recovered']) for q in queries if not q['gold_seen_before']))
            record={'id':row['id'],'question':row['question'],'gold_ids':sorted(gold),'natural_ranked200':ranked200,
                'cached_natural_top20_matches':ranked200[:20]==row['ranked'],
                'gold_natural_ranks':{d:(ranked200.index(d)+1 if d in ranked200 else None) for d in gold},
                'historical_queries':queries,'pre_exposure_recovered':sorted(recovered),'original_gold_seen':seen_gold,
                'gold_sources':texts,'natural_top5_titles':[{'id':d,'title':db.execute('SELECT title FROM documents WHERE id=?',(d,)).fetchone()[0]} for d in ranked200[:5]],
                'bucket':('query_rewrite_recovers_before_gold_exposure' if recovered else 'ranking_21_to_200' if gold&set(ranked200) else 'not_in_natural200_or_pre_exposure_queries')}
            save(target,record);print(json.dumps({'completed':len(list(OUTPUT.glob('qst_*.json'))),'id':row['id'],'bucket':record['bucket']}),flush=True)
    records=[read(p) for p in OUTPUT.glob('qst_*.json')]
    from collections import Counter
    save(OUTPUT/'summary.json',{'semantic_questions':25,'natural_recall20':.44,'missed':14,'completed':len(records),
        'queries_replayed':sum(len(r['historical_queries']) for r in records),'buckets':dict(Counter(r['bucket'] for r in records)),
        'model_calls':0,'gpu':False,'limits':['Queries after original gold exposure are labelled and excluded from bridge counts.','Old queries were adaptive to a hybrid system, not a fresh BM25-only run.','Known dev analysis, no answer-quality claim or parameter promotion.'],'cases':records})


if __name__=='__main__':main()
