"""Local-only reference scoring. A partial candidate is never reported as complete."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/qwen-bm25-20261015'


def metrics(hits, gold):
    ranked = [h['document_id'] for h in hits]
    return {**{f'recall@{k}': len(set(ranked[:k]) & gold)/len(gold) for k in (5,20)},
            **{f'all_gold@{k}': float(gold <= set(ranked[:k])) for k in (5,20)}} if gold else None


def summarize():
    labels = json.loads((RUN/'labels.json').read_text())
    controls = {r['id']:r for r in json.loads((RUN/'input/bm25.json').read_text())}
    candidate_path = RUN/'output/qwen.json'
    candidate = {r['id']:r for r in json.loads(candidate_path.read_text())} if candidate_path.exists() else {}
    pairs = []
    for c in labels:
        gold = set(c['gold_ids'])
        a = metrics(controls[c['id']]['hits'], gold)
        b = metrics(candidate[c['id']]['hits'], gold) if c['id'] in candidate else None
        pairs.append({'id':c['id'],'category':c['category'],'A':a,'B':b})
    result = {'planned':len(labels),'candidate_completed':len(candidate),'complete':len(candidate)==len(labels),
              'limits':'Known development diagnostic pool contains reference documents; not full-corpus or official scores.',
              'groups':{},'pairs':pairs}
    for group in ['all', *sorted({c['category'] for c in labels})]:
        rows = [r for r in pairs if (group=='all' or r['category']==group) and r['A'] is not None]
        data = {'n':len(rows),'A':{k:statistics.mean(r['A'][k] for r in rows) for k in rows[0]['A']} if rows else {}}
        if rows and all(r['B'] is not None for r in rows):
            data['B'] = {k:statistics.mean(r['B'][k] for r in rows) for k in rows[0]['A']}
            data['paired'] = {k:{'wins':sum(r['B'][k]>r['A'][k] for r in rows),
                                 'losses':sum(r['B'][k]<r['A'][k] for r in rows),
                                 'ties':sum(r['B'][k]==r['A'][k] for r in rows)} for k in rows[0]['A']}
        result['groups'][group] = data
    target = RUN/'summary.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('planned','candidate_completed','complete','groups')},ensure_ascii=False))


if __name__=='__main__':
    summarize()
