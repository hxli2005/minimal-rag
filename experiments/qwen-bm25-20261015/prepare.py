"""Prepare the existing diagnostic pool and a pure BM25 control. No model calls."""
import json
import re
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'runs/sentence-offline-dev100-20261012'
OUT = ROOT / 'runs/qwen-bm25-20261015'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def prepare():
    out = OUT / 'input'
    out.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(f'file:{SOURCE}/units.sqlite?mode=ro', uri=True)
    src.row_factory = sqlite3.Row
    aliases = json.loads((SOURCE / 'aliases.json').read_text())
    chunks = [dict(r) for r in src.execute('SELECT id,document_id,offset,end_offset,text FROM chunk_units ORDER BY id')]
    titles = dict(src.execute('SELECT id,title FROM documents'))
    for c in chunks:
        c['canonical_id'] = aliases.get(c['document_id'], c['document_id'])
        c['title'] = titles[c['document_id']]
    with (out / 'chunks.jsonl').open('w') as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    cases = json.loads((SOURCE / 'cases.json').read_text())
    save(out / 'queries.json', [{'id': c['id'], 'question': c['question']} for c in cases])
    # Labels remain local. Neither retrieval model receives references or categories.
    save(OUT / 'labels.json', cases)
    main_ids = json.loads((ROOT / 'experiments/fixed-evidence-v3-20261015/frozen.json').read_text())['main_ids']
    assert set(main_ids) <= {c['id'] for c in cases}
    save(OUT / 'answer-dev24.json', main_ids)
    db = sqlite3.connect(':memory:')
    db.execute("CREATE VIRTUAL TABLE chunks USING fts5(title,text,tokenize='porter unicode61')")
    db.executemany('INSERT INTO chunks(rowid,title,text) VALUES(?,?,?)', [(c['id'], c['title'], c['text']) for c in chunks])
    by_id = {c['id']: c for c in chunks}
    results = []
    for case in cases:
        terms = list(dict.fromkeys(a or b for a, b in re.findall(r'"([^"\n]+)"|([\w]+)', case['question'])))
        query = ' OR '.join('"' + t.replace('"', '""') + '"' for t in terms)
        started = time.monotonic()
        found, seen = [], set()
        if query:
            rows = db.execute("SELECT rowid,bm25(chunks,2,1) AS score FROM chunks WHERE chunks MATCH ? ORDER BY score,rowid", (query,))
            for rowid, score in rows:
                c = by_id[rowid]
                if c['canonical_id'] in seen:
                    continue
                seen.add(c['canonical_id'])
                found.append({'chunk_id': rowid, 'document_id': c['canonical_id'], 'score': -score})
                if len(found) == 100:
                    break
            rows.close()
        results.append({'id': case['id'], 'hits': found, 'seconds': time.monotonic() - started})
    save(out / 'bm25.json', results)
    save(OUT / 'preparation.json', {'documents': len(titles), 'chunks': len(chunks), 'queries': len(cases),
         'answer_dev24': main_ids, 'bm25': 'FTS5 porter unicode61, title/body 2/1, OR literals, no embedding/reranking',
         'candidate_documents_per_path': 100, 'rrf_k': 60, 'output_documents': 20,
         'pool': 'Existing diagnostic pool includes gold documents; not full-corpus retrieval or unseen evaluation.',
         'new_generation_calls': 0, 'new_judge_calls': 0})
    db.close()
    src.close()
    print(json.dumps({'documents': len(titles), 'chunks': len(chunks), 'questions': len(cases),
                      'query_seconds': round(sum(r['seconds'] for r in results), 3)}))


if __name__ == '__main__':
    prepare()
