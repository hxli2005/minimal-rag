"""Small shared helpers; all question/answer-bearing artifacts stay under data/."""
import ast
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'data/benchmarks/browsecomp-plus'
OUT = DATA / 'pilot-20261015'
OFFICIAL = DATA / 'official'
LABEL = 'sol 代理裁判，不能和榜单比较'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def now():
    return datetime.now(timezone.utc).isoformat()


def note(text):
    with (HERE / 'notes.md').open('a') as f:
        f.write(f'- {now()} {text}\n')


def source_string(path, name):
    # Inspect constants without importing or running the official repository.
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == 'strip':
                return ast.literal_eval(value.func.value).strip()
            return ast.literal_eval(value)
    raise ValueError(f'Missing constant: {name}')


def queries():
    return {str(row['query_id']): row for row in map(json.loads, (DATA / 'queries.jsonl').read_text().splitlines())}


def sample(limit=None):
    return read(HERE / 'sample.json')['query_ids'][:limit]


def request_count():
    path = OUT / 'generation-requests.jsonl'
    return len(path.read_text().splitlines()) if path.exists() else 0


def export(trace):
    counts, docids = Counter(), set()
    for e in trace['events']:
        if e['role'] != 'tool':
            continue
        counts[e['name']] += 1
        content = e['content']
        if not isinstance(content, dict) or 'error' in content:
            continue
        if e['name'] == 'search':
            docids.update(str(h['document_id']) for h in content['hits'])
        if e['name'] == 'read':
            docids.add(str(content['document_id']))
    result = trace.get('result', {})
    return {'query_id': trace['query_id'], 'tool_call_counts': dict(counts),
            'status': 'completed' if result.get('status') == 'finished' else result.get('status', 'interrupted'),
            'retrieved_docids': sorted(docids),
            'result': [{'type': 'output_text', 'output': result.get('answer', '')}]}


def parse_judge_response(text):
    # Same three accepted field spellings and yes/no semantics as evaluate_run.py.
    def field(name, value):
        patterns = [rf'\*\*{name}:\*\*\s*({value})', rf'\*\*{name}\*\*:\s*({value})', rf'{name}:\s*({value})']
        for pattern in patterns:
            match = re.search(pattern, text or '', re.I | re.S)
            if match:
                return match.group(1).strip()
    correct = field('correct', 'yes|no')
    confidence = field('confidence', r'\d+(?:\.\d+)?')
    return {'correct': None if correct is None else correct.lower() == 'yes',
            'parse_error': correct is None,
            'extracted_final_answer': field('extracted_final_answer', r'.*?(?=\n|$)'),
            'confidence': min(100., float(confidence)) if confidence else None}


def qrels():
    result = {}
    for line in (OFFICIAL / 'topics-qrels/qrel_evidence.txt').read_text().splitlines():
        qid, _, docid, _ = line.split()
        result.setdefault(qid, []).append(docid)
    return result
