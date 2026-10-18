"""Experiment-only wiring; question/answer-bearing files stay under data/."""
from collections import Counter
from pathlib import Path
import json
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PILOT = ROOT / 'experiments/browsecomp-pilot-20261015'
sys.path.append(str(PILOT))
import common as pilot
import judge as pilot_judge
from common import DATA, OFFICIAL, now, queries, read, save, source_string

OUT = DATA / 'full-20261015'
LIMITS = {'b30': 900, 'dev': 2200, 'full': 7500, 'b200': 5000}


def note(text):
    with (HERE / 'notes.md').open('a') as f:
        f.write(f'- {now()} {text}\n')


def ids(name):
    return read(HERE / f'sample-{name}.json')['query_ids']


def directory(group):
    return OUT / group


def count(stage):
    path = OUT / f'{stage}-requests.jsonl'
    return len(path.read_text().splitlines()) if path.exists() else 0


def config(group, stage):
    base = read(pilot.OUT / 'config.json')
    expected = dict(base, generation_application_request_limit=LIMITS[stage])
    allowed = {'generation_application_request_limit'}
    if group == 'A-lean':
        expected['excerpt_chars'] = 800
        allowed.add('excerpt_chars')
    elif group == 'B':
        expected.update(excerpt_chars=None, query_template_change='QUERY_TEMPLATE_NO_GET_DOCUMENT unchanged',
                        tools=['search'], search_k=5, snippet_max_tokens=512,
                        snippet_source='document prefix', tokenizer='Qwen/Qwen3-0.6B')
        allowed |= {'excerpt_chars', 'query_template_change', 'tools', 'search_k',
                    'snippet_max_tokens', 'snippet_source', 'tokenizer'}
    differences = {k: {'pilot': base.get(k), 'current': expected.get(k)}
                   for k in base.keys() | expected.keys() if base.get(k) != expected.get(k)}
    assert set(differences) <= allowed
    path = directory(group) / f'config-{stage}.json'
    if read(path, expected) != expected:
        raise ValueError('Frozen group configuration changed')
    save(path, expected)
    save(directory(group) / f'config-check-{stage}.json', {'allowed_differences': differences, 'passed': True})
    judge_config = read(pilot.OUT / 'judge-config.json')
    assert judge_config['parameters'] == pilot_judge.PARAMS and judge_config['mode'] == 'openrouter'
    assert judge_config['assistant_prefill'] == pilot_judge.NON_THINKING_PREFIX
    save(directory(group) / 'judge-config.json', judge_config)
    return expected


def export(trace):
    if trace.get('group') != 'B':
        return pilot.export(trace)
    result = trace.get('result', {})
    docids, calls = set(), Counter()
    for event in trace['events']:
        if event['role'] != 'tool':
            continue
        calls[event['name']] += 1
        content = event['content']
        if event['name'] == 'search' and isinstance(content, str):
            docids.update(str(hit['docid']) for hit in json.loads(content))
    return {'query_id': trace['query_id'], 'tool_call_counts': dict(calls),
            'status': 'completed' if result.get('status') == 'finished' else result.get('status', 'interrupted'),
            'retrieved_docids': sorted(docids),
            'result': [{'type': 'output_text', 'output': result.get('answer', '')}]}


def commit(message):
    subprocess.run(['git', 'add', str(HERE), str(ROOT / 'docs/issues.md'),
                    str(ROOT / 'experiments/reports/browsecomp-full-20261015.md')], cwd=ROOT, check=True)
    changed = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT).returncode
    if changed:
        subprocess.run(['git', 'commit', '-m', message], cwd=ROOT, check=True)
