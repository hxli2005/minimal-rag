"""Small shared state, immutable plan, request accounting and paired gates."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import json
import subprocess
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NIGHT = ROOT / 'data/benchmarks/system-nightly/20261015T102540'
OLD = ROOT / 'data/benchmarks/browsecomp-plus/full-20261015'
PRIMARY = ('enterprise', 'browsecomp')

def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)

def now():
    return datetime.now().astimezone().isoformat()

def note(text):
    with (HERE / 'notes.md').open('a') as f:
        f.write(f'- {now()} {text}\n')

def cases(dataset, split='dev'):
    if dataset == 'browsecomp':
        path = ROOT / 'data/benchmarks/browsecomp-plus/queries.jsonl'
        return {str(r['query_id']):r for r in map(json.loads, path.read_text().splitlines())}
    from minimal_rag_experiments.rag_optimization_eval import normalized_case
    return {r['id']:r for r in (normalized_case(dataset, r) for r in read(ROOT / f'data/benchmarks/{dataset}/{split}.json'))}

def samples(stage):
    if stage == 'test':
        return {'enterprise':list(cases('enterprise', 'test'))}
    b = read(ROOT/'experiments/browsecomp-full-20261015/sample-dev100.json')['query_ids']
    e = list(cases('enterprise'))
    n = 30 if stage == 'panel' else 100
    return {'enterprise':e[:n], 'browsecomp':list(map(str,b[:n]))}

class Stopped(Exception):
    pass

class Ledger:
    """Single writer, owned by the inherited launcher lock; reserve before send."""
    def __init__(self, folder=NIGHT):
        self.folder = Path(folder)
        self.plan = read(self.folder/'night-plan.json')
        self.path = self.folder/'requests.jsonl'
        self.entries = [json.loads(x) for x in self.path.read_text().splitlines()] if self.path.exists() else []
        self.deadline = datetime.fromisoformat(self.plan['deadline_at']).timestamp()
    def counts(self):
        sol = [e for e in self.entries if e['provider']=='sol']
        return {'total':len(sol), 'exploration':sum(not e['cleanup'] for e in sol),
                'openrouter':sum(e['provider']=='openrouter' for e in self.entries)}
    def reserve(self, candidate, stage, dataset, qid, role, *, cleanup=False, provider='sol', at=None):
        if cleanup and role != 'judge':
            raise ValueError('cleanup reserve is only for judging existing answers')
        at = datetime.now().timestamp() if at is None else at
        if at >= self.deadline:
            raise Stopped('deadline')
        if role == 'generation' and at >= self.deadline-1800:
            raise Stopped('generation_cutoff')
        n = self.counts()
        if provider == 'sol':
            if n['total'] >= self.plan['sol_limit']:
                raise Stopped('night_budget')
            if not cleanup and n['exploration'] >= self.plan['exploration_limit']:
                raise Stopped('exploration_budget')
            used = sum(e['provider']=='sol' and (stage=='finance' or e['candidate']==candidate) and e['stage']==stage for e in self.entries)
            if used >= self.plan['stage_limits'][stage]:
                raise Stopped('stage_budget')
        entry = {'candidate':candidate,'stage':stage,'dataset':dataset,'id':qid,'role':role,
                 'provider':provider,'cleanup':cleanup,'started':now()}
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('a') as f:
            f.write(json.dumps(entry)+'\n'); f.flush()
        self.entries.append(entry)
        return entry

def initial_state():
    return {'status':'preparing','night_anchor':'v0','current_best':'v0','candidates':[],
            'automation_id':'automation','updated':now()}

def finish_candidate(state, version, adopted, reason):
    state['candidates'][-1].update(status='adopted' if adopted else 'rejected',reason=reason,ended=now())
    if adopted:
        state['current_best'] = version
    state.update(status='awaiting_candidate',active_candidate=None,updated=now())
    return state

def can_start(state, ledger, expected_requests=0, expected_seconds=0, at=None):
    at = datetime.now().timestamp() if at is None else at
    if len(state['candidates']) >= ledger.plan['max_candidates']:
        return 'candidate_limit'
    if ledger.deadline-at <= max(5400, expected_seconds*1.2+1800):
        return 'insufficient_time'
    if ledger.counts()['exploration']+expected_requests*1.2 > ledger.plan['exploration_limit']:
        return 'insufficient_budget'
    if ledger.counts()['total']+expected_requests*1.2 > ledger.plan['sol_limit']:
        return 'insufficient_budget'
    return None

def begin_candidate(state, cfg, ledger, at=None):
    reason=can_start(state,ledger,cfg['expected_requests'],cfg['expected_seconds'],at)
    if reason:raise Stopped(reason)
    if cfg['parent']!=state['current_best']:raise ValueError('candidate does not start from current_best')
    state['candidates'].append({'version':cfg['version'],'parent':state['current_best'],'status':'running','started':now()})
    state.update(status='running',active_candidate=cfg['version'],updated=now())
    return state

def result_folder(version, dataset, split):
    cfg = read(HERE/f'{version}.json')
    key = f'{dataset}_{split}'
    if key in cfg.get('reuse', {}):
        return ROOT / cfg['reuse'][key]
    return NIGHT/version/f'{dataset}-{split}'

def rows(version, dataset, split, selected):
    from minimal_rag.codex_adapter import token_usage
    folder = result_folder(version,dataset,split)
    out = []
    for qid in selected:
        t = read(folder/'traces'/f'{qid}.json', {})
        attempts = [read(p) for p in (folder/'attempts').glob(f'{qid}-*.json')]+([t] if t else [])
        runs = [r for a in attempts for r in a.get('model_runs', [])]
        u = token_usage(runs)
        terminal = bool(t.get('result') and not t.get('recoverable'))
        correct = t.get('judgement', {}).get('correct')
        if dataset == 'browsecomp':
            correct = read(folder/'judgements'/f'{qid}.json',{}).get('judgement',{}).get('correct')
        if terminal and t['result']['status'] != 'finished':
            correct = False
        out.append({'id':qid,'terminal':terminal,'correct':correct,'status':t.get('result',{}).get('status'),
                    'uncached':u.get('input_tokens',0)-u.get('cached_input_tokens',0),
                    'missing_usage':sum(r.get('usage') is None for r in runs),
                    'calls':len(runs),'seconds':sum(a.get('seconds',0) for a in attempts)})
    # A ledger reservation interrupted before its model record must not disappear from cost.
    ledger = read(HERE/f'{version}.json').get('reuse',{})
    if not ledger.get(f'{dataset}_{split}'):
        counts = Counter(e['id'] for e in Ledger().entries if e['candidate']==version and e['dataset']==dataset and e['role']=='generation')
        for row in out:
            row['missing_usage'] += max(0, counts[row['id']]-row['calls'])
    return out

def compare(candidate, baseline, stage, seed=20261016):
    """Only the frozen primary subsets enter metrics; auxiliary keys are ignored."""
    selected = ('enterprise',) if stage=='test' else PRIMARY
    per, differences = {}, []
    cost_c = cost_b = miss_c = miss_b = 0
    for dataset in selected:
        a,b = candidate[dataset],baseline[dataset]
        if [r['id'] for r in a] != [r['id'] for r in b]:
            raise ValueError('paired IDs/order differ')
        if any(not r['terminal'] or r['correct'] is None for r in a+b):
            return {'passed':False,'status':'undetermined','reason':'missing_primary_decisions'}
        d = np.array([int(x['correct'])-int(y['correct']) for x,y in zip(a,b)])
        differences.append(d)
        per[dataset] = {'n':len(a),'candidate_correct':sum(int(r['correct']) for r in a),
            'baseline_correct':sum(int(r['correct']) for r in b),'delta_pp':float(d.mean()*100),
            'wins':int(sum(d==1)),'losses':int(sum(d==-1)),'ties':int(sum(d==0))}
        cost_c += sum(r['uncached'] for r in a); cost_b += sum(r['uncached'] for r in b)
        miss_c += sum(r['missing_usage'] for r in a); miss_b += sum(r['missing_usage'] for r in b)
    if stage=='panel':
        passed = all(x['wins']-x['losses']>=-2 for x in per.values())
        return {'passed':passed,'status':'pass' if passed else 'fail','per_benchmark':per}
    rng = np.random.default_rng(seed)
    boot = np.mean([d[rng.integers(0,len(d),(10000,len(d)))].mean(axis=1) for d in differences],axis=0)
    interval = np.percentile(boot,[2.5,97.5])*100
    ratio = cost_c/cost_b if cost_b else None
    quality = all(x['delta_pp']>=-2-1e-9 for x in per.values()) and interval[0]>=-2-1e-9
    cost = not miss_c and ratio is not None and ratio<=1.10
    return {'passed':bool(quality and cost),'status':'undetermined' if miss_c or ratio is None else ('pass' if quality and cost else 'fail'),
        'per_benchmark':per,'delta_pp':float(np.mean([d.mean() for d in differences])*100),
        'bootstrap_95_pp':interval.tolist(),'cost_ratio':None if miss_c else ratio,
        'cost_ratio_kind':'undetermined' if miss_c else ('upper_bound' if miss_b else 'exact'),
        'recorded_cost_ratio_diagnostic':ratio,
        'candidate_missing_usage':miss_c,'baseline_missing_usage':miss_b,'candidate_uncached':cost_c,'baseline_uncached':cost_b,
        'quality_passed':bool(quality),'cost_passed':bool(cost)}

def gates(version, best, stage):
    split = 'test' if stage=='test' else 'dev'
    def load(v): return {d:rows(v,d,split,ids) for d,ids in samples(stage).items()}
    candidate = load(version)
    return {label:compare(candidate,load(base),stage) for label,base in [('current_best',best),('night_anchor','v0')]}

def commit(message):
    subprocess.run(['git','add',str(HERE),str(HERE/'versions.md'),str(ROOT/'docs/issues.md'),str(ROOT/'experiments/src/minimal_rag_experiments/rag_optimization_eval.py'),str(ROOT/'experiments/reports')],check=True,cwd=ROOT)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
        subprocess.run(['git','commit','-m',message],check=True,cwd=ROOT)
