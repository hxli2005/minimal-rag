"""Run the unchanged official scorer, preserving first valid grades and all attempts."""
import fcntl,json,os,shutil,signal,subprocess,time
from pathlib import Path
ROOT=Path('/root/autodl-tmp/browsecomp-frozen-judge-20261017');BUNDLE=ROOT/'bundle'
lock=(ROOT/'official.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
control=json.loads(ROOT.joinpath('control.json').read_text())
model=(ROOT/'model-path.txt').read_text().strip()
env=dict(os.environ,PATH='/root/miniconda3/bin:'+os.environ.get('PATH',''),CUDA_VISIBLE_DEVICES='0',UV_CACHE_DIR=str(ROOT/'uv-cache'),UV_PYTHON_DOWNLOADS='never',GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='url.file://'+str(ROOT/'tevatron.git')+'.insteadOf',GIT_CONFIG_VALUE_0='https://github.com/texttron/tevatron.git')
env.pop('PYTHONPATH',None)
state_path=ROOT/'official-state.json'
state=json.loads(state_path.read_text()) if state_path.exists() else {'completed':[],'attempts':[],'started':time.time()}
state.update(status='running',pid=os.getpid())
def save():
 p=state_path.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2)+'\n');p.replace(state_path)
def evaluate(group,label):
 left=control['inference_deadline']-time.time()
 if left<=0:raise RuntimeError('inference_deadline')
 cmd=['/root/miniconda3/bin/uv','run','--frozen','python','scripts_evaluation/evaluate_run_single_gpu.py','--input_dir','runs/'+group,'--tensor_parallel_size','1','--model',model]
 state.update(phase=label,phase_started=time.time());save()
 attempt={'group':group,'label':label,'started':time.time(),'log':label+'.log'};state['attempts'].append(attempt);save()
 with (ROOT/(label+'.log')).open('a') as log:
  p=subprocess.Popen(cmd,cwd=BUNDLE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
  state['child_pid']=p.pid;save()
  try:code=p.wait(timeout=left)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise RuntimeError('inference_deadline')
 attempt.update(exit_code=code,ended=time.time());save()
 if code:raise RuntimeError('official_scorer_exit_'+str(code))
def records(group):
 return {str(json.loads(p.read_text())['query_id']):json.loads(p.read_text()) for p in (BUNDLE/'evals'/group).glob('*_eval.json')}
def valid(r):
 j=r.get('judge_result',{})
 return not r.get('is_completed') or (isinstance(j.get('correct'),bool) and not j.get('parse_error'))
def repair(group,ids):
 # Only missing or malformed grades are retried, never correct/incorrect decisions.
 for round_ in range(1,4):
  rs=records(group);pending=[q for q in ids if q not in rs or not valid(rs[q])]
  if not pending:return
  eligible=[]
  for q in pending:
   archive=BUNDLE/'evals/attempts'/group;archive.mkdir(parents=True,exist_ok=True)
   old=list(archive.glob(q+'-*.json'))
   received=len(old)+(1 if q in rs and rs[q].get('judge_response') is not None else 0)
   if received>=3:continue
   eligible.append(q)
   current=BUNDLE/'evals'/group/(q+'_eval.json')
   if current.exists():current.rename(archive/(q+'-'+str(len(old)+1)+'.json'))
  if not eligible:return
  # Other invalid exhausted grades stay cached. Valid grades are never removed.
  state['backfill_ids']=eligible;save();evaluate(group,group+'-backfill-'+str(round_))
try:
 assert json.loads((ROOT/'setup-state.json').read_text())['status']=='ready'
 full=ROOT/'A-complete'
 if not full.exists():shutil.move(BUNDLE/'runs/frozen-20261017',full);(BUNDLE/'runs/frozen-20261017').mkdir()
 if 'smoke' not in state['completed']:
  for q in control['smoke_ids']:shutil.copy2(full/(q+'.json'),BUNDLE/'runs/frozen-20261017'/(q+'.json'))
  evaluate('frozen-20261017','smoke');repair('frozen-20261017',control['smoke_ids'])
  rs=records('frozen-20261017');assert all(q in rs and valid(rs[q]) and rs[q].get('judge_response') is not None for q in control['smoke_ids'])
  state['completed'].append('smoke');save()
 for f in full.glob('*.json'):shutil.copy2(f,BUNDLE/'runs/frozen-20261017'/f.name)
 for group,n in control['groups'].items():
  ids=[p.stem for p in (BUNDLE/'runs'/group).glob('*.json')];assert len(ids)==n
  if group not in state['completed']:
   evaluate(group,group+'-full');repair(group,ids)
   rs=records(group);missing=[q for q in ids if q not in rs or not valid(rs[q])]
   state[group]={'expected':n,'evaluations':len(rs),'missing_or_invalid':missing,'summary':json.loads((BUNDLE/'evals'/group/'evaluation_summary.json').read_text())}
   state['completed'].append(group);save()
 state.update(status='completed' if all(not state[g]['missing_or_invalid'] for g in control['groups']) else 'completed_with_evaluation_limitations',ended=time.time());save()
except BaseException as e:
 state.update(status='failed',error_type=type(e).__name__,reason=str(e),ended=time.time());save();raise
