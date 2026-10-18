"""Summarize returned official results; never changes or reruns a judgement."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import copy,hashlib,importlib.util,json,re,sys
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=ROOT/'data/benchmarks/browsecomp-plus/final-20261015';OLD=ROOT/'data/benchmarks/browsecomp-plus/full-20261015'
FULL=ROOT/'experiments/browsecomp-full-20261015'
sys.path.insert(0,str(FULL))
spec=importlib.util.spec_from_file_location('prior_browsecomp_report',FULL/'report.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def ids(name):return list(map(str,read(FULL/f'sample-{name}.json')['query_ids']))
verification=read(OUT/'retrieval-verification.json');runstate=read(OUT/'runtime-evidence/official-state.json')
assert runstate['status']=='completed'
sm,ev,flags={},{},{}
for g,n in [('A',830),('B',230)]:
 sm[g]=read(OUT/f'evals/{g}/evaluation_summary.json')
 ev[g]={str(r['query_id']):r for r in map(read,(OUT/f'evals/{g}').glob('*_eval.json'))}
 flags[g]={str(r['query_id']):r['correct'] for r in sm[g]['per_query_metrics']}
 assert len(ev[g])==len(flags[g])==n
samples={'A830':ids('all830'),'A_dev100':ids('dev100'),'A_remaining730':ids('remaining730'),'B230':ids('pilot30')+ids('b200')}
assert set(samples['A_dev100']).isdisjoint(samples['A_remaining730']) and set(samples['A_dev100']+samples['A_remaining730'])==set(flags['A'])
results={}
for name,selected in samples.items():
 g='B' if name=='B230' else 'A';k=sum(flags[g][q] for q in selected);n=len(selected)
 results[name]={'n':n,'correct':k,'accuracy_percent':100*k/n,'wilson95_percent':prior.wilson(k,n)}
def official_rows(g,selected):
 rows=prior.rows(g,selected)
 for r in rows:
  q=r['query_id'];r['correct']=flags[g][q];r['recall']=ev[g][q]['retrieval']['recall']
 return rows
paired_ids=samples['B230'];a=official_rows('A',paired_ids);b=official_rows('B',paired_ids)
pair=prior.paired(a,b);paired={k:pair[k] for k in ['planned','paired','bootstrap_samples','seed','wins','losses','ties','accuracy_pp']};assert paired['paired']==230
paired.update(A_correct=sum(r['correct'] is True for r in a),B_correct=sum(r['correct'] is True for r in b))
costs={}
for name,rows in [('A830',prior.rows('A',samples['A830'])),('A_paired230',a),('B230',b)]:
 costs[name]={'n':len(rows),'recorded_uncached_input':sum(r['uncached_input_tokens'] for r in rows),
  'recorded_mean_uncached_input':sum(r['uncached_input_tokens'] for r in rows)/len(rows),'generation_requests':sum(r['generation_requests'] for r in rows),
  'missing_usage_calls':sum(r['missing_usage_calls'] for r in rows)}
costs['paired_recorded_ratio']=costs['A_paired230']['recorded_uncached_input']/costs['B230']['recorded_uncached_input']
costs['paired_ratio_interpretation']='recorded-only diagnostic, not an exact saving rate' if costs['A_paired230']['missing_usage_calls'] or costs['B230']['missing_usage_calls'] else 'exact'
agreement={}
for g in ('A','B'):
 common=[];disagree=[];missing=0;generation_failure=0;confidence_missing=0
 for q,r in ev[g].items():
  if not r['is_completed']:generation_failure+=1;continue
  official=r['judge_result']['correct'];assert isinstance(official,bool)
  confidence_missing+=r['judge_result'].get('confidence') is None
  p=OLD/g/'judgements'/f'{q}.json';proxy=read(p).get('judgement',{}).get('correct') if p.exists() else None
  if not isinstance(proxy,bool):missing+=1;continue
  common.append(q)
  if proxy!=official:disagree.append({'query_id':q,'proxy':proxy,'official':official})
 agreement[g]={'common_valid':len(common),'agree':len(common)-len(disagree),'agreement_percent':100*(len(common)-len(disagree))/len(common),
  'disagreements':disagree,'missing_proxy_decisions':missing,'generation_failures_excluded':generation_failure,'official_confidence_missing':confidence_missing}
common=sum(x['common_valid'] for x in agreement.values());agree=sum(x['agree'] for x in agreement.values())
agreement['combined']={'common_valid':common,'agree':agree,'agreement_percent':100*agree/common}
meta=read(OUT/'evals/model-source.txt');assert meta['runtime']['dtype_config']=='bfloat16' and meta['weights']['shards']==17
w=read(OUT/'runtime-evidence/weight-verification.json')
log=(OUT/'runtime-evidence/A-full.log').read_text();first=re.search(r'Evaluating:[^\r\n]*\b1/13[^\r\n]*?([\d.]+)s/batch',log)
assert first
loads=[]
for label in ['smoke','A-full','B-full']:
 loads.extend(float(x) for x in re.findall(r'Model loading took [\d.]+ GiB and ([\d.]+) seconds',(OUT/'runtime-evidence'/f'{label}.log').read_text()))
initial_setup=float(re.search(r'PHASE extract START ([\d.]+)',(OUT/'runtime-evidence/gpu-setup.log').read_text()).group(1))
env_done=read(OUT/'runtime-evidence/setup-state.json')['environment_completed']
timings={'environment_preparation_span_seconds':env_done-initial_setup,'weights_final_process_including_verification_seconds':w['ended']-w['started'],
 'first64_seconds':float(first.group(1)),'model_weight_load_seconds_per_worker':loads,
 'official_processes':[{**x,'seconds':x['ended']-x['started']} for x in runstate['attempts']],
 'official_total_seconds':runstate['ended']-runstate['started'],'note':'Cold start/compilation included in process durations; install and download overlap. Earlier transport recovery is additional.'}
assert [x['label'] for x in runstate['attempts']]==['smoke','A-full','B-full']
plan=read(OUT/'final-plan.json');final=read(OUT/'final-state.json');billing=read(OUT/'billing-status.json')
upper=(datetime.fromisoformat(final['shutdown_verified_at'])-datetime.fromisoformat(plan['start_at'])).total_seconds()/3600*plan['gpu_hourly_rmb']
billing.update(conservative_time_window_estimate_upper_rmb=upper,basis='Treat the entire post-final preparation-to-shutdown-verification window as billable, although GPU creation was later. Not a settled invoice.');save(OUT/'billing-status.json',billing)
eval_date=datetime.fromtimestamp(runstate['ended'],ZoneInfo('America/Los_Angeles')).date().isoformat()
submission=copy.deepcopy(sm['A']);submission.update(LLM='gpt-6.1-sol (effort high)',Retriever='SQLite FTS5 BM25 (1600-char chunks, title/body weights 2/1), search/read, 2000-char search excerpts',Link='',**{'Evaluation Date':eval_date})
assert submission['per_query_metrics']==sm['A']['per_query_metrics'] and 'Search Calls' not in submission
save(OUT/'submission.json',submission)
save(HERE/'submission.json',submission)
scaffold='''Single-agent gpt-6.1-sol (high), at most 30 model calls/turns per question; no external web.\nSQLite FTS5 BM25 over the fixed corpus: 1600-character chunks, 200-character overlap, title/body weights 2/1.\nsearch returns 1–20 documents with 2000-character excerpts; read defaults to 4000 characters and supports paging.\nThe dev100 was inspected; three later nightly candidates were rejected and the original A/v0 remained frozen.\nScored with official 0469490, Qwen3-32B BF16 and TP=2; first valid judgements retained.\n'''
(OUT/'scaffold.txt').write_text(scaffold)
(HERE/'scaffold.txt').write_text(scaffold)
summary={'selected_version':'v0','selected_group':'A','official_script_revision':'046949032b0328319cc9a02663a759ec601d9402','evaluation_date_client':eval_date,'evaluation_date_raw_server':sm['A']['Evaluation Date'],
 'results':results,'official_summaries':{g:{k:v for k,v in s.items() if k!='per_query_metrics'} for g,s in sm.items()},'A_minus_B_paired230':paired,'costs':costs,'judge_agreement':agreement,'timings':timings,
 'new_sol_requests':0,'new_openrouter_requests':0,'new_GPU_model_responses':sum(v['valid_model_judgements'] for v in verification['groups'].values()),'generation_failures':6,'backfill_model_responses':0,'retrieval_verification':verification,
 'submission_ready':True,'submission_sha256':hashlib.sha256((OUT/'submission.json').read_bytes()).hexdigest(),'billing':billing,
 'cleanup':{'private_payload_removed':read(OUT/'cleanup-receipt.json')['status']=='private_payload_removed','gpu_active':final['gpu_active'],'paid_storage_gb':0,'retained':'public environment on free disk only'},
 'limits':['Known dev/regression samples; no unseen-generalization claim.','No candidate reselection after official grading.','Official calibration implementation retained, including its omitted final bin.','Local official-script result; not maintainer-verified leaderboard placement.','Server summary date uses Asia/Shanghai; submission metadata uses America/Los_Angeles. Raw summaries unchanged.']}
save(HERE/'summary.json',summary)
print(json.dumps({k:summary[k] for k in ['results','A_minus_B_paired230','costs','timings','billing']},ensure_ascii=False,indent=2))
print('agreement',json.dumps({g:{k:v for k,v in a.items() if k!='disagreements'} for g,a in agreement.items()}))
