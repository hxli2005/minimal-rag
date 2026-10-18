"""One isolated process per dataset/stage. Generation and judging have separate gates."""
import argparse
import ast
import asyncio
import importlib
import importlib.util
import json
import os
import signal
import sys
import tempfile
import time
import types
import httpx
from night import HERE, ROOT, NIGHT, OLD, Ledger, Stopped, cases, samples, read, save, now


def restore_history(dataset, restore_adapter=True):
    spec = importlib.util.spec_from_file_location('frozen_loader',ROOT/'experiments/sota-20261014/run_frozen.py')
    loader = importlib.util.module_from_spec(spec); spec.loader.exec_module(loader)
    source = read(loader.HISTORY[dataset]/'code.json')
    extra = read(loader.HISTORY['enterprise']/'code.json')
    if not restore_adapter:
        # Legacy retrieval bundled its tool schemas with the old adapter. Keep
        # those exact definitions without loading or replacing the old transport.
        from dataclasses import dataclass
        from collections.abc import Callable
        from pydantic import BaseModel,ConfigDict,Field
        schema=types.ModuleType('night_legacy_types')
        schema.__dict__.update(dataclass=dataclass,Callable=Callable,BaseModel=BaseModel,ConfigDict=ConfigDict,Field=Field)
        sys.modules[schema.__name__]=schema
        nodes=[n for n in ast.parse(source['agent']).body if isinstance(n,ast.ClassDef) and n.name in ('StrictModel','SearchArgs','ReadArgs','Tool')]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<frozen-tool-schemas>','exec'),schema.__dict__)
    assembly = NIGHT/'assembly'/dataset
    assembly.mkdir(parents=True,exist_ok=True)
    for key,name in loader.MODULES.items():
        if key=='agent' and not restore_adapter:continue
        code = source.get(key,extra[key])
        for old,new in loader.LEGACY.items():
            code = code.replace('from labs.context.'+old+' import','from '+new+' import')
        if not restore_adapter:
            code=code.replace('from minimal_rag.codex_adapter import ReadArgs, SearchArgs, Tool','from night_legacy_types import ReadArgs, SearchArgs, Tool')
            code=code.replace('from minimal_rag.codex_adapter import Tool','from night_legacy_types import Tool')
        code = code.replace('ROOT = Path(__file__).resolve().parents[3]','from minimal_rag.paths import ROOT')
        code = code.replace('ROOT / "model-cache/hybrid-rag/models-minilm.json"','ROOT / "models/hybrid-rag/models-minilm.json"')
        path = assembly/(key+'.py'); path.write_text(code)
        parent_name,attr = name.rsplit('.',1)
        parent = importlib.import_module(parent_name)
        spec = importlib.util.spec_from_file_location(name,path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module; setattr(parent,attr,module); spec.loader.exec_module(module)


def status(error):
    if isinstance(error,Stopped): return str(error)
    m = str(error).lower()
    if 'workspace routing discovery timed out' in m: return 'routing_timeout'
    if any(s in m for s in ('usage limit','usage_limit','quota exceeded','insufficient credits')): return 'usage_limit'
    if any(s in m for s in ('permission denied','sandbox denied','approval required','blocked by sandbox','unauthorized','forbidden')): return 'permission_blocked'
    if isinstance(error,httpx.HTTPStatusError):
        if error.response.status_code in (401,403): return 'permission_blocked'
        if error.response.status_code==402: return 'usage_limit'
    if any(s in m for s in ('429','rate_limit','rate limit')): return 'rate_limited'
    return 'timeout' if isinstance(error,TimeoutError) else 'runtime_error'


async def run(args):
    cfg = read(HERE/f'{args.version}.json')
    settings = cfg[args.dataset if args.dataset!='financebench' else 'enterprise']
    if settings['adapter']=='historical' or settings.get('retrieval_snapshot')=='historical':
        restore_history(args.dataset,restore_adapter=settings['adapter']=='historical')
    from openai_codex import AsyncCodex
    from minimal_rag import codex_adapter as rag
    from minimal_rag.indexed_rag import IndexedKnowledgeBase,indexed_tools
    from minimal_rag.loop import run_agent,Reply
    from minimal_rag_experiments import rag_optimization_eval as evaluation
    from minimal_rag_experiments.rag_judge import PROMPT,Assessment
    sys.path.append(str(ROOT/'experiments/browsecomp-full-20261015'))
    from shared import pilot,pilot_judge,OFFICIAL,source_string
    split = 'test' if args.stage=='test' else 'dev'
    selected = list(cases('financebench')) if args.dataset=='financebench' else samples(args.stage)[args.dataset]
    selected = selected[:args.limit] if args.limit else selected
    dataset_cases = cases(args.dataset,split)
    folder = NIGHT/args.version/f'{args.dataset}-{split}'
    folder.mkdir(parents=True,exist_ok=True)
    state_path = NIGHT/f'{args.version}-{args.stage}-{args.dataset}-state.json'
    ledger = Ledger()
    runtime = read(NIGHT/'runtime.json',{'generation_concurrency':5,'judge_concurrency':1})
    generate_gate,judge_gate = asyncio.Semaphore(runtime['generation_concurrency']),asyncio.Semaphore(1)
    stop = None
    judge_tasks = []
    root_task = asyncio.current_task()
    for sig in (signal.SIGINT,signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig,root_task.cancel)
    excerpt = 4000 if args.stage=='attribution' else settings['excerpt_chars']
    browse = args.dataset=='browsecomp'
    prompt = settings.get('prompt',rag.SYSTEM_PROMPT)
    limit,case_timeout = (30,1800) if browse else (12,900)
    query_template = source_string(OFFICIAL/'search_agent/prompts.py','QUERY_TEMPLATE').replace('get_document','read') if browse else evaluation.NATIVE_FORMAT
    frozen = {'dataset':args.dataset,'split':split,'model':'gpt-6.1-sol','effort':'high','timeout':180,
        'max_turns':limit,'max_calls':limit,'case_timeout':case_timeout,'excerpt_chars':excerpt,
        'prompt':prompt,'query_template':query_template,'assembly':settings,'judge_prompt':PROMPT if not browse else 'official GRADER_TEMPLATE',
        'judge_effort':'medium' if not browse else None,'judge_parameters':pilot_judge.PARAMS if browse else None}
    if read(folder/'config.json',frozen)!=frozen: raise ValueError('frozen configuration changed')
    save(folder/'config.json',frozen)
    db = ROOT/f'data/benchmarks/{"browsecomp-plus" if browse else args.dataset}/corpus.sqlite'
    models = dense = None
    if settings['retrieval']=='hybrid':
        manifest = ROOT/f'models/hybrid-rag/{"models-bge-m3-cascade100.json" if args.dataset=="enterprise" else "models-minilm.json"}'
        data = read(manifest)
        for role in ('embedding','reranker','prefilter'):
            if role in data:
                data[role]['path'] = str((manifest.parent/data[role]['path']).resolve())
        save(NIGHT/'models.json',data)
        models = evaluation.load_models(NIGHT/'models.json')
        dense = evaluation.hybrid.DenseIndex(db.with_suffix('.dense'),db,models)
    save(state_path,{'status':'running','pid':os.getpid(),'updated':now()})
    fatal = ('usage_limit','permission_blocked','routing_timeout','judge_protocol_failed','deadline')
    async with AsyncCodex(rag.rag_codex_config('/private/tmp')) as client:
        account = (await client.account(refresh_token=False)).account
        if getattr(getattr(account,'root',account),'type',None)!='chatgpt': raise Stopped('permission_blocked')

        async def grade(qid,backfill=False):
            nonlocal stop
            async with judge_gate:
                trace_path = folder/'traces'/f'{qid}.json'
                t = read(trace_path,{})
                if not t.get('result') or t.get('recoverable'): return
                judge_path = folder/'judgements'/f'{qid}.json' if browse else trace_path
                report = read(judge_path,{'query_id':qid,'attempts':[]})
                if report.get('judgement',{}).get('correct') is not None: return
                if t['result']['status']!='finished':
                    report['judgement'] = {'correct':False,'reason':'Generation failed; counted as incorrect.'}
                    save(judge_path,report); return
                attempts = report.setdefault('attempts' if browse else 'judge_attempts',[])
                if attempts and not backfill: return
                c = dataset_cases[qid]
                payload = {'question':c.get('question'),'reference_answer':c.get('gold'),
                           'reference_facts':c.get('facts'),'candidate_answer':t['result']['answer']}
                count_responses = lambda: sum(bool(a.get('received_response') or 'response' in a) for a in attempts)
                while count_responses()<3 and stop not in fatal:
                    delay = runtime.get('judge_cooldown_until',0)-time.time()
                    if delay>0: await asyncio.sleep(delay)
                    try:
                        ledger.reserve(args.version,args.stage,args.dataset,qid,'judge',cleanup=backfill,
                                       provider='openrouter' if browse else 'sol')
                    except Stopped as error:
                        stop = str(error); return
                    a = {'started':now(),'backfill':backfill}; attempts.append(a)
                    save(judge_path,report)
                    try:
                        if browse:
                            template = source_string(OFFICIAL/'scripts_evaluation/evaluate_run.py','GRADER_TEMPLATE')
                            judge_prompt = template.format(question=c['query'],response=t['result']['answer'],correct_answer=c['answer'])
                            report['prompt']=judge_prompt
                            raw = await asyncio.to_thread(pilot_judge.router,judge_prompt)
                            a['response'] = raw
                            m = raw['choices'][0]['message']
                            reasoning = bool(m.get('reasoning') or m.get('reasoning_content') or m.get('reasoning_details'))
                            tokens = raw.get('usage',{}).get('completion_tokens_details',{}).get('reasoning_tokens')
                            if raw.get('provider')!='DeepInfra' or reasoning or tokens!=0: raise Stopped('judge_protocol_failed')
                            a.update(reasoning_present=reasoning,reasoning_tokens=tokens)
                            parsed = pilot.parse_judge_response(m.get('content') or '')
                            if parsed.get('correct') is None: raise ValueError('unparseable grade')
                        else:
                            report['judge_input']=payload
                            with tempfile.TemporaryDirectory(prefix='night-judge-') as cwd:
                                model = rag.CodexModel(client,cwd,'gpt-6.1-sol',180,effort='medium',system_prompt=PROMPT)
                                a['runs'] = model.runs
                                try:
                                    raw = await model([{'role':'user','content':json.dumps(payload)}],[])
                                finally:
                                    a['received_response'] = any(r.get('output') is not None for r in model.runs)
                                a['response'] = raw
                                reply = Reply.model_validate_json(raw)
                                if reply.tool_calls: raise ValueError('judge tools')
                                parsed = Assessment.model_validate_json(reply.text).model_dump()
                                if len(parsed['facts_supported'])!=len(c['facts']): raise ValueError('fact count mismatch')
                        a.update(status='completed',assessment=parsed)
                        report.update(judgement=parsed,first_successful_attempt=len(attempts))
                        print(json.dumps({'judge':args.dataset,'id':qid,'correct':parsed['correct']}),flush=True)
                        return
                    except asyncio.CancelledError:
                        a.update(status='interrupted'); raise
                    except Exception as error:
                        category = status(error)
                        a.update(status='failed',error_type=type(error).__name__,category=category)
                        if category in fatal: stop=category
                        if category=='rate_limited':
                            runtime['judge_cooldown_until']=time.time()+60; save(NIGHT/'runtime.json',runtime)
                        if not (a.get('received_response') or 'response' in a) or stop in fatal: return
                    finally:
                        save(judge_path,report)

        async def generate(qid):
            nonlocal stop
            async with generate_gate:
                path = folder/'traces'/f'{qid}.json'
                old = read(path)
                if old and old.get('result') and not old.get('recoverable'):
                    judge_tasks.append(asyncio.create_task(grade(qid))); return
                if stop: return
                if old:
                    save(folder/'attempts'/f'{qid}-{old.get("attempt",1)}.json',old)
                t = {'id':qid,'query_id':qid,'attempt':old.get('attempt',1)+1 if old else 1,'started':now(),
                     'events':[],'model_runs':[],'native_answer':True,'recoverable':False}
                start = time.monotonic()
                kb = None
                def persist(): save(path,t)
                def emit(event): t['events'].append(event); persist()
                persist()
                try:
                    if models:
                        cls = evaluation.PreviewKnowledgeBase if args.dataset=='enterprise' else evaluation.hybrid.HybridKnowledgeBase
                        extra = {'preview_count':15} if args.dataset=='enterprise' else {}
                        kb = cls(db,models,dense,excerpt,candidates=100,rerank=200,mode='hybrid',**extra)
                        toolset = evaluation.preview_tools(kb) if extra else evaluation.hybrid.hybrid_tools(kb)
                    else:
                        kb = IndexedKnowledgeBase(db,excerpt); toolset=indexed_tools(kb)
                    with tempfile.TemporaryDirectory(prefix='night-generate-') as cwd:
                        model = rag.CodexModel(client,cwd,'gpt-6.1-sol',180,effort='high',system_prompt=prompt,max_calls=limit)
                        t['model_runs']=model.runs
                        async def complete(messages,specs):
                            if stop: raise Stopped(stop)
                            ledger.reserve(args.version,args.stage,args.dataset,qid,'generation')
                            try: return await model(messages,specs)
                            finally: persist()
                        c = dataset_cases[qid]
                        question = query_template.format(Question=c['query']) if browse else query_template+c['question']
                        t['result'] = await asyncio.wait_for(run_agent(complete,toolset,question,max_turns=limit,emit=emit),case_timeout)
                except asyncio.CancelledError:
                    t.update(result={'status':'interrupted','answer':''},recoverable=True); raise
                except Exception as error:
                    category = status(error)
                    if category in fatal or category in ('rate_limited','stage_budget','night_budget','exploration_budget','generation_cutoff'):
                        stop=category
                    t.update(result={'status':category,'answer':'','error_type':type(error).__name__,'detail':str(error)},
                             recoverable=category in ('routing_timeout','rate_limited','stage_budget','night_budget','exploration_budget','generation_cutoff','deadline'))
                    if category=='rate_limited':
                        runtime['generation_concurrency']=max(1,runtime['generation_concurrency']//2); save(NIGHT/'runtime.json',runtime)
                finally:
                    if kb: kb.close()
                    t.update(seconds=time.monotonic()-start,ended=now()); persist()
                    print(json.dumps({'generation':args.dataset,'id':qid,'status':t.get('result',{}).get('status'),'calls':len(t['model_runs'])}),flush=True)
                judge_tasks.append(asyncio.create_task(grade(qid)))
        try:
            if args.judge_only:
                for qid in selected: await grade(qid,True)
            else:
                await asyncio.gather(*(generate(qid) for qid in selected))
                await asyncio.gather(*judge_tasks)
        finally:
            if dense: dense.close()
            save(state_path,{'status':stop or 'completed','pid':os.getpid(),'updated':now(),'counts':ledger.counts()})
    return stop

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('version');p.add_argument('stage');p.add_argument('dataset')
    p.add_argument('--limit',type=int);p.add_argument('--judge-only',action='store_true')
    args=p.parse_args()
    try:
        reason=asyncio.run(run(args))
    except BaseException as error:
        reason=status(error)
        save(NIGHT/f'{args.version}-{args.stage}-{args.dataset}-state.json',{'status':reason,'error_type':type(error).__name__,'detail':str(error),'updated':now()})
    sys.exit(0 if not reason else 2)
