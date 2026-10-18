"""Frozen generation with stage ledgers and immediate, concurrent first-valid judging."""
import argparse
import asyncio
import json
import os
import signal
import tempfile
import time
import httpx
from openai_codex import AsyncCodex
from minimal_rag.codex_adapter import CodexModel, rag_codex_config
from minimal_rag.indexed_rag import IndexedKnowledgeBase, indexed_tools
from minimal_rag.loop import run_agent
from shared import (DATA, HERE, OFFICIAL, OUT, LIMITS, config, count, directory, export, ids,
                    note, now, pilot, pilot_judge, queries, read, save, source_string)
from standard_tools import standard_tools, tokenizer


class RunStopped(Exception):
    pass


def judge_response_count(report):
    # Model completion responses count, including unparseable ones; transport
    # failures and HTTP error responses contain no judge completion.
    return sum('response' in attempt for attempt in report.get('attempts', []))


def needs_judgement(report):
    return report.get('judgement', {}).get('correct') is None and judge_response_count(report) < 3


async def grade(group, qid, cases, stop, *, backfill=False):
    folder = directory(group)
    answer = read(folder / 'official' / f'{qid}.json')
    if not answer or answer['status'] != 'completed' or stop['reason'] in ('usage_limit', 'permission_blocked', 'judge_protocol_failed'):
        return
    path = folder / 'judgements' / f'{qid}.json'
    report = read(path, {'query_id': qid, 'mode': 'openrouter', 'attempts': []})
    if not needs_judgement(report):
        return
    if report.get('attempts') and not backfill:
        return  # Deferred failures are retried together, never selected by score.
    template = source_string(OFFICIAL / 'scripts_evaluation/evaluate_run.py', 'GRADER_TEMPLATE')
    prompt = template.format(question=cases[qid]['query'], response=answer['result'][-1]['output'],
                             correct_answer=cases[qid]['answer'])
    report['prompt'] = prompt
    while judge_response_count(report) < 3:
        if stop['reason'] in ('usage_limit', 'permission_blocked', 'judge_protocol_failed'):
            return
        cooldown = read(OUT / 'runtime.json', {}).get('judge_cooldown_until', 0) - time.time()
        if cooldown > 0:
            if not backfill:
                return
            await asyncio.sleep(cooldown)
        attempt = {'started': now(), 'parameters': pilot_judge.PARAMS,
                   'assistant_prefill': pilot_judge.NON_THINKING_PREFIX, 'backfill': backfill}
        report['attempts'].append(attempt)
        save(path, report)
        if backfill:
            number = sum(a.get('backfill', False) for a in report['attempts'])
            note(f'补判{group}/{qid}：累计补判尝试{number}次（本次已记账），此前收到模型响应{judge_response_count(report)}/3；原始尝试全部保留。')
        try:
            raw = await asyncio.to_thread(pilot_judge.router, prompt)
            attempt['response'] = raw
            message = raw['choices'][0]['message']
            reasoning = bool(message.get('reasoning') or message.get('reasoning_content') or message.get('reasoning_details'))
            tokens = raw.get('usage', {}).get('completion_tokens_details', {}).get('reasoning_tokens')
            attempt.update(reasoning_present=reasoning, reasoning_tokens=tokens)
            if raw.get('provider') != 'DeepInfra' or reasoning or tokens != 0:
                stop['reason'] = 'judge_protocol_failed'
                raise RuntimeError('Judge provider or zero-reasoning validation failed')
            attempt['text'] = message.get('content') or ''
            parsed = pilot.parse_judge_response(attempt['text'])
            attempt['parsed'] = parsed
            if parsed['correct'] is None:
                raise ValueError('No parseable correct field')
            attempt['status'] = 'completed'
            report.update(judgement=parsed, first_successful_attempt=len(report['attempts']),
                          received_response_attempts=judge_response_count(report), pending_backfill=False)
            save(path, report)
            print(json.dumps({'group': group, 'query_id': qid, 'correct': parsed['correct'], 'judge_attempts': len(report['attempts'])}), flush=True)
            return
        except Exception as error:
            attempt.update(status='failed', error_type=type(error).__name__)
            if isinstance(error, httpx.HTTPStatusError):
                attempt['http_status'] = error.response.status_code
                if error.response.status_code == 429:
                    runtime = read(OUT / 'runtime.json', {})
                    previous = runtime.get('judge_concurrency', 1)
                    runtime.update(judge_concurrency=max(1, previous // 2), judge_cooldown_until=time.time()+60)
                    save(OUT / 'runtime.json', runtime)
                    note(f'裁判HTTP429：仅裁判并发{previous}→{runtime["judge_concurrency"]}，冷却60秒；生成并发{runtime.get("generation_concurrency",5)}不变。')
            report['received_response_attempts'] = judge_response_count(report)
            report['pending_backfill'] = needs_judgement(report)
            save(path, report)
            note(f'裁判{group}/{qid}累计尝试{len(report["attempts"])}次，模型响应{judge_response_count(report)}/3：{type(error).__name__}；无模型响应则进入统一补判队列。')
            if stop['reason'] == 'judge_protocol_failed' or 'response' not in attempt:
                return
            if judge_response_count(report) < 3:
                await asyncio.sleep(2)


def error_status(error):
    message = str(error).lower()
    if isinstance(error, RunStopped):
        return str(error)
    if any(w in message for w in ['usage limit', 'usage_limit', 'usage_limit_reached', 'quota exceeded']):
        return 'usage_limit'
    if any(w in message for w in ['429', 'rate limit', 'rate_limit']):
        return 'rate_limited'
    if any(w in message for w in ['permission denied', 'sandbox denied', 'approval required', 'blocked by sandbox']):
        return 'permission_blocked'
    return 'timeout' if isinstance(error, TimeoutError) else 'runtime_error'


async def run_stage(group, stage, sample_name, concurrency=5, limit=None):
    if not os.environ.get('OPENROUTER_API_KEY'):
        raise RuntimeError('OPENROUTER_API_KEY missing; no substitute judge permitted')
    settings = config(group, stage)
    selected, cases = ids(sample_name)[:limit], queries()
    folder, stop = directory(group), {'reason': None}
    tok = tokenizer() if group == 'B' else None
    template = source_string(OFFICIAL / 'search_agent/prompts.py',
                             'QUERY_TEMPLATE_NO_GET_DOCUMENT' if group == 'B' else 'QUERY_TEMPLATE')
    if group != 'B':
        template = template.replace('get_document', 'read')
    state_path = OUT / f'{stage}-{group}-state.json'
    concurrency = read(OUT / 'runtime.json', {}).get('generation_concurrency', concurrency)
    semaphore, judge_condition = asyncio.Semaphore(concurrency), asyncio.Condition()
    judge_active = 0
    judge_tasks = []
    started = now()
    def state(status):
        value = {'status': status, 'stage': stage, 'group': group, 'started': started, 'updated': now(),
                 'concurrency': concurrency, 'generation_concurrency': concurrency,
                 'judge_concurrency': read(OUT / 'runtime.json', {}).get('judge_concurrency', 1),
                 'generation_requests': count(stage), 'limit': LIMITS[stage]}
        if status == 'rate_limited':
            value['next_concurrency'] = max(1, concurrency // 2)
            runtime = read(OUT / 'runtime.json', {})
            runtime.update(generation_concurrency=value['next_concurrency'], reason='generation rate limit')
            save(OUT / 'runtime.json', runtime)
        save(state_path, value)
    state('running')
    async def judge_one(qid):
        nonlocal judge_active
        async with judge_condition:
            await judge_condition.wait_for(lambda: judge_active < read(OUT / 'runtime.json', {}).get('judge_concurrency', 1))
            judge_active += 1
        try:
            await grade(group, qid, cases, stop)
        finally:
            async with judge_condition:
                judge_active -= 1
                judge_condition.notify_all()
    root_task = asyncio.current_task()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, root_task.cancel)
    async with AsyncCodex(rag_codex_config('/private/tmp')) as client:
        account = (await client.account(refresh_token=False)).account
        if getattr(getattr(account, 'root', account), 'type', None) != 'chatgpt':
            raise RuntimeError('Requires existing ChatGPT subscription login')

        async def generate(qid):
            async with semaphore:
                path = folder / 'traces' / f'{qid}.json'
                old, attempt = read(path), 1
                if old and old.get('result') and not old.get('recoverable'):
                    save(folder / 'official' / f'{qid}.json', export(old))
                    judge_tasks.append(asyncio.create_task(judge_one(qid)))
                    return
                if stop['reason'] or count(stage) >= LIMITS[stage]:
                    if not stop['reason']:
                        stop['reason'] = 'stage_budget_exhausted'
                    return
                if old:
                    attempt = old['attempt'] + 1
                    archive = folder / 'attempts' / f"{qid}-{old['attempt']}.json"
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    path.rename(archive)
                    note(f'决定：恢复{group}/{qid}中断尝试并保留原始文件和账本；理由：{old.get("result", {}).get("status", "no_terminal_result")}，不按答案分数重跑。')
                trace = {'query_id': qid, 'group': group, 'stage': stage, 'attempt': attempt,
                         'question': cases[qid]['query'], 'started': now(), 'events': [],
                         'model_runs': [], 'recoverable': False}
                start = time.monotonic()
                kb = IndexedKnowledgeBase(DATA / 'corpus.sqlite', excerpt_chars=settings['excerpt_chars'] or 2000)
                def persist():
                    save(path, trace)
                def emit(event):
                    trace['events'].append(event)
                    persist()
                persist()
                try:
                    with tempfile.TemporaryDirectory(prefix='browsecomp-full-') as cwd:
                        model = CodexModel(client, cwd, model=settings['model'], effort=settings['effort'],
                                           timeout=settings['request_timeout'], max_calls=settings['max_calls'])
                        trace['model_runs'] = model.runs
                        async def complete(messages, specs):
                            if stop['reason']:
                                raise RunStopped(stop['reason'])
                            # No await between ledger check and append: one process owns this stage.
                            if count(stage) >= LIMITS[stage]:
                                stop['reason'] = 'stage_budget_exhausted'
                                raise RunStopped(stop['reason'])
                            with (OUT / f'{stage}-requests.jsonl').open('a') as ledger:
                                ledger.write(json.dumps({'group': group, 'query_id': qid, 'attempt': attempt,
                                                         'call': len(model.runs)+1, 'started': now()}) + '\n')
                            try:
                                return await model(messages, specs)
                            finally:
                                persist()
                        tools = standard_tools(kb, tok) if group == 'B' else indexed_tools(kb)
                        trace['result'] = await asyncio.wait_for(run_agent(complete, tools,
                            template.format(Question=cases[qid]['query']), max_turns=settings['max_turns'], emit=emit),
                            settings['case_timeout'])
                except asyncio.CancelledError:
                    trace.update(result={'status': 'interrupted', 'answer': ''}, recoverable=True)
                    raise
                except Exception as error:
                    status = error_status(error)
                    if status in ('usage_limit', 'rate_limited', 'permission_blocked'):
                        if stop['reason'] not in ('usage_limit', 'permission_blocked'):
                            stop['reason'] = status
                    trace.update(result={'status': status, 'answer': '', 'error_type': type(error).__name__, 'detail': str(error)},
                                 recoverable=status in ('usage_limit', 'rate_limited', 'interrupted'))
                finally:
                    kb.close()
                    trace.update(seconds=time.monotonic()-start, ended=now())
                    persist()
                    save(folder / 'official' / f'{qid}.json', export(trace))
                    print(json.dumps({'group': group, 'query_id': qid, 'status': trace.get('result', {}).get('status'),
                                      'calls': len(trace['model_runs']), 'stage_requests': count(stage)}), flush=True)
                judge_tasks.append(asyncio.create_task(judge_one(qid)))
        try:
            await asyncio.gather(*(generate(qid) for qid in selected))
            await asyncio.gather(*judge_tasks)
        except asyncio.CancelledError:
            stop['reason'] = stop['reason'] or 'interrupted'
            raise
        finally:
            state(stop['reason'] or 'completed')
    if stop['reason']:
        note(f'阶段{stage}/{group}停止：{stop["reason"]}，账本{count(stage)}/{LIMITS[stage]}；权限或额度问题不绕过，限流并发降至{max(1, concurrency//2)}。')
    return stop['reason'] or 'completed'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('group', choices=['A', 'A-lean', 'B'])
    parser.add_argument('stage', choices=LIMITS)
    parser.add_argument('sample')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--concurrency', type=int, default=5)
    args = parser.parse_args()
    try:
        result = asyncio.run(run_stage(args.group, args.stage, args.sample, args.concurrency, args.limit))
    except Exception as error:
        status = 'routing_timeout' if 'workspace routing discovery timed out' in str(error) else error_status(error)
        save(OUT / f'{args.stage}-{args.group}-state.json',
             {'status':status,'stage':args.stage,'group':args.group,'updated':now(),
              'generation_requests':count(args.stage),'error_type':type(error).__name__})
        raise
    raise SystemExit(0 if result == 'completed' else 2)
