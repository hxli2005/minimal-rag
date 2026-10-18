"""Run the frozen search/read pilot; never print decrypted questions or answers."""
import argparse
import asyncio
import json
import signal
import tempfile
import time

from openai_codex import AsyncCodex
from minimal_rag.codex_adapter import CodexModel, rag_codex_config
from minimal_rag.indexed_rag import IndexedKnowledgeBase, indexed_tools
from minimal_rag.loop import run_agent
from common import DATA, OFFICIAL, OUT, export, note, now, queries, read, request_count, sample, save, source_string


class RunStopped(Exception):
    pass


async def main(args):
    OUT.mkdir(parents=True, exist_ok=True)
    config = {'model': 'gpt-6.1-sol', 'effort': 'high', 'request_timeout': 180, 'case_timeout': 1800,
              'max_turns': 30, 'max_calls': 30, 'excerpt_chars': 2000, 'retrieval': 'SQLite FTS5 BM25',
              'generation_application_request_limit': 1000, 'sdk_http_retries_observable': False,
              'query_template_change': 'get_document -> read', 'data': read(DATA / 'index.json')}
    if read(OUT / 'config.json', config) != config:
        raise ValueError('Existing frozen configuration differs')
    save(OUT / 'config.json', config)
    template = source_string(OFFICIAL / 'search_agent/prompts.py', 'QUERY_TEMPLATE').replace('get_document', 'read')
    cases, gate, stop = queries(), asyncio.Semaphore(args.concurrency), {'reason': None}
    save(OUT / 'run-state.json', {'status': 'running', 'started': now(), 'concurrency': args.concurrency})
    root_task = asyncio.current_task()
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, root_task.cancel)
    async with AsyncCodex(rag_codex_config('/private/tmp')) as client:
        account = (await client.account(refresh_token=False)).account
        if getattr(getattr(account, 'root', account), 'type', None) != 'chatgpt':
            raise RuntimeError('Requires existing ChatGPT subscription login')

        async def generate(qid):
            async with gate:
                if stop['reason']:
                    return
                path = OUT / 'traces' / f'{qid}.json'
                old = read(path)
                attempt = 1
                if old:
                    result = old.get('result')
                    if result and not old.get('recoverable'):
                        save(OUT / 'official' / f'{qid}.json', export(old))
                        return
                    if not args.recover:
                        raise RuntimeError(f'{qid}: interrupted attempt needs --recover and a recorded reason')
                    attempt = old['attempt'] + 1
                    archive = OUT / 'attempts' / f"{qid}-{old['attempt']}.json"
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    path.rename(archive)
                    note(f'决定：恢复 query_id={qid} 的基础设施中断，保留尝试 {attempt-1}；理由：{old.get("result", {}).get("status", "no_terminal_result")}，不因答案分数重跑。')
                trace = {'query_id': qid, 'attempt': attempt, 'question': cases[qid]['query'],
                         'started': now(), 'events': [], 'model_runs': [], 'recoverable': False}
                start = time.monotonic()
                kb = IndexedKnowledgeBase(DATA / 'corpus.sqlite', excerpt_chars=2000)
                def persist():
                    save(path, trace)
                def emit(event):
                    trace['events'].append(event)
                    persist()
                persist()
                try:
                    with tempfile.TemporaryDirectory(prefix='browsecomp-pilot-') as cwd:
                        model = CodexModel(client, cwd, model='gpt-6.1-sol', effort='high', timeout=180, max_calls=30)
                        trace['model_runs'] = model.runs
                        async def complete(messages, specs):
                            if stop['reason']:
                                raise RunStopped(stop['reason'])
                            if request_count() >= 1000:
                                stop['reason'] = 'global_budget_exhausted'
                                raise RunStopped(stop['reason'])
                            with (OUT / 'generation-requests.jsonl').open('a') as ledger:
                                ledger.write(json.dumps({'query_id': qid, 'attempt': attempt,
                                                         'call': len(model.runs)+1, 'started': now()}) + '\n')
                            try:
                                return await model(messages, specs)
                            finally:
                                persist()
                        trace['result'] = await asyncio.wait_for(run_agent(complete, indexed_tools(kb),
                                                        template.format(Question=cases[qid]['query']), max_turns=30, emit=emit), 1800)
                except asyncio.CancelledError:
                    trace.update(result={'status': 'interrupted', 'answer': ''}, recoverable=True)
                    raise
                except Exception as error:
                    message = str(error)
                    lower = message.lower()
                    if isinstance(error, RunStopped):
                        status = message
                    elif any(word in lower for word in ['429', 'rate limit', 'rate_limit', 'usage limit', 'usage_limit']):
                        status = stop['reason'] = 'rate_limited'
                    elif any(word in lower for word in ['permission denied', 'sandbox denied', 'approval required', 'blocked by sandbox']):
                        status = stop['reason'] = 'permission_blocked'
                    elif isinstance(error, TimeoutError):
                        status = 'timeout'
                    else:
                        status = 'runtime_error'
                    trace.update(result={'status': status, 'answer': '', 'error_type': type(error).__name__, 'detail': message},
                                 recoverable=status in ['rate_limited', 'interrupted'])
                    if status == 'permission_blocked':
                        note('决定：立即停止实验；理由：SDK 权限或沙箱阻塞，不修改 CODEX_HOME 或绕过权限。')
                finally:
                    kb.close()
                    trace.update(seconds=time.monotonic()-start, ended=now())
                    persist()
                    save(OUT / 'official' / f'{qid}.json', export(trace))
                    print(json.dumps({'query_id': qid, 'status': trace.get('result', {}).get('status'),
                                      'calls': len(trace['model_runs']), 'total_calls': request_count()}), flush=True)
        try:
            await asyncio.gather(*(generate(qid) for qid in sample(args.limit)))
        finally:
            save(OUT / 'run-state.json', {'status': stop['reason'] or 'stopped', 'ended': now(),
                                         'generation_requests': request_count(), 'concurrency': args.concurrency})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    parser.add_argument('--concurrency', type=int, default=3)
    parser.add_argument('--recover', action='store_true')
    asyncio.run(main(parser.parse_args()))
