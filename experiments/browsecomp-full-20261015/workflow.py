"""Resume ordered phases, enforce gates, and commit only sanitized deliverables."""
import asyncio
import json
import os
import signal
import sys
from shared import DATA, HERE, OUT, LIMITS, commit, count, directory, ids, note, now, read, save
from report import aggregate, paired, rows, selection, snapshot, submission
from standard_tools import snippet, tokenizer
from minimal_rag.indexed_rag import IndexedKnowledgeBase


async def phase(group, stage, sample, limit=None):
    routing_retries = 0
    while True:
        command = [sys.executable, '-u', str(HERE/'run.py'), group, stage, sample]
        if limit is not None:
            command += ['--limit', str(limit)]
        process = await asyncio.create_subprocess_exec(*command,
            pass_fds=(int(os.environ['BROWSECOMP_RUN_LOCK_FD']),))
        save(OUT/'active-phase-process.json', {'pid':process.pid,'group':group,'stage':stage,'started':now()})
        try:
            code = await process.wait()
        except asyncio.CancelledError:
            process.terminate()
            await process.wait()
            raise
        status = read(OUT/f'{stage}-{group}-state.json')['status']
        if code == 0:
            pending = [qid for qid in ids(sample)[:limit]
                       if not (t := read(directory(group)/'traces'/f'{qid}.json'))
                       or not t.get('result') or t.get('recoverable')]
            if pending:
                raise RuntimeError(f'{stage}/{group}: unfinished generation IDs {pending}')
        if status == 'routing_timeout' and routing_retries < 2:
            routing_retries += 1
            note(f'决定：{stage}/{group}工作区路由超时后用新进程重试{routing_retries}/2；理由：此前同进程阶段切换复现该错误，模型与权限配置不变。')
            await asyncio.sleep(60)
            continue
        if status == 'rate_limited':
            note(f'决定：{stage}/{group}限流后等待60秒并按已保存并发减半续跑；理由：保留全部失败尝试，不改模型与工具。')
            await asyncio.sleep(60)
            continue
        if status not in ('completed','stage_budget_exhausted') or (code != 0 and status == 'completed'):
            raise RuntimeError(f'{stage}/{group}: {status}; stop all stages')
        return status


async def backfill(groups):
    process = await asyncio.create_subprocess_exec(sys.executable, '-u', str(HERE/'backfill.py'), *groups,
        pass_fds=(int(os.environ['BROWSECOMP_RUN_LOCK_FD']),))
    save(OUT/'active-phase-process.json', {'pid':process.pid,'stage':'judge_backfill','groups':groups,'started':now()})
    try:
        code = await process.wait()
    except asyncio.CancelledError:
        process.terminate()
        await process.wait()
        raise
    if code:
        raise RuntimeError(f'Judge backfill stopped, exit={code}; see backfill-state.json')


def smoke_check():
    tok = tokenizer()
    kb = IndexedKnowledgeBase(DATA/'corpus.sqlite')
    lengths, calls, graded = [], 0, 0
    try:
        for qid in ids('pilot30')[:2]:
            trace = read(directory('B')/'traces'/f'{qid}.json')
            official = read(directory('B')/'official'/f'{qid}.json')
            judge = read(directory('B')/'judgements'/f'{qid}.json',{})
            assert official['status'] == 'completed'
            assert set(official) == {'query_id','tool_call_counts','status','retrieved_docids','result'}
            assert judge.get('judgement',{}).get('correct') is not None
            accepted = next(a for a in judge['attempts'] if a.get('status') == 'completed')
            assert accepted['response']['provider'] == 'DeepInfra' and accepted['reasoning_tokens'] == 0
            assert not accepted['reasoning_present']
            graded += 1
            for event in trace['events']:
                if event['role'] != 'tool':
                    continue
                assert event['name'] == 'search'
                hits = json.loads(event['content'])
                assert len(hits) <= 5 and len({h['docid'] for h in hits}) == len(hits)
                args = json.loads(event['arguments_json'])
                ranked = kb.search(args['query'],limit=5)['hits']
                assert [h['docid'] for h in hits] == [h['document_id'] for h in ranked]
                for hit in hits:
                    assert set(hit) == {'docid','snippet'}
                    text = kb.db.execute('SELECT text FROM documents WHERE id=?',(hit['docid'],)).fetchone()[0]
                    assert hit['snippet'] == snippet(text,tok)
                    lengths.append(len(tok.encode(hit['snippet'],add_special_tokens=False)))
                calls += 1
    finally:
        kb.close()
    assert calls > 0 and max(lengths) <= 512
    value = {'passed':True,'cases':2,'graded':graded,'search_calls':calls,
             'snippet_max_observed_tokens':max(lengths),'all_snippets_equal_document_prefix':True,
             'ranking_matches_project_fts5':True,'only_search':True,'official_export_valid':True,
             'judge_provider':'DeepInfra','all_reasoning_tokens_zero':True}
    save(HERE/'smoke-validation.json',value)
    return value


async def main():
    root_task = asyncio.current_task()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, root_task.cancel)
    state_path = OUT/'workflow-state.json'
    state = read(state_path,{'completed_phases':[]})
    state.pop('error_type', None)
    state.pop('reason', None)
    state.update(status='running',pid=os.getpid(),updated=now())
    save(state_path,state)
    def checkpoint(name, message):
        snapshot(publish=True)
        commit(message)
        state['completed_phases'].append(name)
        state.update(updated=now(),current_phase=None)
        save(state_path,state)
    def begin(name):
        state.update(current_phase=name,updated=now())
        save(state_path,state)
    try:
        if 'b30' not in state['completed_phases']:
            begin('b30')
            if not read(HERE/'smoke-validation.json',{}).get('passed'):
                await phase('B','b30','pilot30',limit=2)
                smoke_check()
                note('两题B组冒烟通过：文档前512token、FTS5排序、仅search、官方导出及DeepInfra零推理判分均验证；继续固定30题，不按分数改装配。')
            await phase('B','b30','pilot30')
            a,b = rows('A',ids('pilot30')),rows('B',ids('pilot30'))
            save(HERE/'b30-summary.json',{'A':aggregate(a),'B':aggregate(b),'A_minus_B':paired(a,b)})
            note('B30阶段完成或触达900次上限；决定提交实际摘要后进入预先指定的降本验证，未经对照完成不夸大项目贡献。')
            checkpoint('b30','Complete BrowseComp standard-tools 30-question comparison')
        if 'dev' not in state['completed_phases']:
            begin('dev')
            await phase('A','dev','new70')
            await phase('A-lean','dev','dev100')
            begin('dev_judge_backfill')
            await backfill(['A','A-lean'])
            begin('dev')
            result = selection()
            note(f'决定：最终冻结配置={result["selected"]}，降本门槛通过={result["passed"]}；理由：完整配对/用量={result["complete_pairs_and_usage"]}，未缓存下降={result["uncached_reduction_fraction"]}，损失减胜利={result["paired_lean_minus_A"]["losses"]-result["paired_lean_minus_A"]["wins"]}。')
            checkpoint('dev','Freeze BrowseComp configuration after paired cost gate')
        chosen = read(HERE/'selection.json')['selected']
        if 'full' not in state['completed_phases']:
            begin('full')
            projection_path = HERE/'full-projection.json'
            if not projection_path.exists():
                await phase(chosen,'full','remaining730',limit=100)
                metrics = aggregate(rows(chosen,ids('remaining730')[:100]))
                concurrency = read(OUT/'runtime.json', {}).get('generation_concurrency',5)
                estimates = {'first100':metrics,'projected_requests_for_730':metrics['mean_generation_requests']*730,
                             'projected_remaining_requests':metrics['mean_generation_requests']*630,
                             'projected_remaining_uncached_tokens':metrics['mean_uncached_input_tokens']*630,
                             'projection_concurrency':concurrency,
                             'projected_remaining_wall_seconds':metrics['mean_seconds']*630/concurrency,
                             'stage_limit':LIMITS['full']}
                estimates['stop_for_projection'] = estimates['projected_requests_for_730'] > LIMITS['full']
                save(projection_path,estimates)
                note(f'阶段3前100题：平均请求={metrics["mean_generation_requests"]:.2f}，未缓存token={metrics["mean_uncached_input_tokens"]:.0f}，每题耗时={metrics["mean_seconds"]:.1f}s；预计730题请求={estimates["projected_requests_for_730"]:.0f}，因预算预测停止={estimates["stop_for_projection"]}。')
            if not read(projection_path)['stop_for_projection']:
                await phase(chosen,'full','remaining730')
            else:
                note('决定：阶段3按前100题预测停止，不追加全量生成；理由：预计超过7500次上限，按brief保留当前完成结果，不为省预算改配置。')
            checkpoint('full','Record BrowseComp frozen full-run outcome and budget projection')
        if 'b200' not in state['completed_phases']:
            begin('b200')
            await phase('B','b200','b200')
            checkpoint('b200','Complete BrowseComp 200-question held-out standard-tools comparison')
        begin('final_judge_backfill')
        await backfill(['A','A-lean','B'])
        a,b = rows('A',ids('pilot30')),rows('B',ids('pilot30'))
        save(HERE/'b30-summary.json',{'A':aggregate(a),'B':aggregate(b),'A_minus_B':paired(a,b)})
        begin('report')
        ready = submission()
        note(f'提交文件就绪={ready}；仅在全量终态与评分/校准齐备时生成正式submission，部分运行保留摘要，不伪装830题已完成。')
        note('Scaffold说明：每题独立Codex SDK临时会话，gpt-6.1-sol/high，最多30次应用层请求/30轮；极简模型→工具→观察循环，FTS5 BM25在1600字符块上排序并按文档去重。冻结search/read设置和摘录长度见selection.json；read按字符分页。B用同一排序，但只给前5篇文档各前512个Qwen3-0.6B token及官方无全文读取提示。不是官方Lucene基线；同循环对照识别工具与提示组合贡献。裁判Qwen3-32B经DeepInfra fp8、空think前缀、首次有效判决；dev100纳入全量并单报其余730题。SDK内部重试不可见；不发送邮件或上传。')
        checkpoint('report','Report BrowseComp contribution cost and frozen-run results')
        state.update(status='completed' if ready else 'completed_with_budget_or_evaluation_limitations',updated=now(),submission_ready=ready)
        save(state_path,state)
        print(json.dumps({'status':state['status'],'selected':chosen,'submission_ready':ready,
                          'stage_requests':{k:count(k) for k in LIMITS}},ensure_ascii=False),flush=True)
    except BaseException as error:
        state.update(status='halted',updated=now(),error_type=type(error).__name__,reason=str(error))
        save(state_path,state)
        note(f'全流程停止于{state.get("current_phase")}：{type(error).__name__}；理由见data下workflow-state.json。保留断点，不绕过权限/额度或修改CODEX_HOME。')
        snapshot(publish=True)
        raise


if __name__ == '__main__':
    asyncio.run(main())
