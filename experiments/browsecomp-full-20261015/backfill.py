"""Complete deferred judges; only model responses consume the three-response cap."""
import argparse
import asyncio
import os
from run import grade, needs_judgement, judge_response_count
from shared import OUT, directory, note, now, queries, read, save


def pending(groups):
    result = []
    for group in groups:
        for path in sorted((directory(group) / 'official').glob('*.json')):
            answer = read(path)
            report = read(directory(group) / 'judgements' / path.name, {})
            if answer['status'] == 'completed' and needs_judgement(report):
                result.append((group, path.stem))
    return result


async def main(groups):
    if not os.environ.get('OPENROUTER_API_KEY'):
        raise RuntimeError('OPENROUTER_API_KEY missing')
    cases, stop, round_number = queries(), {'reason': None}, 0
    state_path = OUT / 'backfill-state.json'
    while todo := pending(groups):
        round_number += 1
        attempts, responses, completed = 0, 0, 0
        save(state_path, {'status':'running','pid':os.getpid(),'groups':groups,
                         'round':round_number,'pending':len(todo),'updated':now()})
        for group, qid in todo:
            path = directory(group) / 'judgements' / f'{qid}.json'
            before = read(path, {})
            await grade(group, qid, cases, stop, backfill=True)
            after = read(path, {})
            attempts += len(after.get('attempts', [])) - len(before.get('attempts', []))
            responses += judge_response_count(after) - judge_response_count(before)
            completed += after.get('judgement', {}).get('correct') is not None
            if stop['reason']:
                save(state_path, {'status':stop['reason'],'updated':now(),'groups':groups})
                raise RuntimeError(stop['reason'])
        remaining = len(pending(groups))
        note(f'统一补判第{round_number}轮（{",".join(groups)}）：追加尝试{attempts}次，收到模型响应{responses}次，新增有效判决{completed}题，仍待补判{remaining}题；首次有效判决不重跑。')
        if remaining:
            await asyncio.sleep(60)
    save(state_path, {'status':'completed','groups':groups,'rounds':round_number,'updated':now()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('groups', nargs='+', choices=['A','A-lean','B'])
    asyncio.run(main(parser.parse_args().groups))
