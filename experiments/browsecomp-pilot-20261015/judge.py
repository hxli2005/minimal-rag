"""Official scoring prompt via fixed OpenRouter provider, or labelled sol fallback."""
import argparse
import asyncio
import json
import os
import tempfile
import httpx

from openai_codex import AsyncCodex
from minimal_rag.codex_adapter import CodexModel, rag_codex_config
from minimal_rag.loop import run_agent
from common import LABEL, OFFICIAL, OUT, note, now, parse_judge_response, queries, read, sample, save, source_string

NON_THINKING_PREFIX = '<think>\n\n</think>\n\n'
PARAMS = {'model': 'qwen/qwen3-32b', 'temperature': .7, 'top_p': .8, 'top_k': 20, 'max_tokens': 4096,
          'reasoning': {'enabled': False, 'exclude': False},
          'continue_final_message': True,
          'provider': {'order': ['DeepInfra'], 'allow_fallbacks': False, 'data_collection': 'deny'}}


def router(prompt):
    # Qwen's enable_thinking=False template emits this exact assistant prefix.
    payload = dict(PARAMS, messages=[{'role': 'user', 'content': prompt},
                                    {'role': 'assistant', 'content': NON_THINKING_PREFIX}])
    response = httpx.post('https://openrouter.ai/api/v1/chat/completions', json=payload, timeout=httpx.Timeout(180, connect=20),
                          headers={'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY']})
    response.raise_for_status()
    return response.json()


async def main(args):
    mode = 'openrouter' if os.environ.get('OPENROUTER_API_KEY') else 'sol'
    config = {'mode': mode, 'label': LABEL if mode == 'sol' else 'Qwen3-32B（OpenRouter/DeepInfra fp8）裁判，与官方同一模型，量化不同',
              'parameters': PARAMS if mode == 'openrouter' else {'model': 'gpt-6.1-sol', 'effort': 'medium', 'max_turns': 1},
              'prompt_source': 'scripts_evaluation/evaluate_run.py:GRADER_TEMPLATE',
              'assistant_prefill': NON_THINKING_PREFIX if mode == 'openrouter' else None}
    if read(OUT / 'judge-config.json', config) != config:
        raise ValueError('Judge configuration differs from existing run')
    save(OUT / 'judge-config.json', config)
    template = source_string(OFFICIAL / 'scripts_evaluation/evaluate_run.py', 'GRADER_TEMPLATE')
    cases = queries()
    async def grade(client=None):
        for qid in sample(args.limit):
            answer = read(OUT / 'official' / f'{qid}.json')
            if not answer or answer['status'] != 'completed':
                continue
            path = OUT / 'judgements' / f'{qid}.json'
            report = read(path, {'query_id': qid, 'mode': mode, 'attempts': []})
            if report.get('judgement', {}).get('correct') is not None:
                continue
            prompt = template.format(question=cases[qid]['query'], response=answer['result'][-1]['output'], correct_answer=cases[qid]['answer'])
            report['prompt'] = prompt
            def current_attempts():
                return sum(a.get('parameters') == config['parameters'] for a in report['attempts'])
            while current_attempts() < 3:
                attempt = {'started': now(), 'parameters': config['parameters'], 'assistant_prefill': config['assistant_prefill']}
                report['attempts'].append(attempt)
                save(path, report)
                try:
                    if mode == 'openrouter':
                        raw = await asyncio.to_thread(router, prompt)
                        attempt['response'] = raw
                        message = raw['choices'][0]['message']
                        text = message.get('content') or ''
                        reasoning = bool(message.get('reasoning') or message.get('reasoning_content') or message.get('reasoning_details'))
                        reasoning_tokens = raw.get('usage', {}).get('completion_tokens_details', {}).get('reasoning_tokens')
                        attempt['reasoning_present'] = reasoning
                        attempt['reasoning_tokens'] = reasoning_tokens
                        if raw.get('provider') != 'DeepInfra':
                            raise RuntimeError('Unexpected provider; stop without fallback')
                        if reasoning or (reasoning_tokens is not None and reasoning_tokens > 0):
                            raise RuntimeError('Thinking-disable smoke failed; stop without prompt changes')
                    else:
                        with tempfile.TemporaryDirectory(prefix='browsecomp-judge-') as cwd:
                            model = CodexModel(client, cwd, model='gpt-6.1-sol', effort='medium', timeout=180, max_calls=1)
                            attempt['model_runs'] = model.runs
                            result = await run_agent(model, {}, prompt, max_turns=1)
                            text = result.get('answer', '') if result['status'] == 'finished' else ''
                    attempt['text'] = text
                    parsed = parse_judge_response(text)
                    attempt['parsed'] = parsed
                    if parsed['correct'] is None:
                        raise ValueError('Judge response has no parseable correct field')
                    report['judgement'] = parsed
                    report['first_successful_attempt'] = len(report['attempts'])
                    attempt['status'] = 'completed'
                    save(path, report)
                    print(json.dumps({'query_id': qid, 'judge': mode, 'correct': parsed['correct'], 'attempt': len(report['attempts'])}), flush=True)
                    break
                except Exception as error:
                    # No request headers or secret-bearing objects enter logs.
                    attempt.update(status='failed', error_type=type(error).__name__, error=str(error))
                    if isinstance(error, httpx.HTTPStatusError):
                        attempt['http_status'] = error.response.status_code
                        attempt['http_body'] = error.response.text
                    save(path, report)
                    note(f'query_id={qid} 裁判尝试 {len(report["attempts"])} 失败：{type(error).__name__}；首次有效判决规则不变。')
                    if isinstance(error, RuntimeError) or (qid == sample()[0] and isinstance(error, ValueError)):
                        raise
                    if current_attempts() < 3:
                        await asyncio.sleep(2)
            if report.get('judgement', {}).get('correct') is None:
                print(json.dumps({'query_id': qid, 'judge': mode, 'status': 'unjudged'}), flush=True)
    if mode == 'sol':
        async with AsyncCodex(rag_codex_config('/private/tmp')) as client:
            await grade(client)
    else:
        await grade()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    asyncio.run(main(parser.parse_args()))
