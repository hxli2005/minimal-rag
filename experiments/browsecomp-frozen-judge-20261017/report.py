"""Verify returned first official grades and compare with the historical submission."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'runs/browsecomp-frozen-judge-20261017'


def read(path):
    return json.loads(path.read_text())


def main():
    protocol=read(HERE/'protocol.json')
    state=read(OUT/'runtime/official-state.json')
    assert state['status']=='completed'
    engine=read(HERE/'engine-adjustment.json')
    metadata=read(OUT/'runtime/runtime-metadata.json')
    assert metadata['source_sha256']['scripts_evaluation/evaluate_run.py']==engine['original_sha256']
    assert metadata['source_sha256']['scripts_evaluation/evaluate_run_single_gpu.py']==engine['patched_sha256']
    assert metadata['original_bundle_sha256']==protocol['bundle_sha256']
    summary=read(OUT/'evals/frozen-20261017/evaluation_summary.json')
    old=read(ROOT/'experiments/browsecomp-final-20261015/submission.json')
    new_flags={str(r['query_id']):r['correct'] for r in summary['per_query_metrics']}
    old_flags={str(r['query_id']):r['correct'] for r in old['per_query_metrics']}
    assert len(new_flags)==830 and new_flags.keys()==old_flags.keys()
    for qid in new_flags:
        source=OUT/'bundle/runs/frozen-20261017'/f'{qid}.json'
        assert hashlib.sha256(source.read_bytes()).hexdigest()==protocol['answer_sha256'][qid]
        answer=read(source)
        graded=read(OUT/'evals/frozen-20261017'/f'{qid}_eval.json')
        assert graded['is_completed'] and not graded['judge_result']['parse_error']
        assert graded['judge_result']['correct']==new_flags[qid]
        assert graded['response']==answer['result'][0]['output'], qid
        assert set(graded['retrieval']['retrieved_docids'])==set(answer['retrieved_docids']), qid
        # Official evaluation records retain the input response and retrieval IDs.
        for key in ('query_id','tool_call_counts','retrieved_docids','result','status'):
            if key in graded:
                assert graded[key]==answer[key], (qid,key)
    wins=sorted(q for q in new_flags if new_flags[q] and not old_flags[q])
    losses=sorted(q for q in new_flags if old_flags[q] and not new_flags[q])
    result=dict(questions=830,correct=sum(new_flags.values()),accuracy_percent=sum(new_flags.values())/830*100,
                historical_correct=sum(old_flags.values()),wins=wins,losses=losses,
                ties=830-len(wins)-len(losses),new_generation_calls=0,
                valid_judge_responses=830,backfill_judge_responses=0,
                accuracy_delta_pp=(sum(new_flags.values())-sum(old_flags.values()))/830*100,
                engine_adjustment=engine['only_change'],
                hardware=metadata['gpu'],
                evaluation_date_client=datetime.fromtimestamp(state['ended'],ZoneInfo('America/Los_Angeles')).date().isoformat(),
                execution_attempts=state['attempts'],
                official_summary={k:v for k,v in summary.items() if k!='per_query_metrics'},
                protocol_sha256=hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest(),
                limitations=protocol['limitations']+['Official scoring logic with disclosed one-line engine context adjustment; not yet maintainer verified.'])
    (HERE/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    submission=dict(summary,LLM='gpt-6.1-sol (effort high)',
        Retriever='SQLite FTS5 BM25, search/read; frozen-20261017 search description',Link='',
        **{'Evaluation Date':result['evaluation_date_client']})
    (HERE/'submission.json').write_text(json.dumps(submission,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
