"""Publish bounded fixed-evidence results; never makes model requests."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from minimal_rag.codex_adapter import token_usage
from minimal_rag_experiments.evidence_summary import publish
from evaluate import OUT, ROOT, HERE, ARMS, read, save, summarize


def main():
    result = summarize()
    if not result['complete']:
        raise RuntimeError('All planned outcomes must be scored or retained generation failures')
    protocol = read(OUT / 'protocol.json')
    ledger = [json.loads(l) for l in (OUT / 'calls.jsonl').read_text().splitlines()]
    all_runs = []
    for q in protocol['main_ids'] + protocol['diagnostic_ids']:
        for arm in ARMS:
            row = read(OUT / arm / f'{q}.json')
            assert len(row['model_runs']) == 1, (q, arm)
            assert row['terminal'], (q, arm)
            all_runs += row['model_runs']
            all_runs += [r for a in row['judge_attempts'] for r in a['runs']]
            if 'judgement' in row:
                first = next(a['assessment'] for a in row['judge_attempts'] if 'assessment' in a)
                assert row['judgement'] == first
    assert sorted(r['experiment_call'] for r in all_runs) == list(range(1, len(ledger) + 1))
    assert len(ledger) <= protocol['request_limit']
    assert all(c['model'] == 'gpt-6.1-sol' for c in ledger)
    usage = token_usage(all_runs)
    usage['uncached_input_tokens'] = usage['input_tokens'] - usage['cached_input_tokens']
    result.update(total_new_usage=usage, calls_by_role=dict(Counter(c['role'] for c in ledger)),
                  missing_usage=sum(r.get('usage') is None for r in all_runs),
                  protocol_path=str((OUT / 'protocol.json').relative_to(ROOT)),
                  raw_path=str(OUT.relative_to(ROOT)), new_retrieval_calls=0,
                  no_generation_retries=True, first_valid_judgements_preserved=True)
    c = result['cohorts']['main24']['comparison']
    signal = ((c['correct_delta'] >= 0 and c['completeness_delta'] >= .03)
              or (c['wins'] - c['losses'] >= 2 and c['completeness_delta'] >= 0))
    result['directional_screen'] = dict(quality_signal=signal,
        cost_acceptable=c['cost_complete'] and c['uncached_input_tokens_ratio'] <= 1.10,
        promising=signal and c['cost_complete'] and c['uncached_input_tokens_ratio'] <= 1.10,
        note='A small-sample screening signal, not a formal promotion gate or automatic full-run trigger.')
    result['decision'] = {
        'current_system': 'v3', 'candidate_adopted': False, 'full_evaluation_started': False,
        'recommendation': 'Do not expand this unchanged answer prompt to full evaluation yet.',
        'reason': 'All 24 main accuracy outcomes tied; modest coverage gains were concentrated in a few questions, while important visible facts remained omitted and output grew. One of two deliberately selected probes improved.',
        'threshold_note': 'The decision is not based solely on the 2.61pp gain falling just below the prespecified 3pp screening signal or on a confidence interval crossing zero.'}
    result['limitations'] = [
        'Known, category-balanced dev24; not prevalence-weighted and not unseen evaluation.',
        'Two deliberately selected omission probes are reported separately from main24.',
        'Observed v3 text is deduplicated and supplied in a fresh answer-only context without retrieval history or tools.',
        'Costs measure new final-answer generation only; they exclude sunk v3 retrieval and are not full-RAG costs.',
        'Paired bootstrap resamples questions; it does not capture repeat-generation/judge variation.',
        'Frozen local sol judgements retained, including known qst_0240/qst_0422 reference/judge disputes.'
    ]
    # Provenance fingerprints are only added for the completed, published result.
    paths = [HERE / 'evaluate.py', HERE / 'finalize.py', HERE / 'frozen.json', OUT / 'calls.jsonl',
             ROOT / 'src/minimal_rag/codex_adapter.py', ROOT / 'experiments/src/minimal_rag_experiments/rag_judge.py']
    result['sources'] = [{'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    save(OUT / 'final-summary.json', result)
    publish(ROOT / 'evidence/fixed-evidence-v3-20261015.json', result)
    print(json.dumps({k: result[k] for k in ['new_calls', 'calls_by_role', 'missing_usage', 'directional_screen', 'total_new_usage']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
