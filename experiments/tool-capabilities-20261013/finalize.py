"""Publish the two completed arms once; no model calls or score changes."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from minimal_rag.codex_adapter import token_usage
from minimal_rag_experiments.document_tools import IndexedKnowledgeBase
from minimal_rag_experiments.evidence_summary import publish

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'runs/tool-capabilities-20261013'


def read(path):
    return json.loads(path.read_text())


def main():
    protocol = read(OUTPUT / 'protocol.json')
    ledger = [json.loads(line) for line in (OUTPUT / 'calls.jsonl').read_text().splitlines()]
    groups, details, requests = {}, {}, []
    for arm in protocol['arms']:
        rows = [read(OUTPUT / arm / (qid + '.json')) for qid in protocol['cases']]
        if len(rows) != 24 or not all(r.get('terminal') for r in rows):
            raise RuntimeError('Both complete planned groups are required before publication')
        details[arm] = rows
        gen = [run for r in rows for run in r['model_runs']]
        judge = [run for r in rows for a in r['judge_attempts'] for run in a['runs']]
        requests.extend(gen + judge)
        events = [e for r in rows for e in r['events'] if e.get('role') == 'tool']
        reads = [json.loads(e['arguments_json']) for e in events if e['name'] == 'read']
        metadata = [h['metadata'] for e in events for h in e['content'].get('hits', [e['content']]) if 'metadata' in h]
        docs = {h['document_id'] for e in events for h in e['content'].get('hits', [e['content']]) if h.get('document_id')}
        kb = IndexedKnowledgeBase(ROOT/'data/benchmarks/enterprise/corpus.sqlite', document_tools=True)
        try:
            doc_details = [kb.document_details(doc) for doc in docs]
        finally:
            kb.close()
        groups[arm] = {
            'planned':24, 'completed':24, 'correct':sum(r['metric'] for r in rows),
            'accuracy':sum(r['metric'] for r in rows)/24,
            'result_status':dict(Counter(r['result']['status'] for r in rows)),
            'generation_calls':len(gen), 'judge_calls':len(judge),
            'generation_tokens':token_usage(gen), 'judge_tokens':token_usage(judge),
            'search_calls':sum(e['name']=='search' for e in events), 'read_calls':len(reads),
            'toc_reads':sum(r.get('view')=='toc' for r in reads),
            'section_reads':sum(bool(r.get('section')) for r in reads),
            'cases_using_toc_or_section':sum(any(e.get('role')=='tool' and e.get('name')=='read' and
                (json.loads(e['arguments_json']).get('view')=='toc' or json.loads(e['arguments_json']).get('section')) for e in r['events']) for r in rows),
            'observed_unique_documents':len(docs),
            'observed_documents_with_sections':sum(bool(toc) for _,toc in doc_details),
            'observed_document_metadata_known':{k:sum(m.get(k) is not None for m,_ in doc_details) for k in ['date','version','superseded']},
            'metadata_observations':len(metadata),
            'metadata_known_fields':{k:sum(m.get(k) is not None for m in metadata) for k in ['date','version','superseded']},
        }
    numbers = [r['overnight_call'] for r in requests]
    if len(ledger)>336 or any(r['model']!='gpt-6.1-sol' for r in ledger) or sorted(numbers)!=list(range(1,len(ledger)+1)):
        raise RuntimeError('Request reconciliation failed')
    if any(r['status']!='completed' for r in requests):
        raise RuntimeError('Inspect failed transport records before publication')
    score = [{r['id']:r['metric'] for r in details[a]} for a in protocol['arms']]
    delta = np.array([score[1][i]-score[0][i] for i in protocol['cases']])
    samples = np.random.default_rng(20261013).choice(delta,(20000,24)).mean(axis=1)
    result = {'kind':'known_dev24_tool_capability_comparison','model':'gpt-6.1-sol', 'complete':True,
        'groups':groups, 'paired':{'wins':int(sum(delta>0)),'losses':int(sum(delta<0)),'ties':int(sum(delta==0)),
            'gain':float(delta.mean()),'paired_bootstrap95':np.quantile(samples,[.025,.975]).tolist()},
        'decision':'Keep document tools opt-in; no promotion. No accuracy gain, no section usage, sparse source metadata.',
        'next_recommendation':'Offline corpus-terminology, rare-term and entity-discrimination tool evaluation, then >=24-case matched QA; no additional online arm in this round.',
        'generation_input_increase':groups['document_tools']['generation_tokens']['input_tokens']/groups['baseline']['generation_tokens']['input_tokens']-1,
        'validation':{'core_tests_passed':58,'core_and_related_migration_tests_passed':84,'isolated_wheel_demo':'passed','wheel_document_structure_included':True,'wheel_experiments_excluded':True},
        'new_calls':len(ledger),'new_call_limit':336,'prior_closed_budget_calls':498,
        'all_tokens':token_usage(requests),'requests_reconciled':True,
        'protocol':{k:v for k,v in protocol.items() if k not in ['cases','system_prompt']},
        'system_prompt_unchanged':True, 'core_loop_unchanged':True, 'gpu_used':False,
        'sources':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [
            OUTPUT/'calls.jsonl', OUTPUT/'protocol.json', ROOT/'experiments/overnight-20261012/cohort.json',
            ROOT/'experiments/tool-capabilities-20261013/evaluate.py',ROOT/'experiments/tool-capabilities-20261013/finalize.py',ROOT/'src/minimal_rag/document_structure.py',ROOT/'src/minimal_rag/indexed_rag.py']],
        'limits':['Known development questions, local sol judge, not an untouched benchmark or official score.',
            'One generation per arm; paired bootstrap describes question sampling only, not repeat-generation variance. A zero-width interval for 24 ties does not prove a true population effect of zero.',
            'Metadata availability depends on original source; null is unknown. No inferred source-date/version corrections.',
            'Retains all planned outcomes; no best-of, gold changes or extra experimental arms.'],
        'cases':details}
    (OUTPUT/'final-summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    publish(ROOT/'evidence/tool-capabilities-20261013.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','sources','protocol']},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
