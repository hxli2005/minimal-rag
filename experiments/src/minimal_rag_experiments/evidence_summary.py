"""Keep summaries in Git and complete result payloads under ignored runs/."""
import hashlib
import json
from pathlib import Path

from minimal_rag.paths import ROOT
DETAIL_KEYS = set('cases rows chosen_rows document_versions rankings ranked candidate_answers events model_runs question answer gold_answer reference_facts selected_observation searches judgement paired_once_judgement'.split())


def compact(value):
    if isinstance(value, dict) and value.get('kind') == 'consolidated_meta_validation_records':
        return {**{k:value[k] for k in ['schema_version','kind','date','note'] if k in value},
                'record_count':len(value['records']),
                'records':{k:{x:v[x] for x in ['original_path','original_sha256','archive_path'] if x in v} for k,v in value['records'].items()}}
    if isinstance(value, dict) and 'calls_reserved' in value and 'additional_experiments' in value:
        return {**{k:value[k] for k in ['model','calls_reserved','call_limit','calls_by_role','reported_beijing','limitations','report','repository_result'] if k in value},
                'groups':{k:{x:v[x] for x in ['planned','terminal','scored','planned_denominator_score','generation_calls','generation_tokens'] if x in v} for k,v in value['groups'].items()}}
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items() if k not in DETAIL_KEYS and not k.startswith('qst_')}
    if isinstance(value, list):
        is_detail = any(isinstance(v, str) and v.startswith('qst_') for v in value)
        is_detail |= any(isinstance(v, dict) and set(v) & {'question_id', 'id', 'ranked', 'document_id', 'offset', 'call'} for v in value)
        return {'count': len(value), 'detail': 'See raw_detail'} if is_detail else [compact(v) for v in value]
    return value


def publish(path, value):
    path = Path(path)
    raw = ROOT / 'runs/evidence' / path.name
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    summary = compact(value)
    summary['raw_detail'] = {'path': str(raw.relative_to(ROOT)), 'sha256': hashlib.sha256(raw.read_bytes()).hexdigest()}
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')


def clean_existing():
    # Original bytes were preserved before pruning; never replace them with a summary.
    originals = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'runs/evidence').glob('*.json')}
    def repair(value):
        if isinstance(value, dict):
            p = value.get('path', '')
            if p.startswith('evidence/') and value.get('sha256') == originals.get(Path(p).name):
                value = {**value, 'path': 'runs/evidence/' + Path(p).name}
            return {k: repair(v) for k, v in value.items()}
        if isinstance(value, list):
            return [repair(v) for v in value]
        return value
    for path in (ROOT / 'evidence').glob('*.json'):
        original = json.loads((ROOT / 'runs/evidence' / path.name).read_text())
        summary = repair(compact(original))
        summary['raw_detail'] = {'path': 'runs/evidence/' + path.name, 'sha256': originals[path.name]}
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    clean_existing()
