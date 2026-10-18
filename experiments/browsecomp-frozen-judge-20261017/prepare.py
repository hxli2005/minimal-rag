"""Prepare the existing frozen answers for their first official evaluation; no API calls."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'runs/semantic-tool-20261016/browsecomp-A'
DATA = ROOT / 'data/benchmarks/browsecomp-plus'
OUT = ROOT / 'runs/browsecomp-frozen-judge-20261017'
BUNDLE = OUT / 'bundle'


def main():
    assert not (OUT / 'bundle.tar.gz').exists(), 'Prepared bundle already exists; do not overwrite.'
    files = 0
    with tarfile.open(DATA / 'official.tar.gz') as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            rel = Path(member.name).relative_to(member.name.split('/')[0])
            assert (DATA / 'official' / rel).read_bytes() == archive.extractfile(member).read(), rel
            files += 1
    shutil.copytree(DATA / 'official', BUNDLE, ignore=shutil.ignore_patterns('.git', '__pycache__'))
    target = BUNDLE / 'runs/frozen-20261017'
    target.mkdir(parents=True)
    cfg = json.loads((SOURCE / 'config.json').read_text())
    ids = cfg['ids']
    assert len(ids) == len(set(ids)) == 830
    assert {p.stem for p in (SOURCE / 'official').glob('*.json')} == set(ids)
    hashes = {}
    for qid in ids:
        trace = json.loads((SOURCE / 'traces' / f'{qid}.json').read_text())
        record = json.loads((SOURCE / 'official' / f'{qid}.json').read_text())
        counts, docids = Counter(), set()
        for event in trace['events']:
            if event.get('role') != 'tool':
                continue
            counts[event['name']] += 1
            if event['name'] == 'search':
                docids.update(str(hit['document_id']) for hit in event['content'].get('hits', []))
        assert set(counts) <= {'search', 'read'}
        assert trace['result']['status'] == 'finished'
        assert record == dict(query_id=qid, tool_call_counts=dict(counts),
                              retrieved_docids=sorted(docids), status='completed',
                              result=[dict(type='output_text', output=trace['result']['answer'])]), qid
        path = SOURCE / 'official' / f'{qid}.json'
        shutil.copy2(path, target / path.name)
        hashes[qid] = hashlib.sha256(path.read_bytes()).hexdigest()
    (BUNDLE / 'data').mkdir(exist_ok=True)
    shutil.copy2(DATA / 'queries.jsonl', BUNDLE / 'data/browsecomp_plus_decrypted.jsonl')
    with tarfile.open(OUT / 'bundle.tar.gz', 'w:gz') as archive:
        archive.add(BUNDLE, arcname='bundle')
    protocol = dict(
        status='prepared_not_started', source=str(SOURCE.relative_to(ROOT)), questions=830,
        generated_answers=0, official_revision='046949032b0328319cc9a02663a759ec601d9402',
        model='Qwen/Qwen3-32B', model_revision='9216db5781bf21249d130ec9da846c4624c16137',
        dtype='bfloat16', preferred_gpu='single A800 80GB', tensor_parallel_size=1,
        sampling=dict(temperature=0.7, top_p=0.8, top_k=20, max_output_tokens=4096, batch_size=64),
        budget_cny=dict(target=20, stop=30, enclosing_total=200),
        policy='Retain first valid grades. Retry only missing/invalid technical outputs, at most 3 responses per question. Report both historical and new results regardless of direction. No answer regeneration.',
        historical_reference=dict(correct=772, total=830, tensor_parallel_size=2, gpu='2x H800'),
        limitations=['Known regression set; not an untouched holdout.',
                     'Eight original generation failures were technically retried; preserve original accounting.',
                     'Different GPU/TP may introduce numerical and sampling differences; not a controlled prompt-only ablation.'],
        official_files_verified=files, source_config_sha256=hashlib.sha256((SOURCE/'config.json').read_bytes()).hexdigest(),
        answer_sha256=hashes, bundle_sha256=hashlib.sha256((OUT/'bundle.tar.gz').read_bytes()).hexdigest())
    (HERE / 'protocol.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in protocol.items() if k != 'answer_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
