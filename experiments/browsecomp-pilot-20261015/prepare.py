"""Download data, freeze 30 IDs and build the existing SQLite FTS5 index."""
import base64
import hashlib
import json
import random
import tarfile
import time
import urllib.request

from huggingface_hub import HfApi, HfFileSystem, snapshot_download
import pyarrow as pa
import pyarrow.parquet as pq

from minimal_rag.indexed_rag import IndexedKnowledgeBase
from common import DATA, HERE, OFFICIAL, read, save, source_string

REVISION = '046949032b0328319cc9a02663a759ec601d9402'


def main():
    start = time.monotonic()
    DATA.mkdir(parents=True, exist_ok=True)
    if not (OFFICIAL / 'scripts_evaluation/evaluate_run.py').exists():
        archive = DATA / 'official.tar.gz'
        urllib.request.urlretrieve(f'https://codeload.github.com/texttron/BrowseComp-Plus/tar.gz/{REVISION}', archive)
        unpack = DATA / 'unpacked'
        with tarfile.open(archive) as tar:
            tar.extractall(unpack, filter='data')
        next(unpack.iterdir()).rename(OFFICIAL)
        unpack.rmdir()
    meta = read(DATA / 'download-metadata.json', {})
    for repo in ['Tevatron/browsecomp-plus', 'Tevatron/browsecomp-plus-corpus']:
        if repo not in meta:
            info = HfApi().dataset_info(repo, files_metadata=True)
            meta[repo] = {'revision': info.sha, 'files': [(f.rfilename, f.size) for f in info.siblings]}
    save(DATA / 'download-metadata.json', meta)
    topic_repo = 'Tevatron/browsecomp-plus'
    if not (DATA / 'queries.jsonl').exists():
        fs, rows = HfFileSystem(), []
        for filename, _ in meta[topic_repo]['files']:
            if not filename.endswith('.parquet'):
                continue
            with fs.open(f"datasets/{topic_repo}@{meta[topic_repo]['revision']}/{filename}", 'rb') as stream:
                rows.extend(pq.ParquetFile(stream).read(columns=['query_id', 'query', 'answer']).to_pylist())
            print(json.dumps({'stage': 'questions_download', 'rows': len(rows)}), flush=True)
        pq.write_table(pa.Table.from_pylist(rows), DATA / 'queries-encrypted.parquet')
        password = source_string(OFFICIAL / 'scripts_build_index/decrypt_dataset.py', 'DEFAULT_CANARY')
        key = hashlib.sha256(password.encode()).digest()
        def decrypt(value):
            ciphertext = base64.b64decode(value)
            return bytes(v ^ key[i % len(key)] for i, v in enumerate(ciphertext)).decode()
        with (DATA / 'queries.jsonl.tmp').open('w') as output:
            for row in rows:
                output.write(json.dumps({'query_id': str(row['query_id']), 'query': decrypt(row['query']),
                                         'answer': decrypt(row['answer'])}, ensure_ascii=False) + '\n')
        (DATA / 'queries.jsonl.tmp').replace(DATA / 'queries.jsonl')
    ids = sorted(str(json.loads(line)['query_id']) for line in (DATA / 'queries.jsonl').read_text().splitlines())
    if len(ids) != 830 or len(set(ids)) != 830:
        raise ValueError('Expected 830 unique question IDs')
    chosen = {'seed': 20261015, 'population': 830, 'sort': 'lexicographic string query_id',
              'query_ids': random.Random(20261015).sample(ids, 30), 'role': 'development sample'}
    if read(HERE / 'sample.json', chosen) != chosen:
        raise ValueError('Existing sample differs')
    save(HERE / 'sample.json', chosen)
    repo = 'Tevatron/browsecomp-plus-corpus'
    corpus = DATA / 'corpus-parquet'
    snapshot_download(repo, repo_type='dataset', revision=meta[repo]['revision'],
                      allow_patterns=['data/*.parquet'], local_dir=corpus, max_workers=3)
    files = sorted(corpus.glob('data/*.parquet'))
    print(json.dumps({'stage': 'corpus_schema', 'files': len(files), 'fields': pq.ParquetFile(files[0]).schema_arrow.names}), flush=True)
    if not (DATA / 'corpus.sqlite').exists():
        build_start = time.monotonic()
        def documents():
            count = 0
            for path in files:
                for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=['docid', 'text', 'url']):
                    for row in batch.to_pylist():
                        text, url = row['text'] or '', row['url'] or ''
                        yield {'id': str(row['docid']), 'title': url or text.partition('\n')[0][:120], 'text': text, 'source': url}
                        count += 1
                        if count % 5000 == 0:
                            print(json.dumps({'stage': 'index', 'documents': count}), flush=True)
        stats = IndexedKnowledgeBase.build(DATA / 'corpus.sqlite', documents())
        stats.update(seconds=time.monotonic() - build_start, official_revision=REVISION, data_revisions={k: v['revision'] for k,v in meta.items()})
        save(DATA / 'index.json', stats)
    print(json.dumps({'stage': 'prepared', 'index': read(DATA / 'index.json'), 'total_seconds': time.monotonic()-start}), flush=True)


if __name__ == '__main__':
    main()
