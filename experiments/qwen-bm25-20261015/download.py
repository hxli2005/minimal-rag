"""Download the frozen official Qwen files from ModelScope; no GPU required."""
import concurrent.futures
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


def fetch(entry):
    target = Path('models') / entry['role'] / entry['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == entry['bytes']:
        digest = hashlib.file_digest(target.open('rb'), 'sha256').hexdigest()
        if digest == entry['sha256']:
            return {'file': str(target), 'cached': True, 'bytes': entry['bytes']}
    partial = target.with_suffix(target.suffix+'.partial')
    last = None
    for attempt in range(3):
        try:
            url = 'https://modelscope.cn/api/v1/models/'+entry['repo']+'/repo?'+urllib.parse.urlencode(
                {'Revision': entry['revision'], 'FilePath': entry['path']})
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == entry['bytes']:
                with partial.open('rb') as existing:
                    complete_hash = hashlib.file_digest(existing, 'sha256').hexdigest()
                if complete_hash == entry['sha256']:
                    partial.replace(target)
                    return {'file': str(target), 'cached': True, 'bytes': entry['bytes'], 'resumed_complete': True}
                partial.unlink()
                offset = 0
            request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(request, timeout=90) as response:
                resume = bool(offset and response.status == 206)
                if resume and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('Unexpected resumed byte range')
                digest = hashlib.sha256()
                if resume:
                    with partial.open('rb') as existing:
                        while data := existing.read(4*1024*1024):
                            digest.update(data)
                with partial.open('ab' if resume else 'wb') as out:
                    while data := response.read(4*1024*1024):
                        out.write(data)
                        digest.update(data)
            if partial.stat().st_size != entry['bytes'] or digest.hexdigest() != entry['sha256']:
                partial.unlink(missing_ok=True)
                raise ValueError('Downloaded model file does not match the frozen official manifest')
            partial.replace(target)
            return {'file': str(target), 'cached': False, 'bytes': entry['bytes'], 'resumed_from': offset if resume else 0}
        except Exception as error:
            last = error
            time.sleep(2*(attempt+1))
    raise last


if __name__ == '__main__':
    Path('models-ready.json').unlink(missing_ok=True)
    plan = json.loads(Path('models.json').read_text())
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, f) for f in plan['files']]
        for future in concurrent.futures.as_completed(futures):
            print(json.dumps(future.result()), flush=True)
    Path('models-ready.json').write_text(json.dumps({'complete': True, 'seconds': time.monotonic()-started,
         'source': plan['source'], 'bytes':sum(f['bytes'] for f in plan['files'])})+'\n')
