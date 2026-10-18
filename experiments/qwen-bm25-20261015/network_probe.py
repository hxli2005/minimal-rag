"""Download four frozen 128 MiB prefixes; retain them for the resumable downloader."""
import concurrent.futures
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

LIMIT = 128 * 1024 * 1024


def one(entry):
    target = Path('models') / entry['role'] / entry['path']
    partial = target.with_suffix(target.suffix + '.partial')
    partial.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists() or target.exists():
        raise RuntimeError('Probe must not overwrite existing model data')
    url = 'https://modelscope.cn/api/v1/models/' + entry['repo'] + '/repo?' + urllib.parse.urlencode(
        {'Revision': entry['revision'], 'FilePath': entry['path']})
    request = urllib.request.Request(url, headers={'Range': f'bytes=0-{LIMIT-1}'})
    started = time.monotonic()
    count = 0
    with urllib.request.urlopen(request, timeout=30) as response, partial.open('wb') as f:
        while count < LIMIT:
            if time.monotonic()-started > 120:
                break
            data = response.read(min(4*1024*1024, LIMIT-count))
            if not data:
                break
            f.write(data)
            count += len(data)
        code = response.status
    return {'role': entry['role'], 'file': entry['path'], 'bytes': count,
            'seconds': time.monotonic()-started, 'http_status': code}


if __name__ == '__main__':
    plan = json.loads(Path('models.json').read_text())
    big = [f for f in plan['files'] if f['bytes'] > LIMIT]
    selected = [f for f in big if f['role']=='embedding'][:3] + [f for f in big if f['role']=='reranker'][:1]
    started = time.monotonic()
    results, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for future in concurrent.futures.as_completed([pool.submit(one,f) for f in selected]):
            try:
                results.append(future.result())
            except Exception as error:
                errors.append({'type': type(error).__name__, 'message': str(error)})
    seconds = time.monotonic()-started
    size = sum(r['bytes'] for r in results)
    summary = {'seconds': seconds, 'bytes': size, 'aggregate_MiB_per_second': size/1024**2/seconds,
               'complete': not errors and len(results)==4 and all(r['bytes']==LIMIT for r in results),
               'files': results, 'errors': errors, 'new_model_inference_calls': 0}
    Path('network-probe.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
