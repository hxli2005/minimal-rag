"""Start the authorized parallel B groups as soon as the deployed service is ready."""
import fcntl
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/semantic-tool-20261016'
URL = 'http://127.0.0.1:8765'


def main():
    lock = (RUN/'service-launch.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while not (RUN/'STOP').exists():
        try:
            with urlopen(URL+'/health', timeout=5) as response:
                health = json.load(response)
            break
        except OSError:
            time.sleep(30)
    else:
        return
    assert health['documents'] == dict(enterprise=511962, browsecomp=100195)
    checks = {}
    for corpus, folder in [('enterprise', 'enterprise'), ('browsecomp', 'browsecomp-plus')]:
        payload = dict(corpus=corpus, query='annual report', limit=2, offset=0)
        request = Request(URL+'/search', json.dumps(payload).encode(), {'Content-Type':'application/json'})
        with urlopen(request, timeout=120) as response:
            result = json.load(response)
        assert len(result['hits']) == 2
        path = ROOT/'data/benchmarks'/folder/'corpus.sqlite'
        connection = sqlite3.connect(path.as_uri()+'?mode=ro', uri=True)
        try:
            for hit in result['hits']:
                assert math.isfinite(hit['score'])
                assert connection.execute('SELECT 1 FROM documents WHERE id=?', (hit['document_id'],)).fetchone()
        finally:
            connection.close()
        checks[corpus] = result
    (RUN/'query-service-preflight.json').write_text(json.dumps(dict(checked_at=time.time(), health=health,
        generic_query='annual report', benchmark_questions_used=False, results=checks), indent=2)+'\n')
    subprocess.run([sys.executable, str(ROOT/'experiments/semantic-tool-20261016/start_b_groups.py'),
                    '--url', URL], cwd=ROOT, check=True)
    (RUN/'service-launch-complete.json').write_text(json.dumps(dict(started_b_groups_at=time.time()))+'\n')


if __name__ == '__main__':
    main()
