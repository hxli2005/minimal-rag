"""Transfer the existing full corpus export, then resume the A800 pipeline."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/semantic-tool-20261016"
REMOTE = "/root/autodl-tmp/semantic-tool-20261016"
HELPER = RUN / "connection-a800/remote.py"
STATE = RUN / "corpus-transfer-state.json"
state = {"started": time.time(), "pid": os.getpid(), "stage": "validate_local_export"}


def save(stage, **fields):
    state.update(stage=stage, updated=time.time(), **fields)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(STATE)
    print(json.dumps(state), flush=True)


def ssh(*args):
    for attempt in range(3):
        result = subprocess.run([sys.executable, str(HELPER), *args], timeout=3600)
        if result.returncode == 0:
            return
        if attempt < 2:
            time.sleep(10)
    raise RuntimeError("SSH transfer/command failed after retries")


def remote_python(code):
    ssh("run", "cd " + REMOTE + " && /root/miniconda3/bin/python -c " + shlex.quote(code))


try:
    save("validate_local_export")
    corpus = RUN / "enterprise.jsonl.gz"
    with gzip.open(corpus, "rt") as stream:
        documents = sum(1 for _ in stream)
    assert documents == 511962
    digest = hashlib.file_digest(corpus.open("rb"), "sha256").hexdigest()
    frozen = json.loads((ROOT / "data/benchmarks/enterprise/manifest.json").read_text())
    source = {**next(f for f in frozen["files"] if f["path"] == "data/documents/test.parquet"),
              "repo": frozen["dataset"], "revision": frozen["revision"], "documents": documents,
              "export_sha256": digest, "export_bytes": corpus.stat().st_size,
              "export_method": "Existing corpus.sqlite SELECT id,title,text ORDER BY rowid; identical full released corpus and duplicate aliases"}
    source_path = RUN / "enterprise-source.json"
    source_path.write_text(json.dumps(source, indent=2) + "\n")
    remote_python("""import json,os,signal,subprocess,time
from pathlib import Path
info=json.loads(Path('full-process.json').read_text()); pid=info['pid']
args=subprocess.run(['ps','-p',str(pid),'-o','args='],capture_output=True,text=True).stdout
if 'full_index.py' in args:
    Path('full-before-direct-upload.json').write_text(Path('full-state.json').read_text())
    os.killpg(pid,signal.SIGTERM)
    time.sleep(2)
Path('full-state.json').write_text(json.dumps({'stage':'corpus_direct_upload','status':'preparing','prior_pid':pid,'updated':time.time()},indent=2))
""")
    save("uploading", bytes=corpus.stat().st_size, documents=documents)
    ssh("put", str(corpus), REMOTE + "/enterprise.jsonl.gz.uploading")
    ssh("put", str(source_path), REMOTE + "/enterprise-source.json.uploading")
    ssh("put", str(ROOT / "experiments/semantic-tool-20261016/full_index.py"), REMOTE + "/full_index.py")
    save("verify_and_start")
    remote_python("""import hashlib,json,subprocess,time
from pathlib import Path
source=json.loads(Path('enterprise-source.json.uploading').read_text())
corpus=Path('enterprise.jsonl.gz.uploading')
assert corpus.stat().st_size == source['export_bytes']
assert hashlib.file_digest(corpus.open('rb'),'sha256').hexdigest() == source['export_sha256']
corpus.replace('enterprise.jsonl.gz')
Path('enterprise-source.json.uploading').replace('enterprise-source.json')
deadline=1792237800.0
with Path('full.log').open('a') as log:
    p=subprocess.Popen(['/root/miniconda3/bin/python','-u','full_index.py','--prepared-corpus','--deadline',str(deadline)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
Path('full-process.json').write_text(json.dumps({'pid':p.pid,'started':time.time(),'deadline':deadline,'reason':'Verified full local corpus upload replaces slow mirror download'},indent=2))
print(json.dumps({'full_pid':p.pid,'corpus_verified':True}))
""")
    save("complete", status="complete")
except Exception as error:
    save("failed", status="failed", error=repr(error))
    raise
