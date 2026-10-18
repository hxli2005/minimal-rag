"""Package completed private grading outputs and execution evidence for return."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT=Path('/root/autodl-tmp/browsecomp-frozen-judge-20261017')
BUNDLE=ROOT/'bundle'
state=json.loads((ROOT/'official-state.json').read_text())
assert state['status']=='completed'
assert len(list((BUNDLE/'evals/frozen-20261017').glob('*_eval.json')))==830
source_files=['scripts_evaluation/evaluate_run.py','scripts_evaluation/evaluate_run_single_gpu.py','uv.lock']
metadata={
    'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip(),
    'source_sha256':{name:hashlib.sha256((BUNDLE/name).read_bytes()).hexdigest() for name in source_files},
    'original_bundle_sha256':hashlib.sha256((ROOT/'bundle.tar.gz').read_bytes()).hexdigest(),
    'model_config_sha256':hashlib.sha256((Path((ROOT/'model-path.txt').read_text())/'config.json').read_bytes()).hexdigest(),
}
(ROOT/'runtime-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
with tarfile.open(ROOT/'results.tar.gz','w:gz') as archive:
    archive.add(BUNDLE/'evals',arcname='evals')
    for pattern in ('*.json','*.log','*.py'):
        for path in ROOT.glob(pattern):
            archive.add(path,arcname='runtime/'+path.name)
print(json.dumps({'result_sha256':hashlib.sha256((ROOT/'results.tar.gz').read_bytes()).hexdigest(),'bytes':(ROOT/'results.tar.gz').stat().st_size,'metadata':metadata},indent=2))
