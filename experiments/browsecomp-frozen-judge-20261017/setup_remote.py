"""One-off setup on the budgeted A800; public dependencies and pinned BF16 weights."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/browsecomp-frozen-judge-20261017')
BUNDLE = ROOT / 'bundle'
ENV = dict(os.environ, PATH='/root/miniconda3/bin:/usr/local/bin:/usr/bin:/bin',
           UV_CACHE_DIR=str(ROOT/'uv-cache'), UV_LINK_MODE='hardlink',
           UV_PYTHON_DOWNLOADS='never', GIT_CONFIG_COUNT='1',
           GIT_CONFIG_KEY_0='url.file://'+str(ROOT/'tevatron.git')+'.insteadOf',
           GIT_CONFIG_VALUE_0='https://github.com/texttron/tevatron.git')


def run(cmd, **kwargs):
    subprocess.run(cmd, env=ENV, check=True, **kwargs)


def install():
    run(['tar', '-xzf', str(ROOT/'python310-linux.tar.gz'), '-C', str(ROOT)])
    run(['tar', '-xzf', str(ROOT/'tevatron.git.tar.gz'), '-C', str(ROOT)])
    run(['chown', '-R', 'root:root', str(ROOT/'tevatron.git')])
    uv = '/root/miniconda3/bin/uv'
    python = str(ROOT/'python/bin/python3.10')
    run([uv, 'venv', '--python', python, str(BUNDLE/'.venv')])
    original = (ROOT/'locked-requirements.txt').read_text()
    lines = [line for line in original.splitlines() if not line.startswith('tevatron @ git+')]
    (ROOT/'index-requirements.txt').write_text('\n'.join(lines)+'\n')
    run([uv,'pip','install','--python',str(BUNDLE/'.venv/bin/python'),
         '--require-hashes','--no-deps','-r',str(ROOT/'index-requirements.txt'),
         '--index-url','https://mirrors.aliyun.com/pypi/simple/'])
    run([uv,'sync','--frozen','--python',python], cwd=BUNDLE)
    code = "import torch,vllm,transformers; assert torch.__version__.split('+')[0]=='2.7.0'; assert vllm.__version__=='0.9.0.1'; assert transformers.__version__=='4.53.2'; print(torch.cuda.get_device_name(),torch.cuda.is_bf16_supported())"
    run([str(BUNDLE/'.venv/bin/python'), '-c', code], cwd=BUNDLE)
    (ROOT/'environment-ready.json').write_text(json.dumps({'ready':True,'time':time.time()}))


def weights():
    from modelscope.hub.snapshot_download import snapshot_download
    manifest = json.loads((ROOT/'weight-manifest.json').read_text())
    # Single-card containers have a 60 GiB memory disk. Keep two shards on SSD.
    disk_names=[row['file'] for row in manifest['weights'][:2]]
    disk = Path(snapshot_download('Qwen/Qwen3-32B', revision=manifest['weights'][0]['revision'],
                cache_dir=str(ROOT/'weight-disk'), allow_patterns=disk_names, max_workers=2))
    memory_names=[row['file'] for row in manifest['weights'][2:]]
    model = Path(snapshot_download('Qwen/Qwen3-32B', revision=manifest['weights'][0]['revision'],
                 cache_dir='/dev/shm/browsecomp-qwen3', allow_patterns=memory_names, max_workers=8))
    for name in disk_names:
        (model/name).symlink_to(disk/name)
    for row in manifest['weights']:
        digest = hashlib.sha256()
        with (model/row['file']).open('rb') as source:
            for block in iter(lambda:source.read(8*1024*1024), b''):
                digest.update(block)
        assert digest.hexdigest() == row['sha256'], row['file']
    run(['tar','-xzf',str(ROOT/'model-config.tar.gz'),'-C',str(ROOT)])
    for file in (ROOT/'model-config').iterdir():
        shutil.copy2(file, model/file.name)
    (ROOT/'model-path.txt').write_text(str(model))
    (ROOT/'weights-ready.json').write_text(json.dumps({'ready':True,'time':time.time(),'hf_revision':manifest['hf_revision'],'verified_shards':len(manifest['weights'])}))


if __name__ == '__main__':
    if len(sys.argv)>1:
        {'install':install, 'weights':weights}[sys.argv[1]]()
    else:
        ROOT.mkdir(exist_ok=True)
        run(['tar','-xzf',str(ROOT/'bundle.tar.gz'),'-C',str(ROOT)])
        run(['tar','-xzf',str(ROOT/'setup-assets.tar.gz'),'-C',str(ROOT)])
        run(['/root/miniconda3/bin/python','-m','pip','install','uv==0.13.1','modelscope','--index-url','https://mirrors.aliyun.com/pypi/simple/'])
        processes = []
        for phase in ('install','weights'):
            log=(ROOT/(phase+'.log')).open('a')
            proc=subprocess.Popen([sys.executable,__file__,phase],env=ENV,stdout=log,stderr=subprocess.STDOUT)
            processes.append((phase,proc))
        (ROOT/'setup-state.json').write_text(json.dumps({'status':'running','started':time.time(),'children':{phase:p.pid for phase,p in processes}}))
        codes={phase:proc.wait() for phase,proc in processes}
        (ROOT/'setup-state.json').write_text(json.dumps({'status':'ready' if all(c==0 for c in codes.values()) else 'failed','ended':time.time(),'exit_codes':codes}))
