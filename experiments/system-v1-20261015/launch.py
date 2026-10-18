"""Detached single-owner launch. Secrets enter the child environment only."""
import fcntl
import os
import subprocess
import sys
from night import HERE,ROOT,NIGHT,now,save,read

NIGHT.mkdir(parents=True,exist_ok=True)
lock=(NIGHT/'runner.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
old=read(NIGHT/'active-process.json',{})
if old.get('pid'):
    try: os.kill(old['pid'],0)
    except ProcessLookupError: pass
    else: raise RuntimeError('Recorded phase process is still alive')
env=dict(os.environ,PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'experiments/src'),PYTHONUNBUFFERED='1',
         HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',NIGHT_LOCK_FD=str(lock.fileno()))
if not env.get('OPENROUTER_API_KEY'):
    env['OPENROUTER_API_KEY']=subprocess.check_output(['security','find-generic-password','-s','xiaobu.openrouter.api-key','-a','lhx','-w'],text=True,stderr=subprocess.DEVNULL,timeout=30).strip()
    if not env['OPENROUTER_API_KEY']:raise RuntimeError('Selected keychain entry empty')
version=sys.argv[1]
command=['nohup',sys.executable,'-u',str(HERE/'controller.py'),version]
with (NIGHT/f'{version}.log').open('a') as log:
    p=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
        start_new_session=True,pass_fds=(lock.fileno(),))
receipt={'pid':p.pid,'started':now(),'version':version}
save(NIGHT/'launch.json',receipt)
subprocess.Popen(['caffeinate','-i','-w',str(p.pid)],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
print(receipt)
