"""Detached nohup launcher; the existing selected keychain item goes only into env."""
import argparse
import fcntl
import os
import subprocess
import sys
from shared import DATA, HERE, OUT, ROOT, now, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--script', choices=['run', 'workflow'], default='run')
    args, rest = parser.parse_known_args()
    OUT.mkdir(parents=True, exist_ok=True)
    # One owner for the shared phase budget, including after a detached launch.
    lock = (OUT / 'runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
    env['BROWSECOMP_RUN_LOCK_FD'] = str(lock.fileno())
    if not env.get('OPENROUTER_API_KEY'):
        key = subprocess.check_output(['security', 'find-generic-password', '-s', 'xiaobu.openrouter.api-key',
                                       '-a', 'lhx', '-w'], text=True, stderr=subprocess.DEVNULL, timeout=30).strip()
        if not key:
            raise RuntimeError('Selected keychain item is empty')
        env['OPENROUTER_API_KEY'] = key
    command = ['nohup', sys.executable, '-u', str(HERE / f'{args.script}.py'), *rest]
    with (OUT / f'{args.name}.log').open('a') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                   pass_fds=(lock.fileno(),))
    receipt = {'pid': process.pid, 'started': now(), 'command': command}
    save(OUT / f'{args.name}.launch.json', receipt)
    print(receipt)


if __name__ == '__main__':
    main()
