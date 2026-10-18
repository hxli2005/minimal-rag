"""Detach a stage; fetch the explicitly selected key only into the judge environment."""
import argparse
import json
import os
import subprocess
import sys

from common import DATA, HERE, OUT, ROOT, now, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run', 'judge'])
    parser.add_argument('--name', required=True)
    args, rest = parser.parse_known_args()
    directory = DATA if args.stage == 'prepare' else OUT
    directory.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONUNBUFFERED='1', HF_HUB_DISABLE_PROGRESS_BARS='1')
    if args.stage == 'judge' and not env.get('OPENROUTER_API_KEY'):
        key = subprocess.check_output(['security', 'find-generic-password', '-s', 'xiaobu.openrouter.api-key', '-a', 'lhx', '-w'],
                                      text=True, stderr=subprocess.DEVNULL, timeout=30).strip()
        if not key:
            raise RuntimeError('Selected keychain item is empty')
        env['OPENROUTER_API_KEY'] = key
    command = ['nohup', sys.executable, '-u', str(HERE / (args.stage + '.py')), *rest]
    with (directory / (args.name + '.log')).open('a') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {'pid': process.pid, 'started': now(), 'command': command, 'stage': args.stage}
    save(directory / (args.name + '.launch.json'), receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
