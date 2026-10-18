"""Resume unstarted B questions, then recover only technical failures."""
import fcntl
import json
from pathlib import Path
import sys
import time

from minimal_rag_experiments import benchmark
import retry_generation
from transport_control import install, wait_upstream


def main(argv):
    run = Path(argv[argv.index('--output')+1]).resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run/'transport-run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    install()
    if (run/'config.json').exists() and '--resume' not in argv:
        argv.append('--resume')
    while wait_upstream():
        if (run/'summary.json').exists() and json.loads((run/'summary.json').read_text())['complete']:
            break
        benchmark.cli(argv)
        if '--resume' not in argv:
            argv.append('--resume')
        if not json.loads((run/'summary.json').read_text())['complete']:
            time.sleep(30)
    else:
        return
    load_cases = benchmark.load_cases
    try:
        retry_generation.main(run)
    finally:
        benchmark.load_cases = load_cases
    # A recovered normal answer may still need its first valid original judge result.
    while wait_upstream() and not json.loads((run/'summary.json').read_text())['complete']:
        benchmark.cli(argv)
        if not json.loads((run/'summary.json').read_text())['complete']:
            time.sleep(30)


if __name__ == '__main__':
    main(sys.argv[1:])
