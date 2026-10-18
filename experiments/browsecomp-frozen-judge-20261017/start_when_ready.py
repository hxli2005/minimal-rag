"""Wait for public setup, then run the official judge once before the hard deadline."""
import json
import os
from pathlib import Path
import time

ROOT = Path('/root/autodl-tmp/browsecomp-frozen-judge-20261017')
deadline = json.loads((ROOT/'control.json').read_text())['inference_deadline']
while time.time() < deadline:
    if (ROOT/'setup-state.json').exists():
        state=json.loads((ROOT/'setup-state.json').read_text())
        if state['status']=='failed':
            raise RuntimeError('Setup failed; no grading started')
        if state['status']=='ready':
            os.execv('/root/miniconda3/bin/python', ['python', str(ROOT/'run_remote.py')])
    time.sleep(10)
raise RuntimeError('Deadline reached before setup finished')
