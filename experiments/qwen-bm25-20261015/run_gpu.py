"""Bounded sequential GPU runner. Probes and final computations share one deadline."""
import argparse
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path


def write(value):
    Path('state.json').write_text(json.dumps(value, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-seconds', type=int, default=3600)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    begun = time.time()
    deadline = begun + args.max_seconds
    state = {'status':'running','started':begun,'deadline':deadline,'stages':[]}
    write(state)
    try:
        for stage, probe in [('embed',True),('rerank',True),('embed',False),('rerank',False)]:
            if stage=='embed' and not probe:
                e=json.loads(Path('output/embedding-probe.json').read_text())
                r=json.loads(Path('output/reranker-probe.json').read_text())
                estimate=e['estimated_all_chunks_seconds']+r['estimated_max_20000_pairs_seconds']+e['load_seconds']+r['load_seconds']
                state['estimated_remaining_seconds']=estimate
                if estimate*1.2 > deadline-time.time():
                    state['status']='probe_exceeds_compute_window'
                    break
            label=stage+('-probe' if probe else '')
            state['current_stage']=label
            write(state)
            command=[sys.executable,'gpu.py',stage,'--deadline',str(deadline),'--batch-size',str(args.batch_size)]
            if probe: command.append('--probe')
            start=time.time()
            with Path(label+'.log').open('w') as log:
                p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time()))
            state['stages'].append({'stage':label,'returncode':p.returncode,'seconds':time.time()-start})
            if p.returncode:
                raise RuntimeError(label+' failed; inspect its log')
        else:
            state['status']='completed'
    except Exception as error:
        state.update(status='failed',error_type=type(error).__name__,error=str(error))
    finally:
        state['finished']=time.time()
        write(state)
        with tarfile.open('results.tar.gz','w:gz') as tar:
            for f in [Path('state.json'),*Path('.').glob('*.log'),*Path('output').glob('*.json')]:
                tar.add(f,arcname=str(f))
        print(json.dumps(state),flush=True)


if __name__=='__main__':
    main()
