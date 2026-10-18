"""Prepare the query service from frozen inputs without regenerating embeddings."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
from urllib.request import urlopen


def alive(pid):
    path = Path(f'/proc/{pid}/stat')
    return path.exists() and path.read_text().split()[2] != 'Z'


def main(deadline):
    state = dict(stage='dependencies', started=time.time(), deadline=deadline)
    transfer = server = None

    def save(**fields):
        state.update(updated=time.time(), **fields)
        temporary = Path('service-stage-state.tmp')
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace('service-stage-state.json')

    try:
        save()
        dependency = json.loads(Path('service-dependencies-process.json').read_text())['pid']
        while alive(dependency):
            if time.time() >= deadline:
                raise TimeoutError('Service preparation deadline')
            time.sleep(10)
        subprocess.run(['.gpu-venv/bin/python', '-c',
                        'import torch,transformers,numpy,paramiko; assert torch.cuda.is_bf16_supported(); '
                        'assert transformers.__version__=="4.53.2"; assert numpy.__version__=="1.26.4"'], check=True)
        auth = Path('/tmp/semantic-source-connection')
        auth.mkdir(mode=0o700, exist_ok=True)
        auth.chmod(0o700)
        with tarfile.open('/tmp/semantic-source-connection.tar.gz') as archive:
            for name in ('ssh-command.txt', 'password.txt', 'known_hosts', 'remote.py'):
                path = auth/name
                path.write_bytes(archive.extractfile(name).read())
                path.chmod(0o600)
        Path('/tmp/semantic-source-connection.tar.gz').unlink()
        with Path('index-transfer.log').open('a') as log:
            transfer = subprocess.Popen(['.gpu-venv/bin/python', '-u', 'transfer_indexes.py'],
                                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        Path('index-transfer-process.json').write_text(json.dumps(dict(pid=transfer.pid, started=time.time()))+'\n')
        save(stage='model_and_index_transfer', transfer_pid=transfer.pid)
        model_pid = json.loads(Path('model-download-process.json').read_text())['pid']
        while not (Path('models-ready.json').exists() and Path('index-transfer-ready.json').exists()):
            if transfer.poll() not in (None, 0):
                raise RuntimeError('Index transfer failed; see index-transfer-state.json')
            if not Path('models-ready.json').exists() and not alive(model_pid):
                raise RuntimeError('Frozen model download failed; see model-download.log')
            if time.time() >= deadline:
                raise TimeoutError('Service preparation deadline')
            time.sleep(10)
        command = ['.gpu-venv/bin/python', '-u', 'gpu.py', 'serve', '--model', 'models/embedding',
                   '--enterprise', 'enterprise-index', '--browsecomp', 'browsecomp', '--port', '8765']
        with Path('query-service.log').open('a') as log:
            server = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
        Path('query-service-process.json').write_text(json.dumps(dict(pid=server.pid, started=time.time(), command=command))+'\n')
        save(stage='loading', service_pid=server.pid)
        while time.time() < deadline:
            if server.poll() is not None:
                raise RuntimeError('Query service failed to load; see query-service.log')
            try:
                with urlopen('http://127.0.0.1:8765/health', timeout=5) as response:
                    health = json.load(response)
                assert health['documents'] == dict(enterprise=511962, browsecomp=100195)
                save(stage='ready', health=health, source_cpu_requires_shutdown=True)
                return
            except OSError:
                time.sleep(5)
        raise TimeoutError('Query service did not become ready before deadline')
    except Exception as error:
        save(stage='failed', error=str(error))
        if transfer is not None and transfer.poll() is None:
            transfer.terminate()
        if server is not None and server.poll() is None:
            server.terminate()
        raise
    finally:
        shutil.rmtree('/tmp/semantic-source-connection', ignore_errors=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--deadline', type=float, required=True)
    main(parser.parse_args().deadline)
