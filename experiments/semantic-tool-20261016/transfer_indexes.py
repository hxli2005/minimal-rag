"""Copy the frozen indexes directly between the two task-owned cloud hosts."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import threading
import time

import paramiko

AUTH = Path('/tmp/semantic-source-connection')
SOURCE = '/root/autodl-tmp/semantic-tool-20261016/'
state = dict(status='transferring', started=time.time(), files=0, bytes=0)
lock = threading.Lock()
connection_lock = threading.Lock()
local = threading.local()
clients = []


def sftp():
    if not getattr(local, 'sftp', None):
        parts = shlex.split((AUTH/'ssh-command.txt').read_text())
        port = int(parts[parts.index('-p')+1])
        user, host = next(x for x in parts if '@' in x).split('@', 1)
        client = paramiko.SSHClient()
        client.load_host_keys(str(AUTH/'known_hosts'))
        with connection_lock:
            client.connect(host, port=port, username=user, password=(AUTH/'password.txt').read_text().strip(),
                           look_for_keys=False, allow_agent=False, compress=True,
                           timeout=30, banner_timeout=30, auth_timeout=30)
            clients.append(client)
        local.client = client
        local.sftp = client.open_sftp()
        local.sftp.get_channel().settimeout(120)
    return local.sftp


def save():
    temporary = Path('index-transfer-state.tmp')
    temporary.write_text(json.dumps(state, indent=2)+'\n')
    temporary.replace('index-transfer-state.json')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(4*1024*1024):
            result.update(block)
    return result.hexdigest()


def transfer(item):
    target = Path(item['target'])
    target.parent.mkdir(parents=True, exist_ok=True)
    if not (target.exists() and target.stat().st_size == item['bytes'] and digest(target) == item['sha256']):
        partial = target.with_suffix(target.suffix+'.receiving')
        for attempt in range(3):
            try:
                sftp().get(SOURCE+item['target'], str(partial), max_concurrent_prefetch_requests=64)
                assert partial.stat().st_size == item['bytes'] and digest(partial) == item['sha256']
                partial.replace(target)
                break
            except Exception:
                if getattr(local, 'client', None):
                    local.client.close()
                local.sftp = None
                if attempt == 2:
                    raise
                time.sleep(5)
    with lock:
        state['files'] += 1
        state['bytes'] += item['bytes']
        state['last_file'] = item['target']
        state['updated'] = time.time()
        save()


def main():
    enterprise = json.loads(Path('enterprise-index/manifest.json').read_text())
    browsecomp = json.loads(Path('browsecomp-source.json').read_text())
    tasks = [dict(item, target='enterprise-index/'+item['file']) for item in enterprise['shards']]
    tasks += [dict(item, target='browsecomp/'+Path(item['path']).name) for item in browsecomp['files']]
    state['total_files'] = len(tasks)
    save()
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(transfer, tasks))
        state.update(status='complete', finished=time.time(), all_file_hashes_match=True)
        save()
        Path('index-transfer-ready.json').write_text(json.dumps(state, indent=2)+'\n')
    except Exception as error:
        state.update(status='failed', error=str(error), finished=time.time())
        save()
        raise
    finally:
        for client in clients:
            client.close()
        shutil.rmtree(AUTH, ignore_errors=True)
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
