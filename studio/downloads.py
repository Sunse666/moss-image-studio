"""Resumable bounded HTTP range downloads for large files on unreliable links."""
import concurrent.futures
from pathlib import Path
import time
import urllib.request
import threading
import requests

CHUNK = 1024 * 1024
ASSEMBLY_LOCK = threading.Lock()

def fetch(url, destination, workers=12):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    def resolve():
        for attempt in range(12):
            try:
                req = urllib.request.Request(url, headers={'Range': 'bytes=0-0'})
                with urllib.request.urlopen(req, timeout=45) as response:
                    return int(response.headers['Content-Range'].split('/')[-1]), response.url
            except (OSError, KeyError):
                if attempt == 11:
                    raise
                time.sleep(min(15, attempt + 1))
    total, resolved_url = resolve()
    sessions = threading.local()
    refresh_lock = threading.Lock()
    if destination.exists() and destination.stat().st_size == total:
        return destination
    staging = destination.with_name(destination.name + f'.parts-{CHUNK}')
    staging.mkdir(exist_ok=True)
    chunks = (total + CHUNK - 1) // CHUNK
    def part(index):
        nonlocal resolved_url
        start = index * CHUNK
        end = min(total, start + CHUNK) - 1
        target = staging / str(index)
        if target.exists() and target.stat().st_size == end - start + 1:
            return target.stat().st_size
        for attempt in range(12):
            try:
                if not hasattr(sessions, 'session'):
                    sessions.session = requests.Session()
                with sessions.session.get(resolved_url, headers={'Range': f'bytes={start}-{end}'}, timeout=(30,90)) as response:
                    if response.status_code == 429:
                        time.sleep(min(120, int(response.headers.get('Retry-After', '30'))))
                        raise RuntimeError('Server rate limit; retrying after backoff')
                    if response.status_code == 403:
                        with refresh_lock:
                            _, resolved_url = resolve()
                        raise RuntimeError('Download link refreshed; retrying range')
                    if response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/'):
                        raise RuntimeError('Server returned an incorrect byte range')
                    data = response.content
                if len(data) != end - start + 1:
                    raise RuntimeError('Incomplete byte range')
                temporary = target.with_suffix('.tmp')
                temporary.write_bytes(data)
                temporary.replace(target)
                return len(data)
            except Exception:
                if attempt == 11:
                    raise
                time.sleep(min(10, attempt + 1))
    done = 0
    last_report = 0
    pending = list(range(chunks))
    for batch in range(3):
        failures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(part, index): index for index in pending}
            for future in concurrent.futures.as_completed(futures):
                try:
                    done += future.result()
                except Exception as error:
                    failures.append(futures[future])
                    print(f'{destination.name}: retrying chunk {futures[future]} ({type(error).__name__})', flush=True)
                    continue
                if done - last_report < 128 * 2**20 and done != total:
                    continue
                last_report = done
                print(f'{destination.name}: {done/2**20:.0f}/{total/2**20:.0f} MiB', flush=True)
        if not failures:
            break
        pending = failures
    else:
        raise RuntimeError(f'{destination.name}: {len(failures)} chunks failed; rerun to resume')
    with ASSEMBLY_LOCK:
        temporary = destination.with_name(destination.name + '.assembling')
        with temporary.open('wb') as output:
            for index in range(chunks):
                with (staging / str(index)).open('rb') as source:
                    import shutil
                    shutil.copyfileobj(source, output)
        temporary.replace(destination)
        for child in staging.iterdir():
            child.unlink()
        staging.rmdir()
    return destination

if __name__ == '__main__':
    import sys
    fetch(sys.argv[1], sys.argv[2])
