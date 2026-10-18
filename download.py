"""Download official MOSS models with resumable, hash-verified weight transfers."""
import argparse
import hashlib
import concurrent.futures
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from studio.downloads import fetch
from studio import turbo_config


def download_turbo(workers):
    root = turbo_config.DIRECTORY
    root.mkdir(parents=True, exist_ok=True)
    marker = root / '.download-complete'
    marker.unlink(missing_ok=True)
    for name, (size, expected_hash) in turbo_config.FILES.items():
        path = fetch(f'https://huggingface.co/{turbo_config.REPO}/resolve/{turbo_config.REVISION}/{name}',
                     root / name, workers=max(1, min(32, workers)))
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        if path.stat().st_size != size or digest != expected_hash:
            raise RuntimeError('Turbo adapter checksum mismatch: ' + name)
        print('Verified Turbo', name, flush=True)
    marker.write_text(turbo_config.REVISION, encoding='utf-8')
    print('Downloaded and verified Turbo 4-step adapter. Also requires MOSS-Image-SFT.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', nargs='?', choices=['SFT', 'Base', 'Turbo'], default='SFT')
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--parallel-files', type=int, default=4)
    args = parser.parse_args()
    if args.model == 'Turbo':
        download_turbo(args.workers)
        return
    name = 'MOSS-Image-' + args.model
    repo = 'OpenMOSS-Team/' + name
    root = Path(__file__).resolve().parent / 'models' / name
    info = HfApi().model_info(repo, files_metadata=True)
    snapshot_download(repo, revision=info.sha, local_dir=root,
                      ignore_patterns=['assets/*', '*.md', '*.py', '*.safetensors', '.gitattributes'])
    def download(file):
        destination = (root / file.rfilename).resolve()
        if not destination.is_relative_to(root.resolve()):
            raise ValueError('Invalid model file path')
        path = fetch(f'https://huggingface.co/{repo}/resolve/{info.sha}/{file.rfilename}',
                     destination, workers=max(1, min(32, args.workers)))
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        if not file.lfs or digest != file.lfs.sha256:
            raise RuntimeError('Weight checksum mismatch: ' + str(path))
        print('Verified', file.rfilename, flush=True)
    files = [file for file in info.siblings if file.rfilename.endswith('.safetensors')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(4, args.parallel_files))) as executor:
        list(executor.map(download, files))
    (root / '.download-complete').write_text(info.sha, encoding='utf-8')
    print('Downloaded and verified', name)


if __name__ == '__main__':
    main()
