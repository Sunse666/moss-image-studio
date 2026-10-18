"""Download official MOSS models with resumable, hash-verified weight transfers."""
import argparse
import hashlib
import concurrent.futures
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from studio.downloads import fetch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', nargs='?', choices=['SFT', 'Base'], default='SFT')
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--parallel-files', type=int, default=4)
    args = parser.parse_args()
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
