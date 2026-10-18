"""Install the Windows Python 3.13 CUDA wheels with resumable downloads."""
import hashlib
from pathlib import Path
import subprocess
import sys
from studio import downloads
downloads.CHUNK = 8 * 1024 * 1024
fetch = downloads.fetch

WHEELS = [
    ('torch-2.11.0+cu128-cp313-cp313-win_amd64.whl', '6f367e62fd81b75cdf23ca4b75ced834d2db2cf98d1588ac935bde345de9de23'),
    ('torchvision-0.26.0+cu128-cp313-cp313-win_amd64.whl', '0232cb219927a52d6c98ff202f32d1cdf4802c2195a85fc1f1a0c1b0b4983a4d'),
]

def main():
    if sys.platform != 'win32' or sys.version_info[:2] != (3,13):
        raise SystemExit('This installer requires Windows with Python 3.13.')
    root = Path(__file__).resolve().parent / '.downloads'
    paths = []
    for filename, expected in WHEELS:
        path = fetch('https://download.pytorch.org/whl/cu128/' + filename.replace('+','%2B'), root / filename, workers=32)
        with path.open('rb') as source:
            actual = hashlib.file_digest(source, 'sha256').hexdigest()
        if actual != expected:
            raise RuntimeError('CUDA wheel checksum mismatch: ' + filename)
        paths.append(str(path))
    subprocess.run([sys.executable, '-m', 'pip', 'install', *paths], check=True)

if __name__ == '__main__':
    main()
