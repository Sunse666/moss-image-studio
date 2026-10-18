"""Optional pinned INT8 distilled checkpoint. Does not replace BF16 assets."""
from pathlib import Path
import json
import struct

MODEL = "toxicdog/MOSS-Image-SFT-Turbo-INT8-ConvRot"
REVISION = "a04afd66c4f9a3b41658a611906723b55ee40301"
FILENAME = "diffusion_models/moss_image_sft_turbo_int8_convrot.safetensors"
SIZE = 5413707240
SHA256 = "2a837150ea1b116d939e7516d82b1469c0afc6e311b00a196df557cd258ed223"
DIRECTORY = Path(__file__).resolve().parents[1] / "models" / "MOSS-Image-SFT-Turbo-INT8"
WEIGHTS = DIRECTORY / "transformer.safetensors"


def available():
    try:
        return (WEIGHTS.stat().st_size == SIZE and
                (DIRECTORY / ".download-complete").read_text().strip() == REVISION)
    except OSError:
        return False


def require_weights():
    if not available():
        raise RuntimeError("Experimental INT8 checkpoint missing or incomplete. Run ./download.ps1 -Model INT8.")


def read_header(path):
    with Path(path).open('rb') as source:
        length = struct.unpack('<Q', source.read(8))[0]
        if length > 8 * 2**20:
            raise ValueError('Unexpected checkpoint header size')
        return json.loads(source.read(length)), length + 8


def validate_checkpoint(weights=WEIGHTS, original=None):
    """Check all original tensor shapes and exact supported quant metadata, CPU only."""
    original = Path(original or DIRECTORY.parent / 'MOSS-Image-SFT' / 'transformer')
    expected = {}
    for path in original.glob('*.safetensors'):
        header, _ = read_header(path)
        expected.update({k: v for k, v in header.items() if k != '__metadata__'})
    if not expected:
        raise ValueError('Original SFT transformer headers required for INT8 compatibility check')
    header, base = read_header(weights)
    native = json.loads(header.get('__metadata__', {}).get('_quantization_metadata', '{}'))
    recipes = native.get('layers', {})
    # Current checkpoint has unfused projections, matching WeeLLM exactly.
    for key, value in expected.items():
        actual = header.get(key)
        if actual is None or actual['shape'] != value['shape']:
            raise ValueError('INT8 tensor missing or incompatible: ' + key)
        if actual['dtype'] == 'I8':
            prefix = key.removesuffix('.weight')
            meta = header.get(prefix + '.comfy_quant')
            scale = header.get(prefix + '.weight_scale')
            if not scale or scale['shape'] not in ([actual['shape'][0]], [actual['shape'][0], 1]):
                raise ValueError('Incomplete INT8 scales/metadata: ' + key)
            info = recipes.get(prefix)
            if info is None:
                if not meta:
                    raise ValueError('Missing INT8 recipe: ' + key)
                start, end = meta['data_offsets']
                if end - start > 1024:
                    raise ValueError('Unexpected quantization metadata length')
                with Path(weights).open('rb') as source:
                    source.seek(base + start)
                    info = json.loads(source.read(end - start))
            if info != {'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 256}:
                raise ValueError('Unsupported INT8 recipe: ' + key)
        elif actual['dtype'] not in ('BF16', 'F32', 'F16'):
            raise ValueError('Unsupported weight dtype: ' + key)
    count = sum(v.get('dtype') == 'I8' for k, v in header.items() if k != '__metadata__')
    if count != 252:
        raise ValueError('Expected exactly 252 quantized block projections')
    return {'matched_tensors': len(expected), 'quantized_projections': count, 'format': 'INT8 ConvRot-256'}
