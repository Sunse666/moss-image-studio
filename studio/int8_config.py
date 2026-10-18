"""Optional pinned INT8 distilled checkpoint. Does not replace BF16 assets."""
from pathlib import Path
import json
import struct
import importlib.util

MODEL = "toxicdog/MOSS-Image-SFT-Turbo-INT8-ConvRot"
REVISION = "a04afd66c4f9a3b41658a611906723b55ee40301"
FILENAME = "diffusion_models/moss_image_sft_turbo_int8_convrot.safetensors"
SIZE = 5413707240
SHA256 = "2a837150ea1b116d939e7516d82b1469c0afc6e311b00a196df557cd258ed223"
DIRECTORY = Path(__file__).resolve().parents[1] / "models" / "MOSS-Image-SFT-Turbo-INT8"
WEIGHTS = DIRECTORY / "transformer.safetensors"


def native_available():
    return (importlib.util.find_spec('comfy_kitchen') is not None and
            importlib.util.find_spec('triton') is not None)


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


class OriginalHeadersMissing(ValueError):
    """Lite install: BF16 shards absent; use architecture_shapes() instead."""


def architecture_shapes(transformer_dir=None):
    """Expected tensors for Lite installs without BF16 shards, CPU only.

    Names come from the original SFT transformer index (kept by Lite downloads);
    shapes come from WeeLLM's MOSS transformer built on the meta device from
    the original config, i.e. exactly what the streamer will load into.
    """
    from accelerate import init_empty_weights
    from weellm.models.transformers.moss_image_transformer_2d_model import MOSSImageTransformer
    folder = Path(transformer_dir or DIRECTORY.parent / 'MOSS-Image-SFT' / 'transformer')
    indexes = list(folder.glob('*.safetensors.index.json'))
    if not indexes:
        raise ValueError('Original SFT transformer index required for INT8 compatibility check')
    names = set(json.loads(indexes[0].read_text(encoding='utf-8'))['weight_map'])
    with init_empty_weights():
        model = MOSSImageTransformer.from_config(MOSSImageTransformer.load_config(str(folder)))
    shapes = {k: list(v.shape) for k, v in model.state_dict().items()}
    unknown = sorted(names - shapes.keys())
    if unknown:
        raise ValueError('Original index has tensors unknown to WeeLLM: ' + ', '.join(unknown[:3]))
    return {name: {'shape': shapes[name]} for name in names}


def validate_checkpoint(weights=WEIGHTS, original=None, expected=None):
    """Check all original tensor shapes and exact supported quant metadata, CPU only."""
    if expected is None:
        original = Path(original or DIRECTORY.parent / 'MOSS-Image-SFT' / 'transformer')
        expected = {}
        for path in original.glob('*.safetensors'):
            header, _ = read_header(path)
            expected.update({k: v for k, v in header.items() if k != '__metadata__'})
        if not expected:
            raise OriginalHeadersMissing('Original SFT transformer headers required for INT8 compatibility check')
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
