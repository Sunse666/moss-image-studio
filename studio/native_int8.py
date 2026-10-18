"""Native INT8 ConvRot linears with streamed INT8 parameters and FP32 scales.

Only the experimental INT8 model imports the optional Comfy Kitchen Triton
kernel. No BF16 matmul fallback is permitted in this path.
"""
from contextlib import contextmanager
import os
import torch


def kernel_runtime():
    # Disposable workers can reuse the same GPU/shape-specific tuning results.
    os.environ.setdefault('TRITON_CACHE_AUTOTUNING', '1')
    import comfy_kitchen
    from comfy_kitchen.backends.triton.quantization import int8_linear
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError('Native INT8 requires an Ampere or newer CUDA GPU')
    if not comfy_kitchen.registry.is_available('triton'):
        raise RuntimeError('Comfy Kitchen Triton INT8 backend unavailable; select BF16 decode')
    return int8_linear


class NativeInt8Linear(torch.nn.Module):
    # MOSS reads proj.q.weight.dtype to choose its activation dtype. This empty
    # marker preserves BF16 activations without allocating floating-point weights.
    _dtype_marker = torch.empty(0, dtype=torch.bfloat16)

    def __init__(self, in_features, out_features, kernel, stats):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.qweight = torch.nn.Parameter(torch.empty(out_features, in_features, device='meta',
                                                      dtype=torch.int8), requires_grad=False)
        self.weight_scale = torch.nn.Parameter(torch.empty(out_features, 1, device='meta',
                                                           dtype=torch.float32), requires_grad=False)
        self.kernel, self.stats = kernel, stats

    @property
    def weight(self):
        return self._dtype_marker

    def forward(self, x):
        if self.qweight.is_meta or self.weight_scale.is_meta:
            raise RuntimeError('Native INT8 block weights were not loaded by WeeLLM')
        if self.qweight.dtype != torch.int8 or self.weight_scale.dtype != torch.float32:
            raise RuntimeError('Native INT8 streaming changed weight/scale precision')
        result = self.kernel(x.to(torch.bfloat16), self.qweight, self.weight_scale,
                             out_dtype=torch.bfloat16, convrot=True, convrot_groupsize=256)
        self.stats['linear_calls'] += 1
        return result


@contextmanager
def native_transformer(path, stats, kernel=None):
    """Scoped worker-only adaptation; original installed WeeLLM stays untouched."""
    from unittest.mock import patch
    from accelerate.utils import set_module_tensor_to_device
    from weellm.models.transformers import moss_image_transformer_2d_model as module
    from weellm.io.safetensors.disk_seek import SafetensorsDiskSeeker
    from int8_config import read_header
    kernel = kernel or kernel_runtime()
    header, _ = read_header(path)
    quantized = {k for k, v in header.items() if k != '__metadata__' and v.get('dtype') == 'I8'}

    class RawInt8Seeker(SafetensorsDiskSeeker):
        def get_tensors(self, keys, device='cpu', dtype=None, process_comfy=True):
            expanded = list(keys)
            for key in keys:
                if key in quantized:
                    expanded.append(key.removesuffix('.weight') + '.weight_scale')
            raw = super().get_tensors(list(dict.fromkeys(expanded)), device=device,
                                      dtype=None, process_comfy=False)
            result = {}
            for key, value in raw.items():
                name = key.removesuffix('.weight') + '.qweight' if key in quantized else key
                if dtype is not None and value.is_floating_point() and not key.endswith('.weight_scale'):
                    value = value.to(dtype)
                result[name] = value
            return result

    streamer = module.MOSSImageTransformerStreamer
    original_load = streamer._load_model_on_meta
    original_apply = streamer.apply_state_dict

    def load(cls, model_cls, transformer_dir, device, dtype, seeker):
        model = original_load(model_cls, transformer_dir, device, dtype, seeker)
        for block in model.layers:
            for name, linear in list(block.proj.items()):
                block.proj[name] = NativeInt8Linear(linear.in_features, linear.out_features, kernel, stats).eval()
                stats['projections'] += 1
        return model

    def apply(self, state_dict, skip_errors=True):
        rest = {}
        for key, value in state_dict.items():
            if key.endswith('.weight_scale'):
                set_module_tensor_to_device(self.model, key, self.device, value=value, dtype=torch.float32)
            else:
                rest[key] = value
        original_apply(self, rest, skip_errors=skip_errors)

    with patch.object(module, 'get_seeker', lambda *args, **kwargs: RawInt8Seeker(path)), \
            patch.object(streamer, '_load_model_on_meta', classmethod(load)), \
            patch.object(streamer, 'apply_state_dict', apply):
        # apply_state_dict must remain patched through generation so evicted
        # scales are restored in FP32 on every streaming reload.
        yield
