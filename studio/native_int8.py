"""Native INT8 ConvRot linears with streamed INT8 parameters and FP32 scales.

Only the experimental INT8 model imports the optional Comfy Kitchen Triton
kernel. No BF16 matmul fallback is permitted in this path.
"""
from collections import namedtuple
from contextlib import contextmanager
import os
import threading
import torch

# quantize(x) -> (x_int8, x_scale); matmul(x_int8, x_scale, qweight, scale) -> BF16.
Int8Kernels = namedtuple('Int8Kernels', 'quantize matmul')
CONVROT_GROUPSIZE = 256
# q, k and v consume the same normalized activation: rotate/quantize it once.
SHARED_INPUT = ('q', 'k', 'v')
DTYPES = {'I8': torch.int8, 'F32': torch.float32, 'BF16': torch.bfloat16}


def _host_buffer(nbytes, pin):
    """Uninitialized read buffer; pinned buffers come from PyTorch's caching host allocator."""
    return torch.empty(nbytes, dtype=torch.uint8, pin_memory=pin)


def read_raw(path, header, data_base, keys, pin):
    """Read exact checkpoint bytes for *keys* into one buffer, viewed per tensor.

    WeeLLM's reader zero-fills a fresh bytearray per tensor and copies pageable
    memory to the GPU; with read-ahead threads that delayed kernel launches by
    ~35 ms per block in a no-image benchmark. Pinned memory makes H2D truly
    asynchronous. Bytes are identical, so model outputs are unchanged.
    """
    spans, total = [], 0
    for key in keys:
        meta = header[key]  # KeyError for unknown tensors, as WeeLLM raises
        if meta['dtype'] not in DTYPES:
            raise ValueError(f'Unsupported native INT8 tensor dtype {meta["dtype"]} for {key}')
        start, end = meta['data_offsets']
        spans.append((key, meta, start, end - start, total))
        total += -(-(end - start) // 64) * 64  # keep every view element-aligned
    try:
        buffer = _host_buffer(total, pin)
    except RuntimeError:
        pin = False  # e.g. pinned allocation refused; pageable reads still work
        buffer = _host_buffer(total, pin)
    with open(path, 'rb', buffering=0) as source:
        for key, meta, start, size, offset in spans:
            view = memoryview(buffer[offset:offset + size].numpy())
            source.seek(data_base + start)
            done = 0
            while done < size:  # raw reads may return fewer bytes than requested
                count = source.readinto(view[done:])
                if not count:
                    raise ValueError(f'Short read for {key}: expected {size} B, got {done} B')
                done += count
    return {key: buffer[offset:offset + size].view(DTYPES[meta['dtype']]).reshape(meta['shape'])
            for key, meta, start, size, offset in spans}, pin


def kernel_runtime():
    # Disposable workers can reuse the same GPU/shape-specific tuning results.
    os.environ.setdefault('TRITON_CACHE_AUTOTUNING', '1')
    import comfy_kitchen
    import triton
    from comfy_kitchen.backends.triton import quantization
    from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError('Native INT8 requires an Ampere or newer CUDA GPU')
    if not comfy_kitchen.registry.is_available('triton'):
        raise RuntimeError('Comfy Kitchen Triton INT8 backend unavailable; select BF16 decode')
    # Comfy Kitchen's autotune list tops out at block_k=32/64. On the RTX 3050
    # Laptop, block_k=128 ran the 3,155-token q/ffn_up_gate/ffn_down GEMMs
    # 1.9-2.3x faster. Tiles only change scheduling; INT32 sums are exact.
    gemm = triton.autotune(configs=[
        triton.Config({'block_m': 64, 'block_n': 128, 'block_k': 128, 'group_size_m': 8}, num_stages=3, num_warps=4),
        triton.Config({'block_m': 128, 'block_n': 128, 'block_k': 128, 'group_size_m': 8}, num_stages=3, num_warps=8),
        triton.Config({'block_m': 128, 'block_n': 256, 'block_k': 64, 'group_size_m': 8}, num_stages=3, num_warps=8),
    ], key=['n', 'k'])(quantization._int8_matmul_dequant_per_row_kernel.fn)  # m varies by prompt length

    # Same steps as comfy_kitchen 0.2.38 int8_linear(convrot=True) with
    # per-channel scales, split so one quantized activation can feed several
    # GEMMs. INT32 accumulation is exact, so outputs stay bitwise identical.
    def quantize(x):
        x = x.reshape(-1, x.shape[-1])
        h = _build_hadamard(CONVROT_GROUPSIZE, device=x.device, dtype=x.dtype)
        return quantization.triton_quantize_rowwise(_rotate_activation(x, h, CONVROT_GROUPSIZE))

    def matmul(x_int8, x_scale, weight, weight_scale):
        m, k = x_int8.shape
        n = weight.shape[0]
        output = torch.empty((m, n), device=x_int8.device, dtype=torch.bfloat16)
        gemm[
            lambda meta: (triton.cdiv(m, meta['block_m']) * triton.cdiv(n, meta['block_n']),)](
            a_ptr=x_int8, b_ptr=weight, c_ptr=output, a_scale_ptr=x_scale,
            b_scale_ptr=weight_scale.reshape(n).contiguous(), bias_ptr=x_int8, m=m, n=n, k=k,
            stride_am=x_int8.stride(0), stride_ak=x_int8.stride(1),
            stride_bk=weight.stride(1), stride_bn=weight.stride(0),
            stride_cm=output.stride(0), stride_cn=output.stride(1), has_bias=False)
        return output

    return Int8Kernels(quantize, matmul)


class NativeInt8Linear(torch.nn.Module):
    # MOSS reads proj.q.weight.dtype to choose its activation dtype. This empty
    # marker preserves BF16 activations without allocating floating-point weights.
    _dtype_marker = torch.empty(0, dtype=torch.bfloat16)

    def __init__(self, in_features, out_features, kernel, stats, shared=None, last=True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.qweight = torch.nn.Parameter(torch.empty(out_features, in_features, device='meta',
                                                      dtype=torch.int8), requires_grad=False)
        self.weight_scale = torch.nn.Parameter(torch.empty(out_features, 1, device='meta',
                                                           dtype=torch.float32), requires_grad=False)
        self.kernel, self.stats = kernel, stats
        # shared: dict owned by one block's q/k/v; last releases the cached input.
        self.shared, self.last = shared, last

    @property
    def weight(self):
        return self._dtype_marker

    def _quantized_input(self, x):
        shared = self.shared
        # Reuse only for the very same tensor object, unmodified since quantization.
        if shared is not None and shared.get('source') is x and shared['version'] == x._version:
            self.stats['activation_reuses'] += 1
            return shared['value']
        value = self.kernel.quantize(x.to(torch.bfloat16))
        self.stats['activation_quantizations'] += 1
        if shared is not None:
            shared.update(source=x, version=x._version, value=value)
        return value

    def forward(self, x):
        if self.qweight.is_meta or self.weight_scale.is_meta:
            raise RuntimeError('Native INT8 block weights were not loaded by WeeLLM')
        if self.qweight.dtype != torch.int8 or self.weight_scale.dtype != torch.float32:
            raise RuntimeError('Native INT8 streaming changed weight/scale precision')
        try:
            x_int8, x_scale = self._quantized_input(x)
            result = self.kernel.matmul(x_int8, x_scale, self.qweight, self.weight_scale)
        finally:
            if self.shared is not None and self.last:
                self.shared.clear()
        self.stats['linear_calls'] += 1
        return result.reshape(*x.shape[:-1], self.out_features)


@contextmanager
def native_transformer(path, stats, kernel=None, prefetch_depth=None, pinned=None):
    """Scoped worker-only adaptation; original installed WeeLLM stays untouched."""
    from unittest.mock import patch
    from accelerate.utils import set_module_tensor_to_device
    from weellm.models.transformers import moss_image_transformer_2d_model as module
    from weellm.io.safetensors.disk_seek import SafetensorsDiskSeeker
    from int8_config import read_header
    kernel = kernel or kernel_runtime()
    for key in ('activation_quantizations', 'activation_reuses', 'pinned_reads', 'pageable_reads'):
        stats.setdefault(key, 0)
    pinned = os.getenv('MOSS_INT8_PINNED', '1') != '0' if pinned is None else pinned
    header, data_base = read_header(path)
    quantized = {k for k, v in header.items() if k != '__metadata__' and v.get('dtype') == 'I8'}
    lock = threading.Lock()  # WeeLLM reads from several prefetch threads

    class RawInt8Seeker(SafetensorsDiskSeeker):
        def get_tensors(self, keys, device='cpu', dtype=None, process_comfy=True):
            expanded = list(keys)
            for key in keys:
                if key in quantized:
                    expanded.append(key.removesuffix('.weight') + '.weight_scale')
            if self._keymap_cls is not None:
                raise RuntimeError('Native INT8 expects unmapped checkpoint keys')
            # Pin only CPU staging reads (streamed blocks); direct GPU loads
            # (resident tensors, sync fallback) stay pageable and are copied once.
            pin = pinned and str(device) == 'cpu' and torch.cuda.is_available()
            raw, pin = read_raw(path, header, data_base, list(dict.fromkeys(expanded)), pin)
            with lock:
                stats['pinned_reads' if pin else 'pageable_reads'] += 1
            result = {}
            for key, value in raw.items():
                name = key.removesuffix('.weight') + '.qweight' if key in quantized else key
                value = value.to(device=device)  # same order as WeeLLM: device, then dtype
                if dtype is not None and value.is_floating_point() and not key.endswith('.weight_scale'):
                    value = value.to(dtype)
                result[name] = value
            return result

    streamer = module.MOSSImageTransformerStreamer
    original_load = streamer._load_model_on_meta
    original_apply = streamer.apply_state_dict
    original_depth = streamer._compute_prefetch_depth

    def load(cls, model_cls, transformer_dir, device, dtype, seeker):
        model = original_load(model_cls, transformer_dir, device, dtype, seeker)
        for block in model.layers:
            shared = {}
            for name, linear in list(block.proj.items()):
                group = name in SHARED_INPUT
                block.proj[name] = NativeInt8Linear(linear.in_features, linear.out_features, kernel, stats,
                                                    shared=shared if group else None,
                                                    last=not group or name == SHARED_INPUT[-1]).eval()
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

    def depth(self):
        # WeeLLM sizes disk read-ahead from a RAM budget meant for BF16 blocks.
        # INT8 blocks are ~113 MB, so a few parallel reads keep the SSD busy.
        return original_depth(self) if prefetch_depth is None else max(1, min(6, int(prefetch_depth)))

    with patch.object(module, 'get_seeker', lambda *args, **kwargs: RawInt8Seeker(path)), \
            patch.object(streamer, '_load_model_on_meta', classmethod(load)), \
            patch.object(streamer, '_compute_prefetch_depth', depth), \
            patch.object(streamer, 'apply_state_dict', apply):
        # apply_state_dict must remain patched through generation so evicted
        # scales are restored in FP32 on every streaming reload.
        yield
