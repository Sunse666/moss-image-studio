"""Rapid-MOSS data-space sampler adapted for WeeLLM's streamed BF16 DiT.

SPDX-License-Identifier: Apache-2.0
Adapted from inference.py at revision 1ec327b9fac5a0aa47426aaba6565892616ce30e:
https://huggingface.co/darask0/MOSS-Image_4-step_adapter_lora
Changes: disk-backed per-projection LoRA, token chunking, streamed resident
time-path restoration, and reuse of WeeLLM's component eviction hooks.

Algorithm/weights: darask0/MOSS-Image_4-step_adapter_lora, Apache-2.0.
LoRA remains unmerged and FP32. Projection tensors transfer on demand; a bounded
first-fit cache can reuse a subset. They are never registered as model parameters,
so WeeLLM's block reloads/evictions cannot round, merge, or evict the adapter.
"""
import math
from contextlib import ExitStack, nullcontext

import torch
from safetensors import safe_open
from torch.nn import functional as F


class RapidMossAdapter:
    def __init__(self, transformer, root, *, chunk_tokens=1024, cache_mb=128, vram_budget_gb=2.5):
        self.transformer = transformer
        self.stack = ExitStack()
        self.hooks = []
        self.chunk_tokens = max(128, min(2048, int(chunk_tokens)))
        self.cache_limit = max(0, min(256, int(cache_mb))) * 2**20
        self.vram_limit = int(vram_budget_gb * 2**30)
        self.cache = {}
        self.time_signature = None
        self.stats = dict(cache_hits=0, cache_misses=0, cache_bytes=0,
                          transferred_bytes=0, chunks=0, time_path_loads=0,
                          chunk_tokens=self.chunk_tokens, cache_limit_bytes=self.cache_limit)
        try:
            self.lora = self.stack.enter_context(safe_open(str(root / "lora.safetensors"), framework="pt", device="cpu"))
            self.time = self.stack.enter_context(safe_open(str(root / "time_embedding.safetensors"), framework="pt", device="cpu"))
            self.data = self.stack.enter_context(safe_open(str(root / "dataspace.safetensors"), framework="pt", device="cpu"))
            expected = set()
            for i, layer in enumerate(transformer.layers):
                for name, linear in layer.proj.items():
                    key = f"diffusion_model.layers.{i}.proj.{name}"
                    keys = [key + suffix for suffix in (".lora_down.weight", ".lora_up.weight", ".alpha")]
                    expected.update(keys)
                    down, up, alpha = [self.lora.get_tensor(k) for k in keys]
                    rank = down.shape[0]
                    if (down.dtype != torch.float32 or up.dtype != torch.float32
                            or rank < 1 or tuple(down.shape) != (rank, linear.in_features)
                            or tuple(up.shape) != (linear.out_features, rank)
                            or alpha.numel() != 1 or not math.isfinite(float(alpha))):
                        raise ValueError("Incompatible Turbo projection: " + key)
                    scale = float(alpha) / rank
                    self.hooks.append(linear.register_forward_hook(self._lora_hook(key, scale)))
            if set(self.lora.keys()) != expected:
                raise ValueError("Turbo adapter has unexpected projection keys")
            self.time_keys = {"time_embedding." + k for k in transformer.time_embedding.state_dict()}
            self.time_keys.update(("shared_block_modulation.bias", "video_head_modulation.bias"))
            if set(self.time.keys()) != self.time_keys:
                raise ValueError("Incompatible Turbo time embedding keys")
            for key in self.time_keys:
                value = self.time.get_tensor(key)
                target = transformer.get_parameter(key)
                if value.dtype != torch.float32 or value.shape != target.shape:
                    raise ValueError("Incompatible Turbo time embedding: " + key)
            self.Q = self.data.get_tensor("latent_mix.Q")
            self.codes = self.data.get_tensor("step_codes")
            channels = transformer.config.latent_channels
            if (self.Q.dtype != torch.float32 or self.Q.shape != (channels, channels)
                    or self.codes.shape != (4,) or not torch.isfinite(self.Q).all()
                    or not torch.isfinite(self.codes).all()):
                raise ValueError("Invalid Turbo data-space mixing matrix or step codes")
            # Installed AFTER WeeLLM's resident and text-encoder eviction hooks.
            self.hooks.append(transformer.register_forward_pre_hook(self._restore_time_path))
        except Exception:
            self.close()
            raise

    def _lora_hook(self, key, scale):
        def hook(module, inputs, output):
            x = inputs[0].reshape(-1, inputs[0].shape[-1])
            y = output.reshape(-1, output.shape[-1])
            down, up = self._weights(key, output.device)
            # Cap estimated FP32 activation temporaries at 64 MiB. Larger
            # projections automatically get smaller chunks, without changing dtype.
            per_token = 4 * (module.in_features + 2 * module.out_features + down.shape[0])
            chunk = min(self.chunk_tokens, max(128, (64 * 2**20 // per_token // 128) * 128))
            for start in range(0, x.shape[0], chunk):
                delta = scale * F.linear(F.linear(x[start:start + chunk].float(), down), up)
                y[start:start + chunk].add_(delta.to(output.dtype))
                self.stats["chunks"] += 1
            return y.reshape(output.shape)
        return hook

    def _weights(self, key, device):
        cache_key = (key, str(device))
        if cache_key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[cache_key]
        self.stats["cache_misses"] += 1
        down = self.lora.get_tensor(key + ".lora_down.weight").to(device)
        up = self.lora.get_tensor(key + ".lora_up.weight").to(device)
        size = down.numel() * down.element_size() + up.numel() * up.element_size()
        self.stats["transferred_bytes"] += size
        fits = self.stats["cache_bytes"] + size <= self.cache_limit
        if fits and device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(device)
            fits = (free >= 256 * 2**20 and
                    torch.cuda.memory_reserved(device) <= self.vram_limit - 64 * 2**20)
        if fits:
            # Pin a fixed first-fit subset. LRU would thrash on cyclic scans of
            # all 252 projections, retaining no tensors until their next use.
            self.cache[cache_key] = (down, up)
            self.stats["cache_bytes"] += size
        return down, up

    @staticmethod
    def _time_signature(module):
        parameters = [p for name in ("time_embedding", "shared_block_modulation", "video_head_modulation")
                      for p in getattr(module, name).parameters()]
        if any(p.is_meta for p in parameters):
            raise RuntimeError("WeeLLM did not load Turbo resident weights before forward")
        return tuple((id(p), p.data_ptr(), p.dtype, p.device, p._version) for p in parameters)

    def _restore_time_path(self, module, args):
        from accelerate.utils import set_module_tensor_to_device
        device = args[0].device
        if self._time_signature(module) == self.time_signature:
            return
        # Do not convert the whole DiT: streamed blocks remain BF16/meta.
        # Reapply after resident reloads, which otherwise restore original BF16.
        for name in ("time_embedding", "shared_block_modulation", "video_head_modulation"):
            component = getattr(module, name)
            if any(p.is_meta for p in component.parameters()):
                raise RuntimeError("WeeLLM did not load Turbo resident weights before forward")
            component.float()
        for key in self.time_keys:
            set_module_tensor_to_device(module, key, device, value=self.time.get_tensor(key), dtype=torch.float32)
        self.time_signature = self._time_signature(module)
        self.stats["time_path_loads"] += 1

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.cache.clear()
        self.stack.close()


@torch.no_grad()
def sample_latents(transformer, Q, codes, embeds, mask, width, height, seed, progress):
    if len(codes) != 4 or width % 16 or height % 16:
        raise ValueError("Turbo requires exactly four codes and dimensions divisible by 16")
    device = embeds.device
    z = torch.randn(1, Q.shape[0], height // 16, width // 16,
                    generator=torch.Generator("cpu").manual_seed(seed), dtype=torch.float32).to(device)
    Q, codes = Q.to(device), codes.to(device)
    x0 = torch.zeros_like(z)
    for r in progress(range(4)):
        x = z + torch.einsum("oc,bchw->bohw", Q, x0)
        x0 = z - transformer(x, codes[r], embeds, mask).sample.float()
    return x0


@torch.no_grad()
def generate(pipe, adapter_root, prompt, width, height, seed, progress, *,
             timings=None, chunk_tokens=1024, cache_mb=128, vram_budget_gb=2.5, report=None):
    stage = timings.stage if timings else lambda name: nullcontext()
    # Keep WeeLLM's existing encode/transformer/VAE offload hooks intact.
    with stage("prompt_encode"):
        embeds, mask = pipe.encode_prompt(prompt)
    with stage("adapter_setup"):
        adapter = RapidMossAdapter(pipe.transformer, adapter_root, chunk_tokens=chunk_tokens,
                                   cache_mb=cache_mb, vram_budget_gb=vram_budget_gb)
    try:
        with stage("sampling"):
            latents = sample_latents(pipe.transformer, adapter.Q, adapter.codes,
                                     embeds, mask, width, height, seed, progress)
    finally:
        with stage("adapter_cleanup"):
            adapter.close()
        if report:
            report(adapter_stats=adapter.stats.copy())
    del embeds, mask
    vae = pipe.vae
    z = latents.to(vae.dtype)
    bn, eps = vae.bn, vae.config.batch_norm_eps
    z = z * (bn.running_var + eps).sqrt().to(z).view(1, -1, 1, 1) + bn.running_mean.to(z).view(1, -1, 1, 1)
    b, c, h, w = z.shape
    z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
    # pipe.vae.decode is WeeLLM's patched decode: evicts DiT before decoding.
    with stage("vae_decode"):
        image = vae.decode(z, return_dict=False)[0].clamp(-1, 1)
        pipe.maybe_free_model_hooks()
        result = pipe.image_processor.postprocess(image, output_type="pil")[0]
    return result
