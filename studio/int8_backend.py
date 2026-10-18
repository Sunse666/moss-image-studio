"""INT8 checkpoint storage streamed into BF16 layers; no RapidMOSS adapter.

Euler/simple flow schedule follows ComfyUI's documented schedule for this
checkpoint, not the separate four-step data-space recurrence.
"""
from contextlib import contextmanager
import json

import torch


def checkpoint_seeker(path):
    """Bridge native Comfy header recipes to WeeLLM's existing ConvRot decoder.

    No checkpoint conversion or floating copy is written. Raw INT8 remains
    compressed during disk prefetch and is reconstructed by WeeLLM on demand.
    """
    from weellm.io.safetensors.disk_seek import SafetensorsDiskSeeker
    from weellm.io.safetensors.comfy_dequant import process_comfy_tensors
    from int8_config import read_header
    header, _ = read_header(path)
    recipes = json.loads(header.get('__metadata__', {}).get('_quantization_metadata', '{}')).get('layers', {})

    class NativeInt8Seeker(SafetensorsDiskSeeker):
        def get_tensors(self, keys, device='cpu', dtype=None, process_comfy=True):
            expanded = list(keys)
            for key in keys:
                prefix = key.removesuffix('.weight')
                if key.endswith('.weight') and prefix in recipes:
                    expanded.append(prefix + '.weight_scale')
            # Keep FP32 scales until dequantization instead of rounding them on disk read.
            result = super().get_tensors(list(dict.fromkeys(expanded)), device=device,
                                        dtype=None, process_comfy=False)
            for key, value in list(result.items()):
                prefix = key.removesuffix('.weight')
                if key.endswith('.weight') and prefix in recipes:
                    raw = json.dumps(recipes[prefix]).encode('utf-8')
                    result[prefix + '.comfy_quant'] = torch.tensor(list(raw), dtype=torch.uint8, device=device)
                if dtype is not None and value.is_floating_point() and not key.endswith('.weight_scale'):
                    result[key] = value.to(dtype=dtype)
            return process_comfy_tensors(result) if process_comfy else result

    return NativeInt8Seeker(path)


@contextmanager
def transformer_weights(path):
    # Scoped to pipeline construction inside its disposable worker process.
    # Encoder, tokenizer, VAE and configs still come from the original SFT tree.
    from unittest.mock import patch
    from weellm.models.transformers import moss_image_transformer_2d_model as module
    with patch.object(module, "get_seeker", lambda *args, **kwargs: checkpoint_seeker(path)):
        yield


def simple_sigmas(steps=8, shift=5.0):
    # Comfy ModelSamplingDiscreteFlow: sigma table for t=1..1000; simple
    # scheduler indexes from the top in intervals len(table)/steps.
    t = torch.arange(1, 1001, dtype=torch.float32) / 1000
    table = shift * t / (1 + (shift - 1) * t)
    return torch.tensor([float(table[-(1 + int(i * 1000 / steps))])
                         for i in range(steps)] + [0.0], dtype=torch.float32)


@torch.no_grad()
def sample_latents(transformer, embeds, mask, width, height, seed, progress):
    shape = (1, transformer.config.latent_channels, height // 16, width // 16)
    latents = torch.randn(shape, generator=torch.Generator('cpu').manual_seed(seed),
                          dtype=torch.float32).to(embeds.device)
    sigmas = simple_sigmas().to(embeds.device)
    for index in progress(range(8)):
        prediction = transformer(latents, sigmas[index] * 1000, embeds, mask).sample.float()
        latents = latents + prediction * (sigmas[index + 1] - sigmas[index])
    return latents


@torch.no_grad()
def generate(pipe, prompt, width, height, seed, progress, timings, encode_prompt):
    from rapid_moss import decode_latents
    with timings.stage('prompt_encode'):
        embeds, mask = encode_prompt(prompt)
    with timings.stage('sampling'):
        latents = sample_latents(pipe.transformer, embeds, mask, width, height, seed, progress)
    del embeds, mask
    return decode_latents(pipe, latents, timings)
