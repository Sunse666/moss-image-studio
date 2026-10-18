# Third-party notices

## Rapid-MOSS integration

`studio/rapid_moss.py` adapts the Rapid-MOSS inference implementation by darask0,
published under Apache-2.0 at:

https://huggingface.co/darask0/MOSS-Image_4-step_adapter_lora/blob/1ec327b9fac5a0aa47426aaba6565892616ce30e/inference.py

Local changes add disk-backed per-projection FP32 adapters, bounded token chunks,
restoration after WeeLLM resident reloads, and integration with the existing
WeeLLM component offload hooks. See [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).

Rapid-MOSS weights are downloaded separately from Hugging Face and are not
included in the repository. Their upstream license is Apache-2.0.

## Experimental INT8 checkpoint and sampler

The optional `toxicdog/MOSS-Image-SFT-Turbo-INT8-ConvRot` checkpoint is an
Apache-2.0 derivative of OpenMOSS-Team/MOSS-Image-SFT, pinned at revision
`a04afd66c4f9a3b41658a611906723b55ee40301`. Weights are downloaded separately.
Their author specifies eight steps, CFG 1, Euler/simple and flow shift 5.
`studio/int8_backend.py` implements that flow schedule mathematically; it does
not include ComfyUI source or apply the Rapid-MOSS four-step adapter. The
existing WeeLLM ConvRot reader reconstructs BF16 block weights on demand.

The optional native mode calls Comfy Kitchen 0.2.38's Triton INT8 linear
implementation directly (INT8 dot products, INT32 accumulation). Comfy Kitchen
is distributed under Apache-2.0; see https://github.com/Comfy-Org/comfy-kitchen.
No package source is copied into this repository. BF16 decoding remains an
independent compatibility option. Kernel compilation and tuning caches are
local runtime artifacts.
