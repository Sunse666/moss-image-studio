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
