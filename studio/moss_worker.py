"""MOSS inference in a disposable process; no GPU dependencies in the web server."""
import json
import os
import sys
import time
from pathlib import Path
from timing import StageTimings

def event(**values):
    print("MOSS_EVENT " + json.dumps(values), flush=True)

def main():
    timings = StageTimings()
    spec = json.loads(sys.stdin.readline())
    model = spec["model"]
    turbo = bool(spec.get("turbo"))
    if turbo:
        import turbo_config
        turbo_config.require_adapter()
        if model != "OpenMOSS-Team/MOSS-Image-SFT":
            raise ValueError("Turbo 4-step requires MOSS-Image-SFT")
        if spec["steps"] != 4 or spec["cfg"] != 1 or spec["negative"]:
            raise ValueError("Turbo requires four steps, CFG 1, and no negative prompt")
    if model not in ("OpenMOSS-Team/MOSS-Image-SFT", "OpenMOSS-Team/MOSS-Image-Base"):
        raise ValueError("Unsupported model")
    local = Path(__file__).resolve().parent.parent / "models" / model.split("/")[-1]
    if not (local / "model_index.json").exists():
        raise RuntimeError("Model not downloaded. Run download.ps1 first for " + model)
    for component in ("text_encoder", "transformer", "vae"):
        folder = local / component
        indexes = list(folder.glob("*.safetensors.index.json"))
        if indexes:
            files = set(json.loads(indexes[0].read_text())["weight_map"].values())
        else:
            files = {path.name for path in folder.glob("*.safetensors")}
        if not files or any(not (folder / name).is_file() for name in files):
            raise RuntimeError("Incomplete model download. Rerun download.ps1 for " + model)
    with timings.stage("imports"):
        import torch
        from weellm import WeeTextToImagePipeline
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run setup.ps1 and check your NVIDIA driver.")
    timings.synchronize = torch.cuda.synchronize
    free, total = torch.cuda.mem_get_info()
    budget = min(float(os.getenv("MOSS_VRAM_GB", "2.5")), free / 2**30 - 0.5)
    if budget < 0.75:
        raise RuntimeError("Insufficient free VRAM. Close other GPU applications.")
    torch.cuda.reset_peak_memory_stats()
    event(stage="Loading MOSS weights from SSD")
    with timings.stage("model_load"):
        pipe = WeeTextToImagePipeline.from_pretrained(
            str(local), device="cuda", torch_dtype=torch.bfloat16,
            vram_budget_gb=budget, ram_budget_gb=float(os.getenv("MOSS_RAM_GB", "2")),
            cache_to_ram=False, vae_tile_size=int(os.getenv("MOSS_VAE_TILE", "256")))
    def progress(iterable=None, total=None):
        started = time.monotonic()
        count = len(iterable)
        for index, value in enumerate(iterable):
            event(stage="Sampling", step=index, total=count)
            yield value
            seconds = (time.monotonic() - started) / (index + 1)
            event(step=index + 1, total=count, sec_per_it=seconds,
                  eta=round(seconds * (count - index - 1)))
        event(stage="Decoding image", eta=0)
    pipe._pipeline.progress_bar = progress
    event(stage="Encoding prompt")
    if turbo:
        from rapid_moss import generate
        from prompt_cache import PromptCache, encoder_identity
        from weellm.models.transformers.base_transformer_streamer import BaseTransformerStreamer
        prompt_cache = None
        # Private requests bypass fingerprinting and all cache filesystem access.
        if not spec.get("private"):
            try:
                prompt_cache = PromptCache(Path(__file__).parent / "prompt_cache",
                                           encoder_identity(local, pipe._pipeline.encode_prompt))
            except (OSError, TypeError, ValueError):
                pass  # Optional cache must never prevent normal encoding.
        def encode_prompt(prompt):
            if prompt_cache is None:
                return pipe._pipeline.encode_prompt(prompt)
            return prompt_cache.encode(prompt, pipe._pipeline.encode_prompt,
                                       pipe._pipeline._execution_device)
        # The custom sampler bypasses WeeBasePipeline.__call__; retain its
        # conservative activation headroom used by streaming cache decisions.
        overhead = (spec["width"] // 16) * (spec["height"] // 16) * 500 * 1024
        chunk_tokens = max(128, min(2048, int(os.getenv("MOSS_TURBO_CHUNK", "1024"))))
        cache_mb = max(0, min(256, int(os.getenv("MOSS_TURBO_CACHE_MB", "128"))))
        overhead += (cache_mb + 64) * 2**20
        BaseTransformerStreamer._estimated_vram_overhead_bytes = overhead
        BaseTransformerStreamer._estimated_ram_overhead_bytes = overhead
        image = generate(pipe._pipeline, turbo_config.DIRECTORY, spec["prompt"],
                         spec["width"], spec["height"], spec["seed"], progress,
                         timings=timings, chunk_tokens=chunk_tokens, cache_mb=cache_mb,
                         vram_budget_gb=budget, report=event, encode_prompt=encode_prompt)
        event(prompt_cache=prompt_cache.status if prompt_cache else
              ("private" if spec.get("private") else "disabled"))
        event(turbo_adapter=turbo_config.REPO, turbo_revision=turbo_config.REVISION)
    else:
        with timings.stage("generation"):
            image = pipe.generate(prompt=spec["prompt"], negative_prompt=spec["negative"],
                                  width=spec["width"], height=spec["height"],
                                  num_inference_steps=spec["steps"], guidance_scale=spec["cfg"], seed=spec["seed"])
    temporary = Path(spec["output"] + ".part")
    with timings.stage("png_save"):
        image.save(temporary, format="PNG")
        temporary.replace(spec["output"])
    event(stage_seconds=timings.finish(), gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
          peak_allocated_mb=round(torch.cuda.max_memory_allocated()/2**20),
          peak_reserved_mb=round(torch.cuda.max_memory_reserved()/2**20))

if __name__ == "__main__":
    main()
