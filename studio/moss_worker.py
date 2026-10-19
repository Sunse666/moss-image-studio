"""MOSS inference in a separate process; no GPU dependencies in the web server.

One-shot by default. A native INT8 job may set keep_warm: the process then
reports job_done and serves further jobs of the same profile from stdin with
the model, kernels and imports already loaded, until stdin closes.
"""
import json
import os
import sys
import time
from pathlib import Path
from timing import StageTimings
from contextlib import nullcontext, ExitStack
import int8_config

def event(**values):
    print("MOSS_EVENT " + json.dumps(values), flush=True)

def profile(spec):
    """Settings that fix what the loaded process can serve."""
    int8 = spec["model"] == int8_config.MODEL
    return (spec["model"], spec.get('int8_compute', 'bf16') if int8 else None, bool(spec.get("turbo")))


def check_job(spec):
    model, _, turbo = profile(spec)
    if model == int8_config.MODEL:
        if turbo or spec['steps'] != 8 or spec['cfg'] != 1 or spec['negative']:
            raise ValueError('INT8 uses its own eight-step Euler/simple sampler, CFG 1 and no negative prompt')
    if turbo:
        if model != "OpenMOSS-Team/MOSS-Image-SFT":
            raise ValueError("Turbo 4-step requires MOSS-Image-SFT")
        if spec["steps"] != 4 or spec["cfg"] != 1 or spec["negative"]:
            raise ValueError("Turbo requires four steps, CFG 1, and no negative prompt")
    if model not in ("OpenMOSS-Team/MOSS-Image-SFT", "OpenMOSS-Team/MOSS-Image-Base", int8_config.MODEL):
        raise ValueError("Unsupported model")


def reset_for_next_job(pipe, native_stats):
    """Undo per-job state so a warm process behaves like a fresh one.

    The VAE decode hook evicted every transformer block to meta, so WeeLLM's
    pinned-block bookkeeping must be recalibrated on the next first pass.
    """
    transformer = pipe._pipeline.transformer
    streamer = getattr(transformer, '_weellm_streamer', None)
    if streamer is not None:
        with streamer._lock:
            futures = list(streamer._h2d_futures.values()) + list(streamer._disk_futures.values())
            streamer._h2d_futures.clear()
            streamer._disk_futures.clear()
            streamer._pinned_blocks.clear()
            streamer._cache_budget_bytes = 0
            streamer._calibration_done = False
        for future in futures:
            future.cancel()
    for key, value in native_stats.items():
        if key != 'projections' and isinstance(value, int):
            native_stats[key] = 0


def run(contexts):
    timings = StageTimings()
    spec = json.loads(sys.stdin.readline())
    model, _, turbo = profile(spec)
    int8 = model == int8_config.MODEL
    native = int8 and spec.get('int8_compute', 'bf16') == 'native'
    native_stats = {'linear_calls': 0, 'projections': 0, 'backend': 'comfy-kitchen-triton-int8', 'bf16_matmul_fallbacks': 0}
    lite_check = False
    check_job(spec)
    if int8:
        int8_config.require_weights()
        try:
            event(compatibility=int8_config.validate_checkpoint())
        except int8_config.OriginalHeadersMissing:
            lite_check = True  # no BF16 shards: validate against the architecture after imports
    if turbo:
        import turbo_config
        turbo_config.require_adapter()
    local = Path(__file__).resolve().parent.parent / "models" / ("MOSS-Image-SFT" if int8 else model.split("/")[-1])
    if not (local / "model_index.json").exists():
        raise RuntimeError("Model not downloaded. Run download.ps1 first for " + model)
    # INT8 brings its own transformer file; only shared SFT components are required.
    for component in ("text_encoder", "vae") if int8 else ("text_encoder", "transformer", "vae"):
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
    if lite_check:
        event(compatibility=int8_config.validate_checkpoint(
            expected=int8_config.architecture_shapes(local / "transformer")))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run setup.ps1 and check your NVIDIA driver.")
    timings.synchronize = torch.cuda.synchronize
    if not int8:
        event(backend='weellm-bf16', weight_format='BF16',
              compute_dtype='BF16 base + FP32 adapter' if turbo else 'BF16 transformer', experimental=False)
    free, total = torch.cuda.mem_get_info()
    budget = min(float(os.getenv("MOSS_VRAM_GB", "2.5")), free / 2**30 - 0.5)
    if native:
        # Native blocks stay INT8 (~113 MB each); unused VRAM keeps blocks
        # resident instead of re-reading them from SSD on every step.
        budget = min(float(os.getenv("MOSS_INT8_VRAM_GB", "3.0")), free / 2**30 - 0.5)
    elif int8:
        budget = min(budget, 2.0)  # Headroom for expanded BF16 layers and ConvRot temporaries.
    if budget < 0.75:
        raise RuntimeError("Insufficient free VRAM. Close other GPU applications.")
    torch.cuda.reset_peak_memory_stats()
    event(stage="Loading MOSS weights from SSD")
    if int8:
        from int8_backend import transformer_weights
    with timings.stage("model_load"):
        if native:
            from native_int8 import native_transformer
            contexts.enter_context(native_transformer(int8_config.WEIGHTS, native_stats,
                                                      prefetch_depth=int(os.getenv("MOSS_INT8_PREFETCH", "3"))))
        with (transformer_weights(int8_config.WEIGHTS) if int8 and not native else nullcontext()):
            pipe = WeeTextToImagePipeline.from_pretrained(
                str(local), device="cuda", torch_dtype=torch.bfloat16,
                vram_budget_gb=budget, ram_budget_gb=float(os.getenv("MOSS_RAM_GB", "2")),
                cache_to_ram=False, vae_tile_size=int(os.getenv("MOSS_VAE_TILE", "256")))
    if native and spec.get("keep_warm"):
        # WeeLLM evicts the text encoder's resident tensors (incl. computed rotary
        # buffers) to meta before sampling and cannot reload them, so a warm
        # worker keeps it loaded (~1.9 GB RAM embeddings, ~0.1 GB VRAM) for new prompts.
        pipe._pipeline.transformer._weellm_te_evicted = True
    while True:
        serve(spec, timings, pipe, local, int8, native, turbo, budget, native_stats, torch)
        if not (native and spec.get("keep_warm")):
            return
        # Warm: keep the model loaded and wait for the next job of this profile.
        if hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()  # release cached pinned staging buffers while idle
        event(job_done=True)
        line = sys.stdin.readline()
        if not line:
            return
        next_spec = json.loads(line)
        if profile(next_spec) != profile(spec):
            raise ValueError("Warm worker received a different model profile")
        check_job(next_spec)
        spec = next_spec
        reset_for_next_job(pipe, native_stats)
        torch.cuda.reset_peak_memory_stats()
        timings = StageTimings()
        timings.synchronize = torch.cuda.synchronize


def serve(spec, timings, pipe, local, int8, native, turbo, budget, native_stats, torch):
    if turbo:
        import turbo_config
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
    if turbo or int8:
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
        if native:
            # No BF16 weight reconstruction or adapter cache. Measured 768x1024
            # peak: 1,104 MiB allocated incl. 651 MB resident + one 113 MB block,
            # i.e. ~340 MB activations for 3,155 tokens; keep ~1.5x margin.
            overhead = (spec["width"] // 16) * (spec["height"] // 16) * 160 * 1024
        BaseTransformerStreamer._estimated_vram_overhead_bytes = overhead
        BaseTransformerStreamer._estimated_ram_overhead_bytes = overhead
        if int8:
            from int8_backend import generate
            from stream_timings import collect
            with collect(os.getenv("MOSS_STREAM_TIMINGS") == "1") as stream_log:
                image = generate(pipe._pipeline, spec['prompt'], spec['width'], spec['height'],
                                 spec['seed'], progress, timings, encode_prompt)
            if stream_log is not None:
                event(stream_timings=stream_log.summary(
                    getattr(pipe._pipeline.transformer, '_weellm_streamer', None)))
            event(checkpoint_repo=int8_config.MODEL, checkpoint_revision=int8_config.REVISION,
                  weight_format='INT8 ConvRot', compute_dtype='INT8 GEMM + INT32 accumulation, BF16 activations' if native else 'BF16 streamed dequantization',
                  backend='weellm-triton-int8-flow8' if native else 'weellm-int8-flow8', experimental=True,
                  native_int8_stats=native_stats if native else None)
        else:
            from rapid_moss import generate
            image = generate(pipe._pipeline, turbo_config.DIRECTORY, spec["prompt"],
                             spec["width"], spec["height"], spec["seed"], progress,
                             timings=timings, chunk_tokens=chunk_tokens, cache_mb=cache_mb,
                             vram_budget_gb=budget, report=event, encode_prompt=encode_prompt)
        event(prompt_cache=prompt_cache.status if prompt_cache else
              ("private" if spec.get("private") else "disabled"))
        if turbo:
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

def main():
    with ExitStack() as contexts:
        run(contexts)


if __name__ == "__main__":
    main()
