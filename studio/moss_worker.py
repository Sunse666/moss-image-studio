"""MOSS inference in a disposable process; no GPU dependencies in the web server."""
import json
import os
import sys
import time
from pathlib import Path

def event(**values):
    print("MOSS_EVENT " + json.dumps(values), flush=True)

def main():
    spec = json.loads(sys.stdin.readline())
    model = spec["model"]
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
    import torch
    from weellm import WeeTextToImagePipeline
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run setup.ps1 and check your NVIDIA driver.")
    free, total = torch.cuda.mem_get_info()
    budget = min(float(os.getenv("MOSS_VRAM_GB", "2.5")), free / 2**30 - 0.5)
    if budget < 0.75:
        raise RuntimeError("Insufficient free VRAM. Close other GPU applications.")
    torch.cuda.reset_peak_memory_stats()
    event(stage="Loading MOSS weights from SSD")
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
    image = pipe.generate(prompt=spec["prompt"], negative_prompt=spec["negative"],
                          width=spec["width"], height=spec["height"],
                          num_inference_steps=spec["steps"], guidance_scale=spec["cfg"], seed=spec["seed"])
    temporary = Path(spec["output"] + ".part")
    image.save(temporary, format="PNG")
    temporary.replace(spec["output"])
    event(gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
          peak_allocated_mb=round(torch.cuda.max_memory_allocated()/2**20),
          peak_reserved_mb=round(torch.cuda.max_memory_reserved()/2**20))

if __name__ == "__main__":
    main()
