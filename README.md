<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Windows][windows-shield]][windows-url]
[![Python][python-shield]][python-url]
[![NVIDIA CUDA][cuda-shield]][cuda-url]
[![MOSS-Image][model-shield]][model-url]
[![MIT License][license-shield]][license-url]
[![Stars][stars-shield]][stars-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="#about-the-project">
    <img src="docs/logo.svg" alt="MOSS Image Studio logo" width="96" height="96">
  </a>

  <h3 align="center">MOSS Image Studio</h3>

  <p align="center">
    <a href="https://hits.sh/github.com/sanketshinde3001/moss-image-studio/">
      <img alt="Repository views" src="https://hits.sh/github.com/sanketshinde3001/moss-image-studio.svg?style=for-the-badge&label=Repository%20views&color=f0a14a">
    </a>
  </p>

  <p align="center">
    Generate images from text prompts with MOSS-Image on your Windows PC with an NVIDIA RTX GPU.
    <br />
    A local browser studio with CUDA and WeeLLM weight streaming for low-VRAM systems.
    <br />
    <a href="#getting-started"><strong>Get started »</strong></a>
    <br />
    <br />
    <a href="#usage">Usage</a>
    &middot;
    <a href="#turbo-4-step-setup">Turbo 4-step</a>
    &middot;
    <a href="#benchmark">Benchmark</a>
    &middot;
    <a href="#how-it-works">How it works</a>
    &middot;
    <a href="#limitations">Limitations</a>
    &middot;
    <a href="https://github.com/sanketshinde3001/moss-image-studio/issues">Report a bug</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About the project</a>
      <ul>
        <li><a href="#features">Features</a></li>
        <li><a href="#built-with">Built with</a></li>
      </ul>
    </li>
    <li><a href="#getting-started">Getting started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#turbo-4-step-setup">Turbo 4-step setup</a></li>
        <li><a href="#troubleshooting">Troubleshooting</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#how-it-works">How it works</a>
      <ul>
        <li><a href="#benchmarks">Benchmark</a></li>
        <li><a href="#architecture">Architecture</a></li>
        <li><a href="#privacy">Privacy</a></li>
      </ul>
    </li>
    <li><a href="#limitations">Limitations</a></li>
    <li><a href="#frequently-asked-questions">Frequently asked questions</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#star-history">Star history</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About the project

MOSS Image Studio is a local, open-source text-to-image app for Windows and NVIDIA RTX GPUs. It gives you a browser interface for the official [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) models and uses [WeeLLM](https://github.com/Jit-Roy/WeeLLM) to stream model weights when GPU memory is limited.

[![MOSS Image Studio generation interface][product-screenshot]](#usage)

**Turbo 4-step is the default.** The latest cache-hit 768 × 1024 portrait test completed on an **RTX 3050 Laptop GPU with 4 GB VRAM in 73.77 seconds**, including loading, with a sampled system-wide peak of **1,967 MiB VRAM**. See the [benchmark](#benchmark) for settings and the actual generated image. This is a measured result from one laptop, not a promise of typical speed or identical quality to the normal model.

### Features

**Text-to-image generation**
* Choose the official MOSS-Image-SFT or MOSS-Image-Base model.
* Set prompts, negative prompts, seeds, image variations, styles, and aspect ratios.
* Select Draft, Balanced, or Quality presets, or customize output size and steps.
* Default **Turbo 4-step** generation with the Rapid-MOSS SFT adapter and WeeLLM streaming; one RTX 3050 portrait verified.

**Local image workflow**
* Queue prompts and follow sampling progress; pause, reorder, or cancel queued jobs.
* Browse, search, compare, favorite, and download generated images.
* Save reusable prompt presets and download selected results as a ZIP.
* Use Private mode for temporary results, with Keep or Discard controls.

The current app generates images from text. Image editing, masks, and image-to-image generation are not available.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built with

* [![Python][python-wide]][python-url]
* [![PyTorch][pytorch-shield]][pytorch-url]
* [![NVIDIA CUDA][cuda-wide]][cuda-url]
* [![WeeLLM][weellm-shield]][weellm-url]
* [![Diffusers][diffusers-shield]][diffusers-url]

The UI is a single HTML file and needs no frontend build step. The local Python server listens on `127.0.0.1` by default.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting started

### Prerequisites

* Windows 10 or 11
* Python 3.13 and Git
* NVIDIA GPU with a current driver and CUDA support
* SSD storage for the Python environment, model files, cache, and generated images
* Roughly 28.5 GB for SFT plus the Turbo adapter; leave additional space for Python packages, caches, and outputs

### Installation

1. Clone this repository and enter the project folder:
   ```powershell
   git clone https://github.com/sanketshinde3001/moss-image-studio.git
   cd moss-image-studio
   ```
2. Set up the Python environment and CUDA dependencies:
   ```powershell
   ./setup.ps1
   ```
3. Choose what to download. Every option resumes and verifies SHA256 hashes:

   | Command | Downloads | Size | Studio default |
   |---|---|---:|---|
   | `./download.ps1 -Model Lite` | Shared SFT text encoder, tokenizer and VAE + native INT8 Turbo 8-step transformer (no BF16 transformer) | ~22.4 GB | INT8 Turbo 8-step |
   | `./download.ps1` | Full BF16 MOSS-Image-SFT + Rapid-MOSS Turbo 4-step adapter | ~28.5 GB | BF16 Turbo 4-step |
   | `./download.ps1` then `./download.ps1 -Model INT8` | Both of the above | ~33.9 GB | INT8 Turbo 8-step (BF16 still selectable) |

   `Lite` suits 4 GB laptops and small disks. Choose the full download if you want the original BF16 model; you can add or remove either one later.
4. Start the studio:
   ```powershell
   ./studio.ps1
   ```
5. Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

To download MOSS-Image-Base as well, run `./download.ps1 -Model Base` and select it in the app. Downloads can be resumed by running the download command again. For a slower connection, use:

```powershell
./.venv/Scripts/python.exe download.py SFT --workers 8 --parallel-files 2
```

The model weights are not stored in this GitHub repository. They are downloaded from the official [MOSS-Image-SFT model card](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) and retain their own license terms.

### Turbo 4-step setup

The default `./download.ps1` command installs both SFT and the Rapid-MOSS adapter. If you already have SFT, install only the adapter (about **1.23 GB**) with:

```powershell
./download.ps1 -Model Turbo
```

The studio selects **MOSS-Image SFT** and **Turbo 4-step** by default when the adapter is installed. Draft, Balanced, and Quality remain available for normal sampling. The adapter download resumes and checks SHA256 hashes against a pinned upstream revision. An incomplete download leaves Turbo disabled and selects Draft instead. There are no automatic downloads during generation. API jobs also default to Turbo for SFT when installed; send `"turbo": false` to use normal sampling.

Turbo runs the adapter's dedicated four-step data-space sampler, with no CFG, negative prompt, or ordinary Euler schedule. The original BF16 model still streams through WeeLLM. FP32 LoRA weights remain unmerged on disk and transfer one projection at a time; FP32 activation calculations are chunked to limit temporary memory. Time embeddings and modulation calculations use FP32, including after resident weights reload. Text-encoder eviction and tiled VAE decoding remain enabled. The default VRAM budget stays at 2.5 GB.

Adapter processing now uses chunks of up to 1,024 tokens, reduced automatically for large projections to target at most 64 MiB of FP32 activation temporaries. A bounded 128 MiB first-fit cache reuses a subset of FP32 adapter weights across the four steps. Cache admission also checks free VRAM and the configured budget; the cache is cleared before VAE decoding. Unchanged FP32 time-path weights are reused until WeeLLM reloads them. These changes keep the model, precision, sampler, four steps, and resolution unchanged, but GPU rounding can still affect image details.

Generated image details and JSON sidecars now include synchronized wall-clock `stage_seconds` for startup/imports, model loading, prompt encoding, adapter setup, sampling, cleanup, image decoding, and PNG saving. `adapter_stats` records cache usage and chunk counts. Optional tuning: `MOSS_TURBO_CHUNK` (128–2,048; default 1,024) and `MOSS_TURBO_CACHE_MB` (0–256; default 128, use 0 to disable). These are memory/performance settings, not quality presets.

**Repeated-prompt acceleration:** Turbo now caches the exact prompt embeddings and attention mask locally, retaining their original precision. Reusing the same prompt with a different seed or image size can skip text-encoder inference without changing the encoding or sampler. New prompts still require encoding. Cache identity tracks the encoder/tokenizer files, encoding implementation and package versions. The cache is bounded to 32 entries / 128 MiB in `studio/prompt_cache/`; delete that folder to clear it. Embeddings can contain information about prompts: private mode bypasses all cache reads and writes. Image details report `prompt_cache` alongside stage timings.

**Optional experimental INT8 model:** the model switcher also offers [INT8 Turbo 8-step](https://huggingface.co/toxicdog/MOSS-Image-SFT-Turbo-INT8-ConvRot). Install its separate 5.41 GB transformer with `./download.ps1 -Model INT8`; this SHA256-verifies the pinned checkpoint and preserves all existing models. It shares the original SFT text encoder, tokenizer, FP32 VAE and prompt cache. When the INT8 checkpoint is installed the studio and API default to it (native compute when available); installs without it keep **BF16 SFT + Turbo 4-step** as the default. BF16 models stay selectable whenever their weights are present; a `Lite` install shows them as unavailable.

This experimental profile follows the checkpoint author's **eight-step Euler/simple, shift 5, CFG 1** settings with no negative prompt. It does not apply the incompatible Rapid-MOSS four-step adapter. The compute selector offers **Native INT8 · Triton** and **Compatibility · BF16 decode**. Native mode streams INT8 block weights with FP32 scales, dynamically quantizes rotated activations, and executes INT8 tensor-core GEMMs with INT32 accumulation and BF16 outputs. Compatibility mode reconstructs BF16 layers before ordinary compute. Smaller storage does not guarantee faster generation or identical quality. Model details record the checkpoint revision, storage/compute formats, compatibility check, per-stage times and PyTorch VRAM peaks. Its timing calibration is separate from BF16 Turbo. Private requests bypass prompt caching for either model.

**Verified on RTX 3050 Laptop 4 GB:** one 768 × 1024 portrait, seed 42, CFG 1 and cached prompt. All 463 required tensor shapes and 252 quantization recipes passed compatibility checks before generation.

| Profile | Steps | Actual total time | Sampled peak GPU VRAM | Peak GPU temperature |
|---|---:|---:|---:|---:|
| BF16 Turbo (saved baseline; default on full BF16 installs) | 4 | 73.77 s | 1,967 MiB | 87°C |
| INT8 compatibility / BF16 decode | 8 | 79.35 s | 1,645 MiB | 88°C |
| Native INT8 / Triton, first version | 8 | 62.15 s | 1,489 MiB | 89°C |
| **Native INT8 / Triton, streaming + GEMM tuning (default when installed)** | **8** | **46.91 s** | **1,961 MiB** | **87°C** |

The earlier INT8 compatibility run used **16.4% less peak VRAM** but took **7.6% longer** than BF16 Turbo. Its prompt-cache hit took 0.0198 s; sampling took 54.66 s and VAE decoding 6.42 s. The actual portrait was visually reviewed and is clear, with different framing, hairstyle and background. These profiles use different checkpoints, samplers, step counts and VRAM budgets; this is not an isolated quantization benchmark or a guarantee of unchanged quality, lower heating or repeatable speed. Exactly one compatibility-mode INT8 image was generated; the saved BF16 reference was reused. BF16 Turbo remains the default on installs without the INT8 checkpoint.

**Native INT8 validation (19 October):** one additional native-mode image, same checkpoint, prompt, seed, dimensions, eight steps and cache hit: **62.15 seconds including loading**, **1,489 MiB sampled system-wide peak GPU VRAM** and **89°C peak GPU temperature**. Sampling took 42.84 s, prompt-cache lookup 0.0038 s and image decoding 5.29 s. All **2,016 block projection calls** used the explicit Comfy Kitchen Triton INT8 kernel, with zero BF16 GEMM fallbacks. Compiled PTX was checked for INT8 tensor-core MMA instructions. The portrait was visually reviewed and is similar in clarity and composition to the compatibility image, with different pixels; quality equivalence across prompts is unproven.

Native mode was **21.7% faster** and used **9.5% less peak VRAM** than the saved 79.35-second compatibility run. This is a single-run comparison, not a typical-speed guarantee or a cooling improvement. The **62.15-second timer excludes 11.25 seconds of separate first-time kernel preparation/tuning** using three projection checks; no warmup image was generated. Tuning results are cached across disposable workers. First use or a different size/device may require compilation or tuning again.

**Native INT8 streaming update (19 October):** profiling showed sampling was waiting on weight streaming, not INT8 math. WeeLLM had disabled transfer overlap, re-read every 113 MB block from SSD with one thread, and copied from zero-filled pageable buffers, which delayed GPU kernel launches. Native INT8 now has its own VRAM budget (`MOSS_INT8_VRAM_GB`, default 3.0) so blocks double-buffer, reads three blocks ahead (`MOSS_INT8_PREFETCH`, default 3), and reads raw weights into pinned memory for asynchronous copies (`MOSS_INT8_PINNED=0` disables). INT8 GEMMs use added `block_k=128` tile configurations, and q/k/v share one quantized activation. INT32 accumulation is exact, so these changes do not alter results: the same portrait, seed and prompt produced a **pixel-identical** image (0 differing pixels) in **46.91 s**, with sampling 26.82 s, image decoding 5.32 s, **1,961 MiB** sampled peak VRAM (higher, because more weights stay on the GPU) and **87°C** peak. This is one run with a cached prompt and already-tuned kernels, not a repeatable-speed or cooling guarantee. `MOSS_STREAM_TIMINGS=1` records per-block wait, transfer and compute times in image details.

Install the optional native kernel package after normal setup:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps -r requirements-int8-native.txt
```

Then select the experimental INT8 model and **Native INT8 · Triton**. The tested laptop's driver 572.16 cannot run the package's CUDA 13 extension; this integration explicitly uses its Triton INT8 implementation with the existing Torch CUDA 12.8 environment. It does not update the driver or PyTorch. Native mode fails with an error if unsupported, rather than silently claiming native acceleration through BF16. BF16 decode stays available, and **BF16 SFT + Turbo 4-step remains the studio default**.

<details>
<summary>View the actual native INT8 portrait</summary>
<p align="center"><img src="docs/int8-native-portrait.png" alt="MOSS native INT8 Triton portrait on RTX 3050 4 GB, eight steps in 62.15 seconds and 1489 MiB peak VRAM after kernel preparation" width="384"></p>
</details>

<details>
<summary>View the actual experimental INT8 portrait</summary>
<p align="center"><img src="docs/int8-portrait.png" alt="Experimental MOSS INT8 Turbo portrait generated on RTX 3050 4 GB in 79.35 seconds using eight steps and 1645 MiB peak GPU VRAM" width="384"></p>
</details>


**Verified cache-hit run:** the same 768 × 1024 portrait prompt, seed 42 and four steps completed in **73.77 seconds including loading**, with **1,967 MiB system-wide sampled peak VRAM** (0.5-second polling) and **87°C peak GPU temperature**. Prompt lookup/copy took **0.0139 seconds**, compared with 18.86 seconds encoding previously. Overall time was 11.3% shorter than the prior 83.19-second run; startup and sampling were slower in this run, so the earlier ~64-second estimate was not achieved. Exactly one image was generated after a separate encoding-only cache warmup; preparation time is excluded from this cache-hit measurement. The output was pixel-identical to the previous optimized portrait. This is one measured comparison, not a typical-speed guarantee. Fresh prompts still require encoding.

**Previous uncached run:** 768 × 1024, four steps, seed 42, CFG disabled: **83.19 seconds including loading**, **1,863 MiB sampled peak VRAM**, and **87°C peak GPU temperature**. The completed portrait was visually reviewed. This single result does not establish performance or quality for other prompts, sizes, or devices. The adapter's author documents quality trade-offs; Turbo does not produce the same image as normal sampling.

The app has **no temperature cutoff or automatic thermal cooldown**. Optional fixed queue delays are user-selected and default to Off. GPU and firmware protections are managed by the device, not this app.

Adapter source and instructions: [Rapid-MOSS 4-step adapter](https://huggingface.co/darask0/MOSS-Image_4-step_adapter_lora). This mode uses SFT; it does not support the Base model or a quantized DiT.

### Troubleshooting

**CUDA is unavailable.** Check the NVIDIA driver with `nvidia-smi`, close other GPU-heavy applications, and rerun `./setup.ps1`.

**The selected model is missing.** Run `./download.ps1 -Model Lite` (INT8), `./download.ps1` (full BF16) or `./download.ps1 -Model Base`, then select the downloaded model in the studio.

**Generation runs out of memory.** Close other GPU applications, choose a smaller output size, and lower the `MOSS_VRAM_GB` budget if Windows needs more memory.

**Model loading is slow.** The streaming approach reads model weights from disk. An SSD is recommended, and the first run includes model loading.

**Python imports fail.** Rerun `./setup.ps1` to recreate the project environment and install the pinned dependencies.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE -->
## Usage

1. Start the studio with `./studio.ps1`.
2. Select an installed MOSS model and write a text prompt.
3. Choose a size preset, aspect ratio, and generation settings.
4. Submit the job and follow its progress in the queue.
5. Open the finished image in the gallery to compare, favorite, save, or download it.

Turbo is selected by default with a 768 px size preset and four steps. Draft uses normal sampling at 512 px and 12 steps, Balanced uses 768 px and 30 steps, and Quality uses 1024 px and 50 steps. You can change the aspect ratio and output size. Different samplers and step counts can change the result.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- HOW IT WORKS -->
## How it works

MOSS Image Studio runs the interface and generation locally. WeeLLM streams model weights from SSD instead of requiring the full checkpoint to reside in GPU memory. This helps fit large models into a low VRAM budget, while adding model loading and disk transfer time.

### Benchmark

The portrait runs used an RTX 3050 Laptop GPU with 4 GB VRAM, MOSS-Image-SFT, the same prompt, seed 42, and 768 × 1024 output. The normal run used 30 steps with CFG 3; Turbo used four CFG-free steps with the Rapid-MOSS adapter.

| Task | Settings | Elapsed time | Sampled system-wide peak VRAM |
|---|---|---:|---:|
| Portrait generation | 768 × 1024, 30 steps | 15 min 39 sec, including model loading | 1,823 MiB (about 1.78 GiB) |
| Earlier Turbo implementation | 768 × 1024, four CFG-free steps | 89.25 seconds, including model loading | 1,697 MiB (about 1.66 GiB) |
| **Turbo with chunk/cache tuning (default)** | **Same prompt, seed, size, precision, and four steps** | **83.19 seconds, including loading** | **1,863 MiB (about 1.82 GiB)** |
| **Turbo with prompt cache hit** | **Same prompt, seed, size, precision, and four steps; cache prepared separately** | **73.77 seconds, including loading** | **1,967 MiB (about 1.92 GiB)** |

The chunk/cache check generated exactly one image. It was 6.8% shorter than the earlier Turbo run, with comparable visual clarity, but the output was not pixel-identical and some background details changed. Peak GPU temperature was 87°C. This is a single comparison, not proof of a repeatable speed gain or lower temperature.

Measured stage times for that run:

| Stage | Time |
|---|---:|
| Startup/imports | 5.97 s |
| Model loading | 6.11 s |
| Prompt encoding | 18.86 s |
| Adapter setup | 0.16 s |
| Sampling | 45.11 s |
| Adapter cleanup | 0.21 s |
| Image decoding | 5.05 s |
| PNG saving | 0.10 s |
| **Overall job time** | **83.19 s** |

Worker total was 81.75 s; overall job time also includes process startup and coordination. The adapter cache recorded 87 hits and one time-path load.

These are individual measurements, not repeated controlled benchmarks. Speed and temperature vary with the driver, background GPU use, available RAM, SSD cache, and cooling. System-wide VRAM and temperature were sampled every two seconds. The completed Turbo run peaked at 86°C; PyTorch recorded 1,433 MiB allocated and 1,590 MiB reserved. No landscape or product benchmark was completed, and the two modes produce different images.

<p align="center">
  <img src="docs/turbo-portrait.png" alt="Portrait generated by optimized MOSS Image Studio Turbo in 83.19 seconds on an RTX 3050 Laptop GPU with 4 GB VRAM" width="384">
</p>

<p align="center"><strong>Actual optimized Turbo output · 768 × 1024 · 4 steps · 83.19 seconds · peak VRAM 1,863 MiB</strong></p>

### Architecture

```text
Browser interface (studio/index.html)
        │  local HTTP requests
        ▼
Python web server and serial job queue (studio/server.py)
        │  one isolated worker process per generation job
        ▼
MOSS generation worker (studio/moss_worker.py)
        │  CUDA, Diffusers, WeeLLM weight streaming
        ▼
NVIDIA GPU + local model files on SSD
```

* `download.py` downloads model files and verifies their SHA256 hashes.
* `studio/server.py` handles the local web interface, jobs, gallery, settings, and downloads.
* `studio/moss_worker.py` runs an isolated model generation job.
* `studio/engine.py` configures MOSS and the WeeLLM streaming pipeline.

### Memory and performance

The default settings reserve up to 2.5 GB for VRAM streaming and 2 GB for the RAM cache, with disk-backed model weights and 256 px VAE tiles. Actual process RAM can exceed the cache budget. You can set the budgets before starting the app:

```powershell
$env:MOSS_VRAM_GB = '2.5'
$env:MOSS_RAM_GB = '2'
./studio.ps1
```

A 27 GB model download fitting on disk does not mean a generation will be fast. Streaming lowers memory requirements by moving weights between storage, RAM, and GPU. Close other GPU-heavy apps and let a laptop cool between long runs.

### Privacy

* The web server binds to `127.0.0.1` by default.
* Model downloads connect to Hugging Face; generation runs on your computer after the weights are downloaded.
* Private mode keeps results temporary unless you choose Keep. Private jobs do not write per-job logs or gallery sidecar files.
* Normal generated images and project files remain on local storage. The app does not include a cloud generation service.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LIMITATIONS -->
## Limitations

* NVIDIA CUDA on Windows is the supported setup; other GPU platforms are not covered by these setup scripts.
* The current app supports text-to-image only. Image editing and inpainting are not implemented.
* Model weight streaming reduces the VRAM requirement at the cost of additional loading and transfer time.
* The built-in output size is limited to 1024 pixels per side.
* The recorded RTX 3050 result is one run. It is not a benchmark suite or a speed guarantee.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- FAQ -->
## Frequently asked questions

### What is MOSS Image Studio?
A local Windows application for generating images from text prompts with MOSS-Image models on NVIDIA CUDA GPUs.

### Can MOSS Image Studio run on an RTX 3050 with 4 GB VRAM?
One 768 × 1024 image was successfully generated on an RTX 3050 Laptop GPU with 4 GB VRAM. It took 15 minutes 39 seconds, including model loading, and reached a sampled system-wide peak of 1,823 MiB. Results vary by computer and workload.

### Does the app support image editing or inpainting?
No. The current app supports text-to-image generation only.

### How large is the MOSS-Image-SFT download?
About 27 GB. Leave extra SSD space for the Python environment, model caches, and generated images.

### Are the model weights included in this repository?
No. The download script fetches them from Hugging Face. The weights have separate license terms; see the official model card before use.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] Windows and NVIDIA CUDA setup
- [x] MOSS-Image-SFT and MOSS-Image-Base model selection
- [x] Local text-to-image studio with gallery and generation queue
- [x] Low VRAM generation through WeeLLM model streaming
- [x] Optional Rapid-MOSS four-step sampler with disk-backed FP32 adapters
- [x] Complete and visually review one Turbo portrait on the RTX 3050 4 GB, recording time, peak VRAM, and temperature
- [ ] Evaluate speed and thermal improvements against saved image quality
- [ ] Record repeatable performance results on supported NVIDIA hardware

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Issues, bug reports, and pull requests are welcome. When reporting a generation problem, include your GPU model, VRAM, image size, step count, selected MOSS model, and the relevant error log. Do not include private prompts or images unless you intend to share them publicly.

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-change`).
3. Commit your changes (`git commit -m "Describe your change"`).
4. Push your branch and open a pull request.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

MOSS Image Studio code and scripts are available under the MIT License. See [`LICENSE`](LICENSE). Model weights and upstream libraries have separate licenses; consult their official project pages before use.

The Rapid-MOSS integration in `studio/rapid_moss.py` is adapted from Apache-2.0 code. See [third-party notices](THIRD_PARTY_NOTICES.md) and the [Apache-2.0 license](LICENSES/Apache-2.0.txt).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Project maintainer: [Sanket Shinde](https://github.com/sanketshinde3001)

Questions, bug reports, and feature requests: [open an issue](https://github.com/sanketshinde3001/moss-image-studio/issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- STAR HISTORY -->
## Star history

<a href="https://star-history.com/#sanketshinde3001/moss-image-studio&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=sanketshinde3001/moss-image-studio&type=Date&theme=dark" />
    <img alt="MOSS Image Studio GitHub star history" src="https://api.star-history.com/svg?repos=sanketshinde3001/moss-image-studio&type=Date" />
  </picture>
</a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[windows-shield]: https://img.shields.io/badge/Windows-10%20%2F%2011-0078D6?style=for-the-badge&logo=windows&logoColor=white
[windows-url]: https://www.microsoft.com/windows
[python-shield]: https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white
[python-wide]: https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/
[cuda-shield]: https://img.shields.io/badge/NVIDIA-CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white
[cuda-wide]: https://img.shields.io/badge/NVIDIA%20CUDA-RTX%20GPU-76B900?style=for-the-badge&logo=nvidia&logoColor=white
[cuda-url]: https://developer.nvidia.com/cuda-zone
[model-shield]: https://img.shields.io/badge/Model-MOSS--Image-8A63D2?style=for-the-badge
[model-url]: https://github.com/OpenMOSS/MOSS-Image
[pytorch-shield]: https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[pytorch-url]: https://pytorch.org/
[weellm-shield]: https://img.shields.io/badge/WeeLLM-streaming-333333?style=for-the-badge
[weellm-url]: https://github.com/Jit-Roy/WeeLLM
[diffusers-shield]: https://img.shields.io/badge/Hugging%20Face-Diffusers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black
[diffusers-url]: https://github.com/huggingface/diffusers
[product-screenshot]: docs/moss-studio.png
[license-shield]: https://img.shields.io/badge/License-MIT-2ea44f?style=for-the-badge
[license-url]: LICENSE
[stars-shield]: https://img.shields.io/github/stars/sanketshinde3001/moss-image-studio?style=for-the-badge
[stars-url]: https://github.com/sanketshinde3001/moss-image-studio/stargazers

