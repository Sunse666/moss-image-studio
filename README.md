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

**Turbo 4-step is the default.** The latest 768 × 1024 portrait test completed on an **RTX 3050 Laptop GPU with 4 GB VRAM in 83.19 seconds**, including loading, with a sampled system-wide peak of **1,863 MiB VRAM**. See the [benchmark](#benchmark) for settings and the actual generated image. This is a measured result from one laptop, not a promise of typical speed or identical quality to the normal model.

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
3. Download MOSS-Image-SFT and the Turbo adapter (roughly 28.5 GB combined):
   ```powershell
   ./download.ps1
   ```
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

**Latest verified run:** 768 × 1024, four steps, seed 42, CFG disabled: **83.19 seconds including loading**, **1,863 MiB sampled peak VRAM**, and **87°C peak GPU temperature**. The completed portrait was visually reviewed. This single result does not establish performance or quality for other prompts, sizes, or devices. The adapter's author documents quality trade-offs; Turbo does not produce the same image as normal sampling.

The app has **no temperature cutoff or automatic thermal cooldown**. Optional fixed queue delays are user-selected and default to Off. GPU and firmware protections are managed by the device, not this app.

Adapter source and instructions: [Rapid-MOSS 4-step adapter](https://huggingface.co/darask0/MOSS-Image_4-step_adapter_lora). This mode uses SFT; it does not support the Base model or a quantized DiT.

### Troubleshooting

**CUDA is unavailable.** Check the NVIDIA driver with `nvidia-smi`, close other GPU-heavy applications, and rerun `./setup.ps1`.

**The selected model is missing.** Run `./download.ps1` or `./download.ps1 -Model Base`, then select the downloaded model in the studio.

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

