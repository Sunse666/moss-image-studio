#!/usr/bin/env python3
"""Moss Studio: a local MOSS-Image studio using WeeLLM on NVIDIA CUDA.

Standard library only. Everything it writes stays inside the project folder:
  outputs/         generated, edited and upscaled images, each with a .json settings sidecar
  inputs/          uploaded images and masks (converted to PNG)
  logs/            one log per job
  studio/tmp/      resized references, previews, and private-mode files (wiped on start and exit)
  studio/*.json    calibration, settings, saved presets and favorites
"""
import base64
import io
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

import engine
from engine import ENGINE, Cancelled

STUDIO = engine.STUDIO
ROOT = engine.ROOT
OUTPUTS = os.path.join(ROOT, "outputs")
INPUTS = os.path.join(ROOT, "inputs")
LOGS = os.path.join(ROOT, "logs")
TMP = os.path.join(STUDIO, "tmp")
PRIVATE = os.path.join(TMP, "private")
CALIB = os.path.join(STUDIO, "calibration.json")
SETTINGS = os.path.join(STUDIO, "settings.json")
LIBRARY = os.path.join(STUDIO, "library.json")
for d in (OUTPUTS, INPUTS, TMP, LOGS):
    os.makedirs(d, exist_ok=True)

SAMPLERS = ["euler", "euler_a", "heun", "dpm++2m", "dpm++2m_sde", "res_multistep", "ipndm", "lcm", "ddim_trailing"]
SCHEDULERS = ["default", "simple", "discrete", "karras", "exponential", "sgm_uniform", "beta", "smoothstep", "kl_optimal"]
GEN_MODES = ("gen",)


# ---------------------------------------------------------------- small helpers

def read_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def wipe_private():
    shutil.rmtree(PRIVATE, ignore_errors=True)
    os.makedirs(PRIVATE)


def wipe_tmp():
    """Resized refs and previews from interrupted jobs, plus any private files left by a crash."""
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(PRIVATE)


def image_size(path):
    from PIL import Image
    with Image.open(path) as image:
        return image.size


def to_png(src, dst, max_side=None, exact=None):
    from PIL import Image, ImageOps
    try:
        with Image.open(src) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            if exact:
                image = image.resize(exact, Image.Resampling.LANCZOS)
            elif max_side:
                image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            image.save(dst, "PNG")
        return True
    except (OSError, ValueError):
        return False


def data_url(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def snap16(v):
    return max(256, int(round(v / 16.0)) * 16)


def safe_name(name):
    base = os.path.basename(name or "")
    if not base or base != name or base.startswith("."):
        raise ValueError("bad name")
    return base


def ref_path(name):
    """A reference can be an upload (inputs/), a private file, or an earlier result (outputs/)."""
    for d in (INPUTS, PRIVATE):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(OUTPUTS, name)


def is_private_file(name):
    return bool(name) and os.path.exists(os.path.join(PRIVATE, name))


def write_sidecar(image_path, meta):
    write_json(image_path.rsplit(".", 1)[0] + ".json", meta)


def sidecar_of(name):
    return read_json(os.path.join(OUTPUTS, name.rsplit(".", 1)[0] + ".json"), {})


# ---------------------------------------------------------------- estimates

DEFAULT_CALIB = {
    # Initial estimates only; calibrated after completed runs.
    "gen": 79.0, "edit": 80.0, "gen_fast": 24.0, "edit_fast": 40.0, "gen_turbo": 26.0, "edit_turbo": 22.0,
    "upscale": 355.0,  # seconds per input MP for ESRGAN x4
}


def load_calib():
    c = dict(DEFAULT_CALIB)
    c.update(read_json(CALIB, {}))
    return c


def work_mp(job):
    """Output megapixels."""
    return job["width"] * job["height"] / 1048576.0


EDIT_OVERHEAD = 20  # seconds: reading the reference on the first step of an edit


def calib_key(job):
    if job.get("model") == engine.int8_config.MODEL:
        return job["mode"] + ("_int8native8" if job.get('int8_compute') == 'native' else "_int8flow8")
    speed = "_rapid4" if job.get("turbo") else "_fast" if job.get("fast") else ""
    return job["mode"] + speed


def estimate_seconds(job):
    c = load_calib()
    if job["mode"] == "upscale":
        return c["upscale"] * job["src_mp"] + 5
    if job["mode"] == "assist":
        return 25
    extra = EDIT_OVERHEAD if job["mode"] == "edit" else 0
    return c.get(calib_key(job), 80.0) * (work_mp(job) ** 1.2) * job["steps"] + 30 + extra


def learn_speed(job):
    if job.get("private"):
        return  # private mode writes nothing, not even speed data
    c = read_json(CALIB, {})
    spent = job["finished"] - job["started"]
    if job["mode"] == "upscale":
        c["upscale"] = round(spent / max(0.05, job["src_mp"]), 1)
    elif job["mode"] in GEN_MODES:
        extra = EDIT_OVERHEAD if job["mode"] == "edit" else 0
        c[calib_key(job)] = round(max(1.0, spent - 30 - extra) / (job["steps"] * work_mp(job) ** 1.2), 2)
    write_json(CALIB, c)


# ---------------------------------------------------------------- settings and library

SETTINGS_DEFAULT = {"idle_minutes": 10, "cooldown": "off"}
COOLDOWN_CHOICES = ("off", "30", "60", "90")


def load_settings():
    s = dict(SETTINGS_DEFAULT)
    s.update(read_json(SETTINGS, {}))
    if s.get("cooldown") not in COOLDOWN_CHOICES:
        s["cooldown"] = "off"  # Migrate legacy automatic cooldown settings.
    return s


def apply_settings(s):
    ENGINE.idle_seconds = max(1, int(s.get("idle_minutes", 10))) * 60


def load_library():
    lib = read_json(LIBRARY, {})
    return {"presets": lib.get("presets", {}), "favorites": lib.get("favorites", [])}


# ---------------------------------------------------------------- job queue

JOBS = {}      # id -> job dict
ORDER = []     # ids in submission order (queued ones can be reordered)
LOCK = threading.Lock()
WAKE = threading.Event()
QUEUE = {"paused": False}
# Private mode lives on the server so a page reload or a second tab can't silently turn it off.
PRIVATE_MODE = {"on": False}
# Optional user-selected fixed delay between queued jobs; off by default.
COOL = {"until": 0.0, "reason": ""}


def plan_cooldown(job):
    mode = load_settings().get("cooldown", "off")
    if mode == "off":
        return
    COOL.update(until=time.time() + int(mode), reason="User-selected queue delay")

PUBLIC_KEYS = ("id", "mode", "state", "prompt", "width", "height", "steps", "seed", "step", "total",
               "sec_per_it", "stage", "created", "started", "finished", "eta", "estimate", "output",
               "error", "refs", "has_preview", "private", "turbo", "task", "source", "result", "engine", "model")


def public(job):
    return {k: job.get(k) for k in PUBLIC_KEYS}


def drop_jobs(pred):
    with LOCK:
        for i in list(ORDER):
            if pred(JOBS[i]):
                ORDER.remove(i)
                JOBS.pop(i)


def open_log(job, name):
    """Per-job log. Private jobs get none. The output name is written first so delete can find it."""
    if job.get("private"):
        return None
    log = open(os.path.join(LOGS, "studio-%s-%s.log" % (time.strftime("%Y%m%d-%H%M%S"), job["id"])), "wb")
    log.write(("output: %s\nmode: %s  size: %dx%d  seed: %s  engine: %s\n\n" % (
        name, job["mode"], job["width"], job["height"], job.get("seed"), job.get("engine"))).encode())
    return log


def run_image_job(job):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    private = job["private"]
    name = "%s_" % job["id"] + "%s_%s_%dx%d_s%d.png" % (stamp, job["mode"], job["width"], job["height"], job["seed"])
    out_path = os.path.join(PRIVATE if private else OUTPUTS, name)
    job.update(state="running", started=time.time(), step=0, engine="cold", stage="Loading MOSS")
    log = open_log(job, name)
    try:
        ENGINE.generate(job, out_path, log)
    except Exception:
        for partial in (out_path, out_path + ".part"):
            if os.path.exists(partial):
                os.remove(partial)
        raise
    finally:
        if log:
            log.close()
    if job["state"] == "cancelled":
        if os.path.exists(out_path):
            os.remove(out_path)
        raise Cancelled()
    finish(job, name, {k: job.get(k) for k in (
        "mode", "prompt", "negative", "width", "height", "steps", "cfg", "seed", "model",
        "turbo", "turbo_adapter", "turbo_revision", "sampler", "stage_seconds", "adapter_stats", "prompt_cache",
        "checkpoint_repo", "checkpoint_revision", "weight_format", "compute_dtype", "backend", "experimental",
        "scheduler", "flow_shift",
        "compatibility",
        "int8_compute", "native_int8_stats", "stream_timings",
        "gpu", "torch_version", "peak_allocated_mb", "peak_reserved_mb")})


def finish(job, name, meta):
    job["finished"] = time.time()
    meta.update(seconds=round(job["finished"] - job["started"]), sec_per_it=job.get("sec_per_it"),
                created=time.strftime("%Y%m%d-%H%M%S"))
    job.update(state="done", output=name, stage="Done", eta=0)
    learn_speed(job)
    if job["private"]:
        job["meta"] = meta  # memory only; written if the user presses Keep
    else:
        write_sidecar(os.path.join(OUTPUTS, name), meta)


def worker():
    while True:
        WAKE.wait()
        if COOL["until"] > time.time() and not QUEUE["paused"]:
            time.sleep(1)  # cooling down; the next queued job waits
            continue
        with LOCK:
            nxt = None if QUEUE["paused"] else next((JOBS[i] for i in ORDER if JOBS[i]["state"] == "queued"), None)
            if nxt is None:
                WAKE.clear()
                continue
            nxt["state"] = "starting"
        try:
            run_image_job(nxt)
            if nxt["state"] == "done" and nxt["mode"] != "assist":
                plan_cooldown(nxt)
        except Cancelled:
            nxt.update(state="cancelled", stage="Cancelled")
        except Exception as e:  # keep the worker alive
            if nxt["state"] != "cancelled":
                nxt.update(state="error", error=str(e)[:300], stage="Failed")


def new_job(mode, private, **fields):
    job = {"id": uuid.uuid4().hex[:10], "mode": mode, "state": "queued", "private": private,
           "created": time.time(), "step": 0, "total": None, "prompt": "", "width": 0, "height": 0,
           "steps": 0, "seed": None, "refs": [], "ref_sizes": [], "turbo": False}
    job.update(fields)
    job["estimate"] = round(estimate_seconds(job))
    return job


def enqueue(jobs):
    with LOCK:
        for job in jobs:
            JOBS[job["id"]] = job
            ORDER.append(job["id"])
    WAKE.set()
    return [j["id"] for j in jobs]


MAX_PROMPTS, MAX_JOBS = 50, 100


def split_prompts(text):
    """Bulk text: prompts separated by one or more empty lines. Line breaks inside a prompt become spaces."""
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    return [" ".join(l.strip() for l in b.splitlines() if l.strip()) for b in blocks if b.strip()]


def submit(spec):
    mode = spec.get("mode", "gen")
    if mode in ("upscale", "assist"):
        raise ValueError("MOSS supports text-to-image generation only")
    if mode not in GEN_MODES:
        raise ValueError("Unknown mode")
    # An explicit Turbo request implies its BF16 SFT base; otherwise use the install default.
    model = spec.get("model") or (engine.SFT_MODEL if spec.get("turbo") else engine.default_model())
    if model not in engine.MODELS:
        raise ValueError("Unsupported MOSS model")
    if model != engine.int8_config.MODEL and not engine.bf16_available(model):
        raise ValueError("BF16 transformer for " + model.split("/")[-1] + " is not downloaded. Run ./download.ps1 -Model "
                         + ("SFT" if model == engine.SFT_MODEL else "Base") + " or select INT8.")
    int8 = model == engine.int8_config.MODEL
    if int8 and not engine.int8_config.available():
        raise ValueError("Experimental INT8 checkpoint missing or incomplete. Run ./download.ps1 -Model INT8.")
    int8_compute = spec.get('int8_compute', 'native' if engine.int8_config.native_available() else 'bf16') if int8 else None
    if int8 and int8_compute not in ('bf16', 'native'):
        raise ValueError('INT8 compute must be native or bf16')
    if int8_compute == 'native' and not engine.int8_config.native_available():
        raise ValueError('Native INT8 requires optional requirements-int8-native.txt; select BF16 decode instead')
    text = (spec.get("prompt") or "").strip()
    prompts = split_prompts(text) if spec.get("bulk") else ([text] if text else [])
    if not prompts:
        raise ValueError("Prompt is required")
    if len(prompts) > MAX_PROMPTS:
        raise ValueError("Up to %d prompts at once" % MAX_PROMPTS)
    batch = max(1, min(8, int(spec.get("batch", 1))))
    if len(prompts) * batch > MAX_JOBS:
        raise ValueError("Up to %d images at once (prompts x variations)" % MAX_JOBS)
    refs = [safe_name(r) for r in spec.get("refs", [])][:3]
    if mode == "edit" and not refs:
        raise ValueError("Add at least one image to edit")
    mask = safe_name(spec["mask"]) if spec.get("mask") and mode == "edit" else None
    seed_in = int(spec.get("seed", -1))  # a fixed seed is shared by every prompt, so prompts compare fairly
    turbo = bool(spec.get("turbo", model == engine.SFT_MODEL and engine.turbo_available()))
    if turbo:
        if model != engine.SFT_MODEL:
            raise ValueError("Turbo 4-step requires MOSS-Image-SFT; select SFT first")
        if not engine.turbo_available():
            raise ValueError("Turbo adapter missing or incomplete. Run ./download.ps1 -Model Turbo first.")
    # anything made from a private image stays private
    private = PRIVATE_MODE["on"] or bool(spec.get("private")) or any(is_private_file(r) for r in refs + [mask])
    ref_max = int(spec.get("ref_max", 1024))
    sizes = []
    for r in refs:
        w, h = image_size(ref_path(r))
        s = min(1.0, ref_max / float(max(w, h)))
        sizes.append((w * s, h * s))
    jobs = []
    for prompt in prompts:
        seed = seed_in if seed_in >= 0 else random.randint(0, 2**31 - 1)
        for i in range(batch):
            job = dict(
                model=model, prompt=prompt, negative=(spec.get("negative") or "").strip(),
                width=min(1024, snap16(int(spec.get("width", 512)))), height=min(1024, snap16(int(spec.get("height", 512)))),
                steps=max(1, min(100, int(spec.get("steps", 16)))), cfg=max(0, min(10, float(spec.get("cfg", 3.0)))),
                img_cfg=spec.get("img_cfg") and float(spec["img_cfg"]), seed=seed + i,
                sampler=spec.get("sampler") if spec.get("sampler") in SAMPLERS else "euler",
                scheduler=spec.get("scheduler") if spec.get("scheduler") in SCHEDULERS else "default",
                flow_shift=spec.get("flow_shift") and float(spec["flow_shift"]),
                fast=False, fast_threshold=float(spec.get("fast_threshold", 0.2)),
                preview=False, refs=refs, mask=mask,
                ref_max=ref_max, turbo=turbo,
                turbo_schedule="trained" if spec.get("turbo_schedule") == "trained" else "matched")
            if turbo:
                job.update(steps=4, cfg=1.0, img_cfg=None, negative="", sampler="rapid-moss", scheduler="dataspace",
                           flow_shift=None, fast=False)
            if int8:
                job.update(steps=8, cfg=1.0, img_cfg=None, negative="", sampler="euler",
                           scheduler="simple", flow_shift=5.0, fast=False, int8_compute=int8_compute)
            jobs.append(new_job(mode, private, ref_sizes=sizes, **job))
    return enqueue(jobs)


def cancel(job_id):
    job = JOBS.get(job_id)
    if not job or job["state"] in ("done", "error", "cancelled"):
        return
    was_running = job["state"] in ("running", "starting")
    job["_cancel_requested"] = True
    job.update(state="cancelled", stage="Cancelled")
    if was_running:
        if job.get("engine") == "warm":
            ENGINE.stop()   # Terminate the CUDA worker to release memory
        else:
            ENGINE.stop()


def move(job_id, direction):
    with LOCK:
        queued = [i for i in ORDER if JOBS[i]["state"] == "queued"]
        if job_id not in queued:
            return
        k = queued.index(job_id)
        j = k + (1 if direction > 0 else -1)
        if 0 <= j < len(queued):
            a, b = ORDER.index(queued[k]), ORDER.index(queued[j])
            ORDER[a], ORDER[b] = ORDER[b], ORDER[a]


# ---------------------------------------------------------------- gallery and private tray

def gallery():
    items = []
    for f in os.listdir(OUTPUTS):
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            p = os.path.join(OUTPUTS, f)
            items.append({"name": f, "mtime": os.path.getmtime(p), "meta": sidecar_of(f)})
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return items


def delete_image(name):
    """Remove an image and everything that records it: settings sidecar, every log that mentions
    it, its queue entry, and uploaded sources or masks no other gallery image still uses."""
    removed = []

    def rm(p):
        if os.path.exists(p):
            os.remove(p)
            removed.append(os.path.relpath(p, ROOT))

    img = os.path.join(OUTPUTS, name)
    meta = sidecar_of(name)
    uses = list(meta.get("refs") or []) + ([meta["mask"]] if meta.get("mask") else [])
    rm(img)
    rm(img.rsplit(".", 1)[0] + ".json")

    needle = name.encode()
    for f in os.listdir(LOGS):
        p = os.path.join(LOGS, f)
        try:
            with open(p, "rb") as fh:
                if needle in fh.read():
                    rm(p)
        except OSError:
            pass

    still_used = set()
    for g in gallery():
        still_used.update(g["meta"].get("refs") or [])
        if g["meta"].get("mask"):
            still_used.add(g["meta"]["mask"])
    for r in uses:
        if r not in still_used:
            rm(os.path.join(INPUTS, r))

    drop_jobs(lambda j: j.get("output") == name)
    return removed


def private_items():
    with LOCK:
        jobs = [JOBS[i] for i in ORDER if JOBS[i].get("private") and JOBS[i]["state"] == "done" and JOBS[i].get("output")]
    return [{"name": j["output"], "meta": j.get("meta", {})} for j in reversed(jobs)
            if os.path.exists(os.path.join(PRIVATE, j["output"]))]


def keep_private(name):
    """Move a private result into the gallery. Private sources it depends on are copied to inputs/
    so its before/after and history keep working after the private tray is wiped."""
    name = safe_name(name)
    shutil.move(os.path.join(PRIVATE, name), os.path.join(OUTPUTS, name))
    job = next((j for j in JOBS.values() if j.get("output") == name), None)
    meta = job.get("meta", {}) if job else {}
    for r in (meta.get("refs") or []) + ([meta["mask"]] if meta.get("mask") else []):
        if is_private_file(r) and not os.path.exists(os.path.join(INPUTS, r)):
            shutil.copy(os.path.join(PRIVATE, r), os.path.join(INPUTS, r))
    if job:
        job["private"] = False
    write_sidecar(os.path.join(OUTPUTS, name), meta)
    return name


def discard_private(name=None):
    if name:
        p = os.path.join(PRIVATE, safe_name(name))
        if os.path.exists(p):
            os.remove(p)
        drop_jobs(lambda j: j.get("output") == name)
        return
    wipe_private()
    drop_jobs(lambda j: j.get("private") and j["state"] not in ("queued", "running", "starting"))


def save_upload(data, filename, private=False, kind="image"):
    b64 = data.split(",", 1)[1]
    ext = os.path.splitext(filename or "")[1].lower() or ".img"
    raw_path = os.path.join(TMP, "upload_" + uuid.uuid4().hex[:8] + ext)
    with open(raw_path, "wb") as f:
        f.write(base64.b64decode(b64))
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.splitext(filename or kind)[0])[:40] or kind
    name = "%s_%s%s.png" % (time.strftime("%Y%m%d-%H%M%S"), "mask_" if kind == "mask" else "", stem)
    out = os.path.join(PRIVATE if private else INPUTS, name)
    try:
        ok = to_png(raw_path, out, max_side=2048)  # handles HEIC, JPEG, WebP, TIFF
    except Exception:
        ok = False
    os.remove(raw_path)
    if not ok:
        raise ValueError("Could not read that image format")
    w, h = image_size(out)
    return {"name": name, "width": w, "height": h, "private": private}


def zip_of(names):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for n in names:
            n = safe_name(n)
            p = os.path.join(OUTPUTS, n) if os.path.exists(os.path.join(OUTPUTS, n)) else os.path.join(PRIVATE, n)
            if os.path.exists(p):
                z.write(p, n)
    return buf.getvalue()


# ---------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send_bytes(self, body, ctype, code=200, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, obj, code=200):
        self.send_bytes(json.dumps(obj).encode(), "application/json", code)

    def send_file(self, path):
        if not os.path.isfile(path):
            return self.send_json({"error": "not found"}, 404)
        ext = path.rsplit(".", 1)[-1].lower()
        ctype = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp",
                 "html": "text/html; charset=utf-8"}.get(ext, "application/octet-stream")
        with open(path, "rb") as f:
            self.send_bytes(f.read(), ctype)

    def body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        try:
            if path in ("/", "/index.html"):
                return self.send_file(os.path.join(STUDIO, "index.html"))
            if path == "/api/state":
                with LOCK:
                    jobs = [public(JOBS[i]) for i in ORDER[-60:]]
                return self.send_json({
                    "jobs": jobs, "calib": load_calib(), "paused": QUEUE["paused"],
                    "cooling": max(0, round(COOL["until"] - time.time())), "cool_reason": COOL["reason"],
                    "private_mode": PRIVATE_MODE["on"], "engine_warm": ENGINE.running,
                    "turbo": engine.turbo_available(), "int8": engine.int8_config.available(), "bf16": engine.bf16_available(),
                    "default_model": engine.default_model(), "int8_native": engine.int8_config.native_available(), "upscaler": engine.upscaler_available(),
                    "assistant": bool(engine.llama_cli()), "settings": load_settings(),
                    "models": engine.MODELS, "samplers": ["euler"], "schedulers": ["default"]})
            if path == "/api/gallery":
                return self.send_json(gallery())
            if path == "/api/private":
                return self.send_json(private_items())
            if path == "/api/library":
                return self.send_json(load_library())
            if path.startswith("/outputs/"):
                return self.send_file(os.path.join(OUTPUTS, safe_name(path[9:])))
            if path.startswith("/private/"):
                return self.send_file(os.path.join(PRIVATE, safe_name(path[9:])))
            if path.startswith("/ref/"):
                return self.send_file(ref_path(safe_name(path[5:])))
            if path.startswith("/preview/"):
                return self.send_file(os.path.join(TMP, safe_name(path[9:]) + "_preview.png"))
        except ValueError:
            return self.send_json({"error": "bad path"}, 400)
        self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            b = self.body()
            if path == "/api/jobs":
                return self.send_json({"ids": submit(b)})
            if path == "/api/upload":
                private = PRIVATE_MODE["on"] or bool(b.get("private"))
                return self.send_json(save_upload(b["data"], b.get("filename"), private, b.get("kind", "image")))
            if path == "/api/private/mode":
                PRIVATE_MODE["on"] = bool(b.get("on"))
                return self.send_json({"on": PRIVATE_MODE["on"]})
            if path == "/api/private/keep":
                return self.send_json({"name": keep_private(b["name"])})
            if path == "/api/private/discard":
                discard_private(b.get("name"))
                return self.send_json({"ok": True})
            if path == "/api/queue/skip-cooldown":
                COOL["until"] = 0
                WAKE.set()
                return self.send_json({"ok": True})
            if path == "/api/queue/pause":
                QUEUE["paused"] = bool(b.get("paused"))
                WAKE.set()
                return self.send_json({"paused": QUEUE["paused"]})
            if path == "/api/clear":
                drop_jobs(lambda j: j["state"] in ("done", "error", "cancelled") and not j.get("private"))
                return self.send_json({"ok": True})
            if path == "/api/settings":
                s = load_settings()
                if "idle_minutes" in b:
                    s["idle_minutes"] = max(1, min(120, int(b["idle_minutes"])))
                if b.get("cooldown") in COOLDOWN_CHOICES:
                    s["cooldown"] = b["cooldown"]
                write_json(SETTINGS, s)
                apply_settings(s)
                return self.send_json(s)
            if path == "/api/engine/stop":
                ENGINE.stop()
                return self.send_json({"ok": True})
            if path == "/api/library":
                lib = load_library()
                if "presets" in b:
                    lib["presets"] = {str(k)[:40]: v for k, v in b["presets"].items()}
                if "favorites" in b:
                    lib["favorites"] = [str(p)[:2000] for p in b["favorites"]][:200]
                write_json(LIBRARY, lib)
                return self.send_json(lib)
            if path == "/api/zip":
                data = zip_of(b.get("names") or [])
                return self.send_bytes(data, "application/zip", extra={
                    "Content-Disposition": 'attachment; filename="moss-images-%s.zip"' % time.strftime("%Y%m%d-%H%M%S")})
            m = re.match(r"^/api/jobs/(\w+)/(cancel|up|down)$", path)
            if m:
                if m.group(2) == "cancel":
                    cancel(m.group(1))
                else:
                    move(m.group(1), -1 if m.group(2) == "up" else 1)
                return self.send_json({"ok": True})
        except (ValueError, KeyError) as e:
            return self.send_json({"error": str(e)}, 400)
        self.send_json({"error": "not found"}, 404)

    def do_DELETE(self):
        path = unquote(urlparse(self.path).path)
        if path.startswith("/api/gallery/"):
            try:
                name = safe_name(path[13:])
            except ValueError:
                return self.send_json({"error": "bad name"}, 400)
            return self.send_json({"removed": delete_image(name)})
        self.send_json({"error": "not found"}, 404)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860
    wipe_tmp()
    apply_settings(load_settings())
    threading.Thread(target=worker, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)

    def shutdown(*_):
        ENGINE.stop()
        wipe_private()
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print("Moss Studio running at http://127.0.0.1:%d  (Ctrl+C to stop)" % port, flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
