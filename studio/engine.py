"""Isolated CUDA worker control. Cancellation releases the GPU process."""
import json
import os
import subprocess
import sys
import threading
import turbo_config
import int8_config

STUDIO = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(STUDIO)
MODELS = ["OpenMOSS-Team/MOSS-Image-SFT", "OpenMOSS-Team/MOSS-Image-Base", int8_config.MODEL]
SFT_MODEL = MODELS[0]  # BF16 SFT; the only base for the Rapid-MOSS Turbo adapter
DEFAULT_MODEL = SFT_MODEL  # fallback when the INT8 checkpoint is not installed


def bf16_available(model=SFT_MODEL):
    """True when every BF16 transformer shard listed in the model index exists.

    Lite installs keep the shared encoder/VAE but omit these shards.
    """
    folder = os.path.join(ROOT, "models", model.split("/")[-1], "transformer")
    try:
        names = os.listdir(folder)
        index = next((n for n in names if n.endswith(".safetensors.index.json")), None)
        if index:
            with open(os.path.join(folder, index), encoding="utf-8") as handle:
                files = set(json.load(handle)["weight_map"].values())
        else:
            files = {n for n in names if n.endswith(".safetensors")}
    except (OSError, ValueError, KeyError):
        return False
    return bool(files) and all(os.path.isfile(os.path.join(folder, n)) for n in files)


def default_model():
    # Native INT8 measured faster with less VRAM on the 4 GB laptop; installs
    # without it (full BF16 downloads) keep BF16 Turbo as before.
    return int8_config.MODEL if int8_config.available() else SFT_MODEL

class Cancelled(Exception):
    pass

def turbo_available():
    return turbo_config.available()

def upscaler_available():
    return False

def llama_cli():
    return None

class Engine:
    idle_seconds = 600
    def __init__(self):
        self.process = None
        self.lock = threading.Lock()

    @property
    def running(self):
        return self.process is not None and self.process.poll() is None

    def stop(self):
        with self.lock:
            process = self.process
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    def generate(self, job, out_path, log):
        with self.lock:
            if job["state"] == "cancelled" or job.get("_cancel_requested"):
                raise Cancelled()
            self.process = subprocess.Popen(
                [sys.executable, "-u", os.path.join(STUDIO, "moss_worker.py")],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", cwd=ROOT)
            process = self.process
        spec = {key: job[key] for key in ("model", "prompt", "negative", "width", "height", "steps", "cfg", "seed")}
        spec["turbo"] = bool(job.get("turbo"))
        spec["private"] = bool(job.get("private"))
        spec["int8_compute"] = job.get("int8_compute", "bf16")
        spec["output"] = out_path
        process.stdin.write(json.dumps(spec) + "\n")
        process.stdin.close()
        tail = []
        for line in process.stdout:
            if line.startswith("MOSS_EVENT "):
                job.update(json.loads(line[len("MOSS_EVENT "):]))
            else:
                tail.append(line.strip())
                tail = tail[-12:]
                if log:
                    log.write(line.encode("utf-8"))
                    log.flush()
        code = process.wait()
        if job["state"] == "cancelled":
            raise Cancelled()
        if code or not os.path.isfile(out_path):
            raise RuntimeError("MOSS worker failed. " + " ".join(tail)[-1800:])

ENGINE = Engine()
