"""Isolated CUDA worker control. Cancellation releases the GPU process."""
import json
import os
import subprocess
import sys
import threading
import time
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
    """Runs MOSS jobs in a worker process; optionally keeps it warm between jobs.

    Warm reuse is limited to native INT8 (the verified profile). Cancel, errors,
    a different profile or warm_seconds of inactivity stop the process, which
    releases all of its GPU and RAM.
    """
    worker_script = os.path.join(STUDIO, "moss_worker.py")
    idle_seconds = 600   # legacy setting; unused
    warm_seconds = 300   # 0 = one-shot worker per job

    def __init__(self):
        self.process = None
        self.profile = None
        self.busy = False
        self.last_used = 0.0
        self.lock = threading.Lock()
        threading.Thread(target=self._reap_idle, daemon=True).start()

    @property
    def running(self):
        return self.process is not None and self.process.poll() is None

    @staticmethod
    def job_profile(job):
        int8 = job["model"] == int8_config.MODEL
        return (job["model"], job.get("int8_compute", "bf16") if int8 else None, bool(job.get("turbo")))

    def keeps_warm(self, job):
        model, compute, turbo = self.job_profile(job)
        return self.warm_seconds > 0 and model == int8_config.MODEL and compute == "native" and not turbo

    def will_reuse(self, job):
        return self.running and not self.busy and self.profile == self.job_profile(job)

    def status(self):
        if not self.running:
            return {"state": "cold", "idle_left": 0}
        if self.busy:
            return {"state": "busy", "idle_left": 0}
        left = max(0, round(self.last_used + self.warm_seconds - time.time()))
        return {"state": "warm", "idle_left": left}

    def stop(self):
        with self.lock:
            process = self.process
            self.process, self.profile = None, None
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    def _reap_idle(self):
        while True:
            time.sleep(2)
            if self.running and not self.busy and time.time() - self.last_used > self.warm_seconds:
                self.stop()

    def generate(self, job, out_path, log):
        with self.lock:
            if job["state"] == "cancelled" or job.get("_cancel_requested"):
                raise Cancelled()
        warm = self.keeps_warm(job)
        with self.lock:
            reuse = self.running and self.profile == self.job_profile(job)
            self.busy = True
        kept = False
        try:
            if not reuse:
                self.stop()
                with self.lock:
                    self.process = subprocess.Popen(
                        [sys.executable, "-u", self.worker_script],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, encoding="utf-8", errors="replace", cwd=ROOT)
                    self.profile = self.job_profile(job) if warm else None
            process = self.process
            job["engine"] = "warm" if reuse else "cold"
            spec = {key: job[key] for key in ("model", "prompt", "negative", "width", "height", "steps", "cfg", "seed")}
            spec["turbo"] = bool(job.get("turbo"))
            spec["private"] = bool(job.get("private"))
            spec["int8_compute"] = job.get("int8_compute", "bf16")
            spec["output"] = out_path
            spec["keep_warm"] = warm
            process.stdin.write(json.dumps(spec) + "\n")
            process.stdin.flush()
            if not warm:
                process.stdin.close()
            tail, done = [], False
            for line in process.stdout:
                if line.startswith("MOSS_EVENT "):
                    values = json.loads(line[len("MOSS_EVENT "):])
                    if values.pop("job_done", False):
                        done = True  # warm worker finished this job and stays loaded
                        break
                    job.update(values)
                else:
                    tail.append(line.strip())
                    tail = tail[-12:]
                    if log:
                        log.write(line.encode("utf-8"))
                        log.flush()
            if job["state"] == "cancelled":
                raise Cancelled()
            if not done:
                code = process.wait()
                if code or not os.path.isfile(out_path):
                    raise RuntimeError("MOSS worker failed. " + " ".join(tail)[-1800:])
            elif not os.path.isfile(out_path):
                raise RuntimeError("MOSS worker reported success without an image")
            kept = done
        finally:
            if not kept:
                self.stop()  # one-shot, error or cancel: never reuse this process
            self.last_used = time.time()
            self.busy = False


ENGINE = Engine()
