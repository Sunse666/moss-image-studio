"""Bounded, lossless local prompt tensors. Private requests never touch disk."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

from safetensors.torch import load_file, save_file
from safetensors import SafetensorError


def encoder_identity(model, encoder):
    """Fingerprint encoder weights/config, tokenizer and installed encoding code."""
    root = Path(model)
    files = []
    for folder in ("text_encoder", "tokenizer"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file():
                stat = path.stat()
                files.append((path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns))
    versions = {}
    for package in ("torch", "transformers", "weellm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    # Include local implementation source so editable installs invalidate too.
    import inspect
    source = inspect.getsource(encoder)
    return {"schema": 1, "model": str(root.resolve()), "files": files,
            "versions": versions, "encode_source": source, "dtype": "bfloat16"}


class PromptCache:
    def __init__(self, root, identity, *, private=False, max_bytes=128 * 2**20, max_entries=32):
        self.root = Path(root)
        self.identity = identity
        self.private = private
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.status = "private" if private else "miss"

    def key(self, prompt):
        value = json.dumps([self.identity, prompt], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def encode(self, prompt, encoder, device):
        if self.private:
            self.status = "private"
            return encoder(prompt)
        path = self.root / (self.key(prompt) + ".safetensors")
        try:
            if path.stat().st_size <= self.max_bytes:
                tensors = load_file(str(path), device="cpu")
                embeds, mask = tensors["embeds"], tensors["mask"]
                if (set(tensors) != {"embeds", "mask"} or embeds.ndim != 3 or
                        mask.ndim != 2 or embeds.shape[:2] != mask.shape or
                        not embeds.is_floating_point()):
                    raise ValueError("Invalid cached prompt tensors")
                result = embeds.to(device), mask.to(device)
                os.utime(path, None)
                self.status = "hit"
                return result
        except (OSError, ValueError, RuntimeError, KeyError, SafetensorError):
            pass
        self.status = "miss"
        embeds, mask = encoder(prompt)
        temporary = None
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self.root, suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
            save_file({"embeds": embeds.detach().cpu().contiguous(),
                       "mask": mask.detach().cpu().contiguous()}, str(temporary))
            if temporary.stat().st_size <= self.max_bytes:
                temporary.replace(path)
            entries = sorted(self.root.glob("*.safetensors"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
            total = 0
            for index, entry in enumerate(entries):
                total += entry.stat().st_size
                if index >= self.max_entries or total > self.max_bytes:
                    entry.unlink(missing_ok=True)
        except (OSError, ValueError, RuntimeError, SafetensorError):
            self.status = "miss (cache unavailable)"
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
        return embeds, mask
