"""Pinned Rapid-MOSS release metadata; safe to import without torch/CUDA."""
from pathlib import Path

REPO = "darask0/MOSS-Image_4-step_adapter_lora"
REVISION = "1ec327b9fac5a0aa47426aaba6565892616ce30e"
DIRECTORY = Path(__file__).resolve().parents[1] / "models" / "rapid-moss-4step"
FILES = {
    "lora.safetensors": (1189176832, "4a4c8e2d6244a0fcdc2822b87b3435226ae57c289aef99fc2ac54edeab9e09bb"),
    "time_embedding.safetensors": (41017896, "2ebabc91e2a9e68c00e948dd80b05235a826e808817b5b2600b441457570d244"),
    "dataspace.safetensors": (65784, "d19f0e695bb26886cfce57441fd412f358e159d6eec8555ec1a71daaf30a7968"),
}


def available(root=DIRECTORY):
    root = Path(root)
    try:
        return ((root / ".download-complete").read_text().strip() == REVISION
                and all((root / name).stat().st_size == size for name, (size, _) in FILES.items()))
    except OSError:
        return False


def require_adapter(root=DIRECTORY):
    if not available(root):
        raise RuntimeError("Turbo adapter missing or incomplete. Run ./download.ps1 -Model Turbo first.")
